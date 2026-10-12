"""Official SAM 2.1 decoder-only adaptation from bbox-prompted teacher masks.

The bounding boxes are prompts, never segmentation targets.  Pseudo masks are
generated only for the train split and remain explicitly marked as teacher
predictions rather than ground truth.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import numpy as np


SCHEMA = "urap.frame_manifest.v1"
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/sam2"
OFFICIAL_TINY_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
OFFICIAL_LARGE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
TINY_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
LARGE_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _finite_xyxy(value, width=None, height=None):
    box = [float(x) for x in value]
    if len(box) != 4 or not all(math.isfinite(x) for x in box):
        raise ValueError("bbox must contain four finite xyxy values")
    x1, y1, x2, y2 = box
    if not (x1 < x2 and y1 < y2):
        raise ValueError("bbox is degenerate")
    if width is not None and not (0 <= x1 < x2 <= float(width)):
        raise ValueError("bbox x coordinates are outside the image")
    if height is not None and not (0 <= y1 < y2 <= float(height)):
        raise ValueError("bbox y coordinates are outside the image")
    return box


def read_manifest(path):
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else payload.get("records", [])
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest contains no frame rows")
    return rows


def guard_sequence_splits(rows: Iterable[dict]):
    seen = {}
    for row in rows:
        sequence = str(row.get("sequence_id") or row.get("source_video_id") or "")
        split = str(row.get("split") or "")
        if not sequence or split not in {"train", "val", "test"}:
            raise ValueError("every row needs sequence_id/source_video_id and train/val/test split")
        previous = seen.setdefault(sequence, split)
        if previous != split:
            raise ValueError(f"sequence {sequence!r} occurs in multiple splits: {previous}, {split}")
    return seen


def _resolve_image_ref(row, manifest_dir):
    value = row.get("frame_path") or row.get("rgb_path") or row.get("image_uri") or row.get("source_video_path")
    if not value:
        raise ValueError(f"sample {row.get('sample_id')} has no frame_path/rgb_path/image_uri/source_video_path")
    value = str(value)
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        return value
    path = Path(parsed.path if parsed.scheme == "file" else value)
    return str(path if path.is_absolute() else (Path(manifest_dir) / path).resolve())


def _boxes(row):
    if row.get("boxes") is not None:
        for box in row["boxes"]:
            yield box.get("bbox_xyxy_px") or box.get("xyxy") or box.get("bbox_xyxy"), box
    elif row.get("objects") is not None:
        for box in row["objects"]:
            yield box.get("xyxy") or box.get("bbox_xyxy_px"), box
    elif row.get("bbox_xyxy") is not None:  # legacy materializer
        values = row["bbox_xyxy"]
        if values and isinstance(values[0], (list, tuple)):
            for index, value in enumerate(values):
                yield value, {"class_id": (row.get("class_id") or [0] * len(values))[index]}
        else:
            yield values, row


def normalize_manifest_rows(rows, manifest_dir):
    guard_sequence_splits(rows)
    records = []
    for row in rows:
        if row.get("record_type", "frame") != "frame":
            continue
        if row.get("schema_version") not in {None, SCHEMA}:
            raise ValueError(f"unsupported schema_version {row.get('schema_version')!r}")
        if row.get("annotation_state", "positive") != "positive":
            continue
        width, height = row.get("width") or row.get("source_width"), row.get("height") or row.get("source_height")
        image_ref = _resolve_image_ref(row, manifest_dir)
        for box_index, (xyxy, box_meta) in enumerate(_boxes(row)):
            if xyxy is None:
                raise ValueError(f"sample {row.get('sample_id')} has a box without xyxy")
            records.append({
                "sample_id": str(row.get("sample_id") or f"{row.get('sequence_id')}-{row.get('frame_id')}-{box_index}"),
                "box_index": box_index,
                "dataset": str(row.get("dataset") or "unknown"),
                "domain": str(row.get("source_domain") or row.get("dataset") or "unknown"),
                "sequence_id": str(row.get("sequence_id") or row.get("source_video_id")),
                "frame_id": int(row.get("frame_id", row.get("source_frame_id", 0))),
                "split": str(row["split"]), "image_ref": image_ref,
                "image_source_type": "video_frame" if row.get("source_video_path") and not (row.get("frame_path") or row.get("rgb_path")) else "image",
                "width": None if width is None else int(width), "height": None if height is None else int(height),
                "xyxy": _finite_xyxy(xyxy, width, height), "box_metadata": box_meta,
                "boxes_complete_for_frame": row.get("boxes_complete_for_frame", "unknown"),
                "evaluation_eligibility": (row.get("evaluation_eligibility") or
                                           ("reviewed" if row.get("boxes_complete_for_frame") is True else "provisional_only")),
                "supervision": "bbox_for_teacher_prompt_only",
            })
    if not records:
        raise ValueError("manifest contains no positive boxes")
    return records


def select_training_records(records, maximum=0, order="balanced"):
    train = [x for x in records if x["split"] == "train"]
    limit = len(train) if int(maximum) <= 0 else min(len(train), int(maximum))
    if order == "manifest":
        return train[:limit]
    if order == "sequence_frame":
        key = lambda x: (str(x.get("domain", "")), str(x.get("sequence_id", "")),
                         int(x.get("frame_id", 0)), int(x.get("box_index", 0)), str(x.get("sample_id", "")))
        return sorted(train, key=key)[:limit]
    if order != "balanced":
        raise ValueError(f"unsupported selection order {order!r}")
    # Round-robin domain/sequence ordering avoids a large domain monopolising a cap.
    buckets = {}
    for record in train:
        buckets.setdefault((record["domain"], record["sequence_id"]), deque()).append(record)
    selected = []
    keys = sorted(buckets)
    while keys and len(selected) < limit:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].popleft())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def pseudo_cache_key(record, teacher_checkpoint_sha256, *, min_crop=256, box_scale=6.0):
    payload = {"sample_id": record["sample_id"], "box_index": int(record["box_index"]),
               "xyxy": [float(x) for x in record["xyxy"]], "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
               "min_crop": int(min_crop), "box_scale": float(box_scale)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def deadline_exceeded(prior_elapsed, *, started_at, max_seconds, now=None):
    if not max_seconds: return False
    current = time.monotonic() if now is None else float(now)
    return float(prior_elapsed) + current - float(started_at) >= float(max_seconds)


def context_crop_xyxy(box, image_wh, min_crop=256, box_scale=6.0):
    width, height = map(int, image_wh)
    x1, y1, x2, y2 = _finite_xyxy(box, width, height)
    crop_w = min(width, max(float(min_crop), (x2 - x1) * float(box_scale)))
    crop_h = min(height, max(float(min_crop), (y2 - y1) * float(box_scale)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(math.floor(min(max(0.0, cx - crop_w / 2), width - crop_w)))
    top = int(math.floor(min(max(0.0, cy - crop_h / 2), height - crop_h)))
    right, bottom = min(width, int(math.ceil(left + crop_w))), min(height, int(math.ceil(top + crop_h)))
    left, top = max(0, right - int(math.ceil(crop_w))), max(0, bottom - int(math.ceil(crop_h)))
    return [left, top, right, bottom]


def box_to_crop(box, crop):
    x1, y1, x2, y2 = map(float, box)
    left, top, right, bottom = map(float, crop)
    return [max(0.0, x1-left), max(0.0, y1-top), min(right-left, x2-left), min(bottom-top, y2-top)]


def jitter_box(box, image_wh, *, fraction=0.05, seed=0):
    """Independently jitter box sides, clip them, and preserve positive area."""
    width, height = map(float, image_wh)
    x1, y1, x2, y2 = _finite_xyxy(box, width, height)
    rng = np.random.default_rng(seed)
    dx, dy = (x2 - x1) * fraction, (y2 - y1) * fraction
    candidate = [x1 + rng.uniform(-dx, dx), y1 + rng.uniform(-dy, dy),
                 x2 + rng.uniform(-dx, dx), y2 + rng.uniform(-dy, dy)]
    candidate = [max(0.0, min(width, candidate[0])), max(0.0, min(height, candidate[1])),
                 max(0.0, min(width, candidate[2])), max(0.0, min(height, candidate[3]))]
    return _finite_xyxy(candidate, width, height)


def assess_pseudo_mask(mask, prompt_xyxy, teacher_score, *, min_score=0.70, min_pixels=6, max_outside_fraction=0.35):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("pseudo mask must be a 2-D array")
    x1, y1, x2, y2 = [int(round(x)) for x in prompt_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
    prompt = np.zeros_like(mask); prompt[y1:y2, x1:x2] = True
    pixels = int(mask.sum()); inside = int((mask & prompt).sum())
    outside_fraction = 1.0 if pixels == 0 else 1.0 - inside / pixels
    reasons = []
    if not math.isfinite(float(teacher_score)) or float(teacher_score) < min_score: reasons.append("teacher_mask_quality_below_threshold")
    if pixels < min_pixels: reasons.append("empty_or_too_small")
    if pixels and outside_fraction > max_outside_fraction: reasons.append("excess_outside_prompt")
    if pixels and np.array_equal(mask, prompt): reasons.append("rectangular_prompt_fill")
    return {
        "accepted": not reasons, "reasons": reasons, "mask_pixels": pixels,
        "outside_prompt_fraction": outside_fraction, "teacher_predicted_mask_quality": float(teacher_score),
        "score_semantics": "teacher_predicted_mask_quality_not_drone_confidence",
        "supervision": "teacher_pseudo_mask_not_ground_truth",
    }


def freeze_for_decoder_only(model):
    for parameter in model.parameters(): parameter.requires_grad_(False)
    decoder = getattr(model, "sam_mask_decoder", None)
    if decoder is None: raise ValueError("official SAM2 model has no sam_mask_decoder")
    for parameter in decoder.parameters(): parameter.requires_grad_(True)
    trainable = sorted(name for name, p in model.named_parameters() if p.requires_grad)
    if not trainable or any(not name.startswith("sam_mask_decoder.") for name in trainable):
        raise RuntimeError("decoder-only freeze invariant failed")
    return {"policy": "decoder_only", "trainable_names": trainable,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters())}


def audit_decoder_gradients(model):
    decoder_grads = sorted(name for name, p in model.named_parameters()
                           if name.startswith("sam_mask_decoder.") and p.grad is not None)
    frozen_grads = sorted(name for name, p in model.named_parameters()
                          if not name.startswith("sam_mask_decoder.") and p.grad is not None)
    if not decoder_grads:
        raise RuntimeError("no decoder gradient; refusing false-success training")
    if frozen_grads:
        raise RuntimeError(f"frozen parameters received gradients: {frozen_grads[:5]}")
    return {"decoder_gradient_parameter_count": len(decoder_grads),
            "decoder_gradient_names": decoder_grads, "frozen_gradient_names": frozen_grads}


def pseudo_mask_loss(mask_logits, pseudo_masks, predicted_iou):
    import torch
    import torch.nn.functional as F
    target = pseudo_masks.float()
    if target.shape != mask_logits.shape:
        target = F.interpolate(target, size=mask_logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(mask_logits, target)
    probs = mask_logits.sigmoid()
    dice = 1 - ((2 * (probs * target).sum((2, 3)) + 1) / (probs.sum((2, 3)) + target.sum((2, 3)) + 1)).mean()
    binary = mask_logits.detach() > 0
    truth = target > 0.5
    actual = ((binary & truth).sum((2, 3)).float() / (binary | truth).sum((2, 3)).clamp_min(1).float())
    score = predicted_iou
    if score.ndim == 1: score = score[:, None]
    if score.shape[1] != actual.shape[1]: score = score[:, :actual.shape[1]]
    iou_loss = F.l1_loss(score, actual)
    loss = bce + dice + iou_loss
    return loss, {"bce": float(bce.detach()), "dice": float(dice.detach()),
                  "iou_score_l1": float(iou_loss.detach()), "actual_pseudo_iou": float(actual.mean())}


def _load_rgb(path):
    from PIL import Image
    parsed = urlparse(str(path))
    if parsed.scheme not in {"", "file"}: raise ValueError("remote image_uri must be materialized to frame_path before SAM adaptation")
    with Image.open(parsed.path if parsed.scheme == "file" else path) as image:
        return np.asarray(image.convert("RGB"))


def _load_record_rgb(record):
    if record.get("image_source_type") != "video_frame":
        return _load_rgb(record["image_ref"])
    import cv2
    capture = cv2.VideoCapture(str(record["image_ref"]))
    try:
        frame_id = int(record["frame_id"])
        if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id):
            raise RuntimeError(f"cannot seek video frame {frame_id}: {record['image_ref']}")
        ok, bgr = capture.read()
        actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        if not ok or bgr is None or actual != frame_id:
            raise RuntimeError(f"exact video decode failed requested={frame_id} actual={actual}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    os.replace(temporary, path)


def recover_jsonl(path):
    """Recover a durable append journal, tolerating only a torn final line."""
    path = Path(path)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"corrupt JSONL before final line: {path}")
            _write_jsonl(path, rows)
            return rows
    return rows


def _append_jsonl(stream, row, durable=False):
    stream.write(json.dumps(row, sort_keys=True) + "\n")
    stream.flush()
    if durable:
        os.fsync(stream.fileno())


def coverage_progress(total, processed, accepted, rejected, elapsed_seconds):
    total, processed, accepted, rejected = map(int, (total, processed, accepted, rejected))
    if min(total, processed, accepted, rejected) < 0 or accepted + rejected != processed or processed > total:
        raise ValueError("invalid unique-record coverage counters")
    return {"unique_records_total": total, "unique_records_processed": processed,
            "unique_records_remaining": total - processed, "accepted": accepted, "rejected": rejected,
            "coverage_fraction": 1.0 if total == 0 else processed / total,
            "elapsed_seconds": float(elapsed_seconds)}


def load_teacher_coverage(pseudo_manifest):
    """Load the teacher-run coverage that produced a (possibly partial) journal."""
    path = Path(pseudo_manifest).resolve().with_name("pseudo_progress.json")
    if not path.is_file():
        return {"available": False, "complete": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    coverage = coverage_progress(payload["unique_records_total"], payload["unique_records_processed"],
                                 payload["accepted"], payload["rejected"], payload["elapsed_seconds"])
    coverage["stopped_by_max_seconds"] = bool(payload.get("stopped_by_max_seconds", False))
    coverage["complete"] = coverage["unique_records_processed"] == coverage["unique_records_total"]
    return coverage


def repair_training_log(path, checkpoint_global_step):
    path = Path(path)
    rows = recover_jsonl(path)
    kept = [row for row in rows if int(row.get("global_step", -1)) <= int(checkpoint_global_step)]
    if len(kept) != len(rows):
        _write_jsonl(path, kept)
    return kept


def _write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _teacher_predictor(config, checkpoint, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(config, checkpoint, device=device, mode="eval"))


def generate_pseudo_masks(records, output_dir, teacher_config, teacher_checkpoint, device="cuda", min_score=0.70,
                          min_crop=256, box_scale=6.0, max_seconds=0, flush_every=25):
    from PIL import Image
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    teacher_sha = file_sha256(teacher_checkpoint); predictor = None
    manifest_path = output / "pseudo_manifest.jsonl"
    rows = recover_jsonl(manifest_path)
    existing = {}
    for row in rows:
        key = row.get("pseudo_cache_key")
        if not key or (key in existing and existing[key] != row):
            raise ValueError("pseudo journal has a missing or conflicting cache key")
        existing[key] = row
    requested_keys = [pseudo_cache_key(x, teacher_sha, min_crop=min_crop, box_scale=box_scale) for x in records]
    if len(requested_keys) != len(set(requested_keys)):
        raise ValueError("selected records contain duplicate pseudo cache keys")
    progress_path = output / "pseudo_progress.json"; prior_elapsed = 0.0
    if progress_path.is_file(): prior_elapsed = float(json.loads(progress_path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
    started_at = time.monotonic(); stopped = False; processed = accepted = rejected = 0
    journal = manifest_path.open("a", encoding="utf-8")
    for index, (record, identity) in enumerate(zip(records, requested_keys)):
        if deadline_exceeded(prior_elapsed, started_at=started_at, max_seconds=max_seconds): stopped = True; break
        if identity in existing:
            cached = existing[identity]; processed += 1
            accepted += cached.get("status") == "accepted"; rejected += cached.get("status") == "rejected"
            continue
        cache_json = output / f"{identity}.json"
        if cache_json.is_file():
            cached = json.loads(cache_json.read_text(encoding="utf-8"))
            mask_path = cached.get("pseudo_mask_path")
            if cached.get("status") == "rejected" or (mask_path and Path(mask_path).is_file()):
                rows.append(cached); existing[identity] = cached; _append_jsonl(journal, cached)
                processed += 1; accepted += cached["status"] == "accepted"; rejected += cached["status"] == "rejected"
                print(f"pseudo {index+1}/{len(records)} cached_{cached['status']}", flush=True); continue
        if predictor is None: predictor = _teacher_predictor(teacher_config, teacher_checkpoint, device)
        rgb = _load_record_rgb(record); height, width = rgb.shape[:2]
        crop = context_crop_xyxy(record["xyxy"], (width, height), min_crop=min_crop, box_scale=box_scale)
        left, top, right, bottom = crop
        crop_rgb = rgb[top:bottom, left:right]; local_box = box_to_crop(record["xyxy"], crop)
        predictor.set_image(crop_rgb)
        masks, scores, _ = predictor.predict(box=np.asarray(local_box, np.float32), multimask_output=True)
        chosen = None
        for candidate in np.argsort(np.asarray(scores))[::-1]:
            mask = np.asarray(masks[int(candidate)], bool)
            quality = assess_pseudo_mask(mask, local_box, float(scores[int(candidate)]), min_score=min_score)
            if quality["accepted"]: chosen = (mask, quality); break
        row = {**record, "crop_xyxy": crop, "prompt_crop_xyxy": local_box,
               "teacher": {"repository": OFFICIAL_REPOSITORY, "commit": SAM2_COMMIT,
                           "config": teacher_config, "checkpoint": str(Path(teacher_checkpoint).resolve()),
                           "checkpoint_sha256": teacher_sha, "model_role": "official_sam2.1_large_teacher"},
               "pseudo_cache_key": identity, "crop_policy": {"min_crop": min_crop, "box_scale": box_scale}}
        if chosen is None:
            row.update({"status": "rejected", "pseudo_mask_path": None,
                        "quality": assess_pseudo_mask(np.zeros(crop_rgb.shape[:2], bool), local_box, float(np.max(scores)), min_score=min_score)})
        else:
            mask, quality = chosen; mask_path = output / f"{identity}.png"
            mask_tmp = mask_path.with_name(mask_path.name + f".tmp.{os.getpid()}")
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_tmp, format="PNG"); os.replace(mask_tmp, mask_path)
            row.update({"status": "accepted", "pseudo_mask_path": str(mask_path), "quality": quality})
        cache_tmp = cache_json.with_name(cache_json.name + f".tmp.{os.getpid()}")
        cache_tmp.write_text(json.dumps(row, sort_keys=True), encoding="utf-8"); os.replace(cache_tmp, cache_json)
        rows.append(row); existing[identity] = row
        processed += 1; accepted += row["status"] == "accepted"; rejected += row["status"] == "rejected"
        _append_jsonl(journal, row, durable=bool(flush_every and processed % flush_every == 0))
        if not flush_every or processed % flush_every == 0:
            progress = coverage_progress(len(records), processed, accepted, rejected,
                                         prior_elapsed + time.monotonic() - started_at)
            _write_json(progress_path, {**progress, "stopped_by_max_seconds": False})
        print(f"pseudo {index+1}/{len(records)} {row['status']}", flush=True)
    journal.flush(); os.fsync(journal.fileno()); journal.close()
    if predictor is not None: release_gpu(predictor)
    progress = coverage_progress(len(records), processed, accepted, rejected,
                                 prior_elapsed + time.monotonic() - started_at)
    _write_json(progress_path, {**progress, "stopped_by_max_seconds": stopped})
    return rows


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""): digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload, path):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def make_training_state(model, optimizer, *, epoch, next_step, global_step, elapsed_seconds, **metadata):
    decoder = getattr(model, "sam_mask_decoder", None)
    return {"state_format": "urap.sam_decoder_checkpoint.v1",
            "weights_scope": "sam_mask_decoder" if decoder is not None else "full_model",
            "model": (decoder if decoder is not None else model).state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": int(epoch), "next_step": int(next_step),
            "global_step": int(global_step), "elapsed_seconds": float(elapsed_seconds),
            "rng_state": _rng_state(), **metadata}


def restore_training_state(path, model, optimizer):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    target = getattr(model, "sam_mask_decoder", None) if state["weights_scope"] == "sam_mask_decoder" else model
    if target is None: raise ValueError("checkpoint requires a sam_mask_decoder")
    target.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
    rng = state["rng_state"]; random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng["torch_cuda"] is not None: torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return {key: state[key] for key in ("epoch", "next_step", "global_step", "elapsed_seconds")}


def install_stop_handlers(stop_state):
    def request_stop(signum, frame):
        stop_state["requested"] = True; stop_state["signal"] = int(signum)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGINT"): signal.signal(signal.SIGINT, request_stop)


def release_gpu(owner):
    """Move a model (or predictor-wrapped model) off GPU and empty CUDA cache."""
    model = getattr(owner, "model", owner)
    if hasattr(model, "to"):
        model.to("cpu")
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass


def _student_forward(model, image, box):
    import torch
    expected_side = int(model.image_size)
    if expected_side != 1024 or tuple(image.shape[-2:]) != (expected_side, expected_side):
        raise ValueError("official SAM2.1 Tiny decoder path requires a 1024x1024 input")
    with torch.no_grad():
        backbone = model.forward_image(image)
        _, feats, pos, sizes = model._prepare_backbone_features(backbone)
        # Clones escape inference-mode storage and remain constants for decoder training.
        feats = [x.detach().clone() for x in feats]
    if len(feats) != 3 or len(sizes) != 3:
        raise RuntimeError(f"expected three SAM2 FPN levels, got feats={len(feats)} sizes={len(sizes)}")
    image_embed = feats[-1].permute(1, 2, 0).reshape(image.shape[0], -1, *sizes[-1])
    high_res = [x.permute(1, 2, 0).reshape(image.shape[0], -1, *size) for x, size in zip(feats[:-1], sizes[:-1])]
    if len(high_res) != 2 or tuple(image_embed.shape[-2:]) != (model.sam_image_embedding_size,) * 2:
        raise RuntimeError("SAM2 decoder feature pyramid has unexpected shapes")
    coords = torch.tensor([[box[:2], box[2:]]], device=image.device, dtype=torch.float32)
    labels = torch.tensor([[2, 3]], device=image.device, dtype=torch.int32)
    outputs = model._forward_sam_heads(image_embed, {"point_coords": coords, "point_labels": labels},
                                       high_res_features=high_res, multimask_output=False)
    return outputs[1], outputs[2]


def train_decoder(pseudo_manifest, output_dir, student_config, student_checkpoint, *, device="cuda", epochs=5,
                  learning_rate=1e-5, max_seconds=0, resume=None, checkpoint_every_steps=25):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from sam2.build_sam import build_sam2
    manifest_rows = recover_jsonl(pseudo_manifest)
    teacher_coverage = load_teacher_coverage(pseudo_manifest)
    if any(x.get("status") == "accepted" and x.get("split") != "train" for x in manifest_rows):
        raise ValueError("accepted non-train pseudo mask found; refusing split leakage")
    rows = [x for x in manifest_rows if x.get("status") == "accepted"]
    identities = [x.get("pseudo_cache_key") for x in rows]
    if any(not x for x in identities) or len(identities) != len(set(identities)):
        raise ValueError("accepted pseudo masks need unique pseudo_cache_key values")
    if not rows: raise ValueError("no accepted train pseudo masks")
    model = build_sam2(student_config, student_checkpoint, device=device, mode="train")
    audit = freeze_for_decoder_only(model); model.train()
    for module_name in ("image_encoder", "sam_prompt_encoder", "memory_encoder", "memory_attention"):
        module = getattr(model, module_name, None)
        if module is not None: module.eval()
    if int(model.image_size) != 1024:
        raise ValueError(f"expected official SAM2.1 Tiny image_size=1024, got {model.image_size}")
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    student_sha = file_sha256(student_checkpoint)
    pseudo_manifest_sha = file_sha256(pseudo_manifest)
    audit["student_provenance"] = {"repository": OFFICIAL_REPOSITORY, "commit": SAM2_COMMIT,
                                   "config": student_config, "checkpoint": str(Path(student_checkpoint).resolve()),
                                   "checkpoint_sha256": student_sha}
    (output / "trainable_params.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.01)
    start = time.monotonic(); best = float("inf")
    progress = {"epoch": 0, "next_step": 0, "global_step": 0, "elapsed_seconds": 0.0}
    if resume:
        resume_payload = torch.load(resume, map_location="cpu", weights_only=False)
        if resume_payload.get("student_checkpoint_sha256") != student_sha:
            raise ValueError("resume checkpoint was created from a different student checkpoint")
        if resume_payload.get("pseudo_manifest_sha256") != pseudo_manifest_sha:
            raise ValueError("resume checkpoint was created from a different pseudo manifest")
        progress = restore_training_state(resume, model, optimizer)
        best = float(resume_payload.get("best_training_loss", best))
    log_path = output / "train_log.jsonl"
    if resume:
        repair_training_log(log_path, progress["global_step"])
    elif log_path.exists() and recover_jsonl(log_path):
        raise ValueError("training log exists; use --resume or a new output directory")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    stopped = False; stop_state = {"requested": False, "signal": None}; install_stop_handlers(stop_state)
    global_step = int(progress["global_step"]); prior_elapsed = float(progress["elapsed_seconds"])
    log_stream = log_path.open("a", encoding="utf-8")

    def elapsed(): return prior_elapsed + time.monotonic() - start

    def checkpoint(epoch_value, next_step_value, stopped_value, epoch_loss_value=0.0, epoch_count_value=0):
        state = make_training_state(model, optimizer, epoch=epoch_value, next_step=next_step_value,
                                    global_step=global_step, elapsed_seconds=elapsed(),
                                    best_training_loss=best, sam2_commit=SAM2_COMMIT,
                                    student_checkpoint_sha256=student_sha,
                                    pseudo_manifest_sha256=pseudo_manifest_sha,
                                    supervision="teacher_pseudo_mask_not_ground_truth",
                                    teacher_pseudo_coverage=teacher_coverage,
                                    best_metric="complete_epoch_training_loss_against_teacher_pseudo_masks_not_mask_gt",
                                    epoch_accumulated_loss=float(epoch_loss_value),
                                    epoch_accumulated_steps=int(epoch_count_value),
                                    stopped=stopped_value, termination_signal=stop_state["signal"])
        atomic_torch_save(state, output / "last.pt")
        log_stream.flush(); os.fsync(log_stream.fileno())
        planned_steps = int(epochs) * len(rows)
        _write_json(output / "training_progress.json", {
            "accepted_unique_records": len(rows), "epoch": int(epoch_value), "next_step": int(next_step_value),
            "global_step": global_step, "planned_steps": planned_steps,
            "completion_fraction": 1.0 if planned_steps == 0 else global_step / planned_steps,
            "elapsed_seconds": elapsed(), "stopped": bool(stopped_value),
            "teacher_pseudo_coverage": teacher_coverage,
            "supervision": "teacher_pseudo_mask_not_ground_truth"})

    for epoch in range(int(progress["epoch"]), epochs):
        first_step = int(progress["next_step"]) if epoch == int(progress["epoch"]) else 0
        if resume and epoch == int(progress["epoch"]):
            epoch_loss = float(resume_payload.get("epoch_accumulated_loss", 0.0))
            completed = int(resume_payload.get("epoch_accumulated_steps", 0))
        else:
            epoch_loss = 0.0; completed = 0
        for step in range(first_step, len(rows)):
            row = rows[step]
            if stop_state["requested"] or (max_seconds and elapsed() >= max_seconds): stopped = True; break
            rgb = _load_record_rgb(row); l, t, r, b = row["crop_xyxy"]; rgb = rgb[t:b, l:r]
            mask = np.asarray(Image.open(row["pseudo_mask_path"]).convert("L")) > 0
            side = int(model.image_size)
            image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(device) / 255
            old_h, old_w = image.shape[-2:]
            image = F.interpolate(image[None], (side, side), mode="bilinear", align_corners=False)
            image = (image - mean) / std
            target = F.interpolate(torch.from_numpy(mask.copy())[None, None].float().to(device), (side, side), mode="nearest")
            sx, sy = side / old_w, side / old_h
            prompt = jitter_box(row["prompt_crop_xyxy"], (old_w, old_h), seed=epoch * len(rows) + step)
            box = [prompt[0]*sx, prompt[1]*sy, prompt[2]*sx, prompt[3]*sy]
            optimizer.zero_grad(set_to_none=True)
            logits, ious = _student_forward(model, image, box)
            if logits.shape[1] > 1: logits, ious = logits[:, :1], ious[:, :1]
            loss, metrics = pseudo_mask_loss(logits, target, ious)
            loss.backward()
            if global_step == 0:
                audit["first_step_gradient_audit"] = audit_decoder_gradients(model)
                (output / "trainable_params.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
            optimizer.step()
            global_step += 1; epoch_loss += float(loss.detach()); completed += 1
            log = {"epoch": epoch, "step": step, "global_step": global_step, "loss": float(loss.detach()), **metrics,
                   "supervision": "teacher_pseudo_mask_not_ground_truth"}
            _append_jsonl(log_stream, log); print(json.dumps(log), flush=True)
            next_epoch, next_step = (epoch + 1, 0) if step + 1 == len(rows) else (epoch, step + 1)
            if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                if next_epoch == epoch:
                    checkpoint(next_epoch, next_step, False, epoch_loss, completed)
                else:
                    checkpoint(next_epoch, next_step, False, 0.0, 0)
        average = epoch_loss / max(1, completed)
        if completed and not stopped and average < best:
            best = average
            best_state = make_training_state(model, optimizer, epoch=epoch + 1, next_step=0,
                                             global_step=global_step, elapsed_seconds=elapsed(),
                                             best_training_loss=best, sam2_commit=SAM2_COMMIT,
                                             student_checkpoint_sha256=student_sha,
                                             pseudo_manifest_sha256=pseudo_manifest_sha,
                                             supervision="teacher_pseudo_mask_not_ground_truth",
                                             teacher_pseudo_coverage=teacher_coverage,
                                             best_metric="complete_epoch_training_loss_against_teacher_pseudo_masks_not_mask_gt",
                                             epoch_accumulated_loss=0.0, epoch_accumulated_steps=0)
            atomic_torch_save(best_state, output / "best.pt")
        if stopped:
            checkpoint(epoch, step, True, epoch_loss, completed)
        else:
            checkpoint(epoch + 1, 0, False, 0.0, 0)
        if stopped: break
    result = {"epochs_completed": epoch + (0 if stopped else 1), "steps_completed": global_step,
              "stopped": stopped, "termination_signal": stop_state["signal"], "elapsed_seconds": elapsed(),
              "best_metric": "complete_epoch_training_loss_against_teacher_pseudo_masks_not_mask_gt",
              "best_training_loss": best, "teacher_pseudo_coverage": teacher_coverage}
    log_stream.flush(); os.fsync(log_stream.fileno()); log_stream.close(); release_gpu(model)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pseudo = sub.add_parser("pseudo")
    pseudo.add_argument("--manifest", required=True); pseudo.add_argument("--output", required=True)
    pseudo.add_argument("--teacher-checkpoint", required=True); pseudo.add_argument("--teacher-config", default=LARGE_CONFIG)
    pseudo.add_argument("--device", default="cuda")
    pseudo.add_argument("--maximum", "--max-samples", "--max-masks", dest="maximum", type=int, default=0,
                        help="train boxes to select; 0 means all")
    pseudo.add_argument("--selection-order", choices=("sequence_frame", "manifest", "balanced"), default="sequence_frame")
    pseudo.add_argument("--flush-every", type=int, default=25)
    pseudo.add_argument("--min-score", type=float, default=0.70)
    pseudo.add_argument("--min-crop", type=int, default=256); pseudo.add_argument("--box-scale", type=float, default=6.0)
    pseudo.add_argument("--max-seconds", type=float, default=0)
    train = sub.add_parser("train")
    train.add_argument("--pseudo-manifest", required=True); train.add_argument("--output", required=True)
    train.add_argument("--student-checkpoint", required=True); train.add_argument("--student-config", default=TINY_CONFIG)
    train.add_argument("--device", default="cuda"); train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=1e-5); train.add_argument("--max-seconds", type=float, default=0)
    train.add_argument("--resume"); train.add_argument("--checkpoint-every-steps", type=int, default=25)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "pseudo":
        rows = read_manifest(args.manifest)
        records = normalize_manifest_rows(rows, Path(args.manifest).resolve().parent)
        records = select_training_records(records, args.maximum, args.selection_order)
        result = generate_pseudo_masks(records, args.output, args.teacher_config, args.teacher_checkpoint, args.device,
                                       args.min_score, args.min_crop, args.box_scale, args.max_seconds, args.flush_every)
        print(json.dumps({"selected": len(records), "accepted": sum(x["status"] == "accepted" for x in result)}))
    else:
        print(json.dumps(train_decoder(args.pseudo_manifest, args.output, args.student_config, args.student_checkpoint,
                                       device=args.device, epochs=args.epochs, learning_rate=args.learning_rate,
                                       max_seconds=args.max_seconds, resume=args.resume,
                                       checkpoint_every_steps=args.checkpoint_every_steps)))


if __name__ == "__main__": main()
