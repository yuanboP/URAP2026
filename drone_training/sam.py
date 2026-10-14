"""Portable entry points for official SAM 2.1 pseudo-label adaptation.

SAM is prompt-conditioned here.  Inference therefore requires externally
generated proposal boxes and reports mask-quality scores, not drone scores.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np


ADAPTER_GRAPH_VERSION = "sam2_image_predictor_feature_parity_v2"


def _config(value):
    return value if not isinstance(value, dict) else SimpleNamespace(**value)


def _get(config, name, default=None, *aliases):
    for key in (name, *aliases):
        if hasattr(config, key):
            return getattr(config, key)
    return default


def _reference(config=None):
    injected = _get(config, "_reference") if config is not None else None
    if injected is not None:
        return injected
    source = Path(__file__).resolve().parents[1] / "docs" / "training-prompt" / "reference_implementation" / "sam_adapt.py"
    spec = importlib.util.spec_from_file_location("_urap_sam_adapt_reference", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load immutable SAM adaptation reference: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _student_forward(model, image, box):
    """Decoder-training graph matching SAM2ImagePredictor image features."""
    import torch
    expected_side = int(model.image_size)
    if expected_side != 1024 or tuple(image.shape[-2:]) != (expected_side, expected_side):
        raise ValueError("official SAM2.1 Tiny decoder path requires a 1024x1024 input")
    with torch.no_grad():
        backbone = model.forward_image(image)
        _, feats, _, sizes = model._prepare_backbone_features(backbone)
        if getattr(model, "directly_add_no_mem_embed", False):
            # This is the exact constant addition used by SAM2ImagePredictor.
            # Detach explicitly because adaptation freezes every non-decoder parameter.
            feats[-1] = feats[-1] + model.no_mem_embed.detach()
        feats = [value.detach().clone() for value in feats]
    if len(feats) != 3 or len(sizes) != 3:
        raise RuntimeError(f"expected three SAM2 FPN levels, got feats={len(feats)} sizes={len(sizes)}")
    image_embed = feats[-1].permute(1, 2, 0).reshape(image.shape[0], -1, *sizes[-1])
    high_res = [value.permute(1, 2, 0).reshape(image.shape[0], -1, *size)
                for value, size in zip(feats[:-1], sizes[:-1])]
    if len(high_res) != 2 or tuple(image_embed.shape[-2:]) != (model.sam_image_embedding_size,) * 2:
        raise RuntimeError("SAM2 decoder feature pyramid has unexpected shapes")
    coords = torch.tensor([[box[:2], box[2:]]], device=image.device, dtype=torch.float32)
    labels = torch.tensor([[2, 3]], device=image.device, dtype=torch.int32)
    outputs = model._forward_sam_heads(image_embed, {"point_coords": coords, "point_labels": labels},
                                       high_res_features=high_res, multimask_output=False)
    return outputs[1], outputs[2]


def _read_rows(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix.lower() == ".json" else None
    rows = (payload if isinstance(payload, list) else payload.get("records", [])) if payload is not None else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not rows:
        raise ValueError(f"manifest contains no records: {path}")
    return rows


def _canonical_train_rows(rows, data_root):
    """Resolve release video paths using data_root and normalize split spelling."""
    root = Path(data_root).resolve()
    selected = []
    for original in rows:
        if str(original.get("split", "")).lower() != "train":
            continue
        # Detection manifests can include certified empty frames.  They are
        # valid for a detector head but cannot prompt a SAM teacher.
        if not (original.get("boxes") or original.get("objects") or original.get("bbox_xyxy")):
            continue
        row = dict(original)
        row["split"] = "train"
        source = row.get("source_video_path")
        sha = row.get("source_sha256")
        if source:
            parsed = urlparse(str(source))
            if parsed.scheme not in {"", "file"}:
                raise ValueError("SAM training requires locally materialized source videos")
            path = Path(parsed.path if parsed.scheme == "file" else str(source))
            candidates = [path] if path.is_absolute() else [root / path, root / "videos" / path.name]
            resolved = next((p.resolve() for p in candidates if p.is_file()), None)
            if resolved is None and sha:
                candidate = (root / "videos" / f"{sha}.mp4").resolve()
                resolved = candidate if candidate.is_file() else None
            if resolved is None:
                raise FileNotFoundError(f"source video for {row.get('sample_id')} was not found under data_root {root}")
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"source video is outside data_root: {resolved}") from exc
            if sha and resolved.name != f"{sha}.mp4":
                raise ValueError(f"source video filename does not match source_sha256 for {row.get('sample_id')}")
            row["source_video_path"] = str(resolved)
        selected.append(row)
    if not selected:
        raise ValueError("manifest has no train records; provide the positive training manifest, not validation/test data")
    return selected


def pseudo(config):
    config = _config(config); reference = _reference(config)
    manifest = Path(_get(config, "manifest")).resolve()
    rows = _canonical_train_rows(_read_rows(manifest), _get(config, "data_root"))
    records = reference.normalize_manifest_rows(rows, manifest.parent)
    maximum = int(_get(config, "maximum", _get(config, "limit", 0), "max_samples"))
    records = reference.select_training_records(records, maximum, _get(config, "selection_order", "sequence_frame"))
    if not records:
        raise ValueError("manifest produced no positive train box prompts; check bbox_xyxy and split=train")
    output = Path(_get(config, "out", None, "output")).resolve()
    generated = reference.generate_pseudo_masks(
        records, output, _get(config, "teacher_config", reference.LARGE_CONFIG),
        str(Path(_get(config, "teacher_checkpoint")).resolve()), _get(config, "device", "cuda"),
        float(_get(config, "min_score", 0.70)), int(_get(config, "min_crop", 256)),
        float(_get(config, "box_scale", 6.0)), float(_get(config, "max_seconds", 0)),
        int(_get(config, "flush_every", 25)),
    )
    accepted = sum(row.get("status") == "accepted" for row in generated)
    return {"selected": len(records), "accepted": accepted, "rejected": len(records) - accepted,
            "pseudo_manifest": str(output / "pseudo_manifest.jsonl")}


def train_sam(config):
    config = _config(config); reference = _reference(config)
    pseudo_manifest = Path(_get(config, "pseudo_manifest")).resolve()
    rows = reference.recover_jsonl(pseudo_manifest) if hasattr(reference, "recover_jsonl") else _read_rows(pseudo_manifest)
    accepted = [row for row in rows if row.get("status") == "accepted" and row.get("split") == "train"]
    if not accepted:
        raise ValueError("pseudo manifest has no accepted train masks; run SAM pseudo generation and inspect rejection counts")
    student = Path(_get(config, "student_checkpoint")).resolve()
    resume_value = _get(config, "resume")
    resume = Path(resume_value).resolve() if resume_value else None
    student_config = _get(config, "student_config", reference.TINY_CONFIG)
    learning_rate = float(_get(config, "learning_rate", _get(config, "lr", 1e-5)))
    epochs = int(_get(config, "epochs", 5))
    adapter_identity = {"student_checkpoint_sha256": _sha256(student),
                        "pseudo_manifest_sha256": _sha256(pseudo_manifest),
                        "student_config": student_config, "learning_rate": learning_rate,
                        "adapter_graph_version": ADAPTER_GRAPH_VERSION,
                        "adapter_source_sha256": _sha256(__file__)}
    if resume:
        import torch
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("student_checkpoint_sha256") != _sha256(student):
            raise ValueError("resume checkpoint was created from a different student checkpoint")
        recorded_pseudo = state.get("pseudo_manifest_sha256")
        if recorded_pseudo is not None and recorded_pseudo != _sha256(pseudo_manifest):
            raise ValueError("resume checkpoint was created from a different pseudo manifest")
        recorded_identity = state.get("adapter_identity")
        if recorded_identity is None:
            raise ValueError("resume checkpoint is missing immutable adapter identity")
        if recorded_identity != adapter_identity:
            raise ValueError("resume rejected because immutable SAM training configuration changed")
        if int(state.get("epoch", 0)) >= epochs and int(state.get("next_step", 0)) == 0:
            return {"epochs_completed":int(state["epoch"]),"steps_completed":int(state.get("global_step",0)),
                    "stopped":False,"elapsed_seconds":float(state.get("elapsed_seconds",0.0)),
                    "best_training_loss":float(state.get("best_training_loss",float("inf"))),
                    "teacher_pseudo_coverage":state.get("teacher_pseudo_coverage",{}),"already_complete":True}
    output = Path(_get(config, "out", None, "output")).resolve()
    stop_after = int(_get(config, "stop_after", 0))
    if stop_after:
        if resume:
            raise ValueError("stop_after is a fresh-run smoke bound and cannot be combined with resume")
        if stop_after < 1 or stop_after > len(accepted):
            raise ValueError(f"stop_after must be between 1 and the {len(accepted)} accepted masks")
    original_make_state = reference.make_training_state
    def identified_state(*args, **kwargs):
        kwargs["adapter_identity"] = adapter_identity
        return original_make_state(*args, **kwargs)
    reference.make_training_state = identified_state
    original_forward = getattr(reference, "_student_forward", None)
    reference._student_forward = _student_forward
    original_time = getattr(reference, "time", None)
    max_seconds = float(_get(config, "max_seconds", 0))
    if stop_after and original_time is not None:
        counter = {"steps": 0, "origin": original_time.monotonic()}
        def counted_forward(*args, **kwargs):
            result = _student_forward(*args, **kwargs); counter["steps"] += 1; return result
        class _BoundedTime:
            def monotonic(self):
                return counter["origin"] + (2.0 if counter["steps"] >= stop_after else 0.0)
            def __getattr__(self, name): return getattr(original_time, name)
        reference._student_forward = counted_forward; reference.time = _BoundedTime(); max_seconds = 1.0
    try:
        result = reference.train_decoder(
            str(pseudo_manifest), str(output), student_config, str(student),
            device=_get(config, "device", "cuda"), epochs=epochs, learning_rate=learning_rate,
            max_seconds=max_seconds, resume=str(resume) if resume else None,
            checkpoint_every_steps=int(_get(config, "checkpoint_every_steps", _get(config, "checkpoint_every", 25))),
        )
    finally:
        reference.make_training_state = original_make_state
        if original_forward is not None: reference._student_forward = original_forward
        elif hasattr(reference, "_student_forward"): delattr(reference, "_student_forward")
        if original_time is not None: reference.time = original_time
    # Preserve the reference checkpoint format while adding wrapper-level resume identity.
    import torch
    for name in ("last.pt", "best.pt"):
        path = output / name
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=False)
            state["adapter_identity"] = adapter_identity
            temporary = path.with_name(path.name + ".tmp")
            torch.save(state, temporary); temporary.replace(path)
    return result


def _load_image(value):
    if isinstance(value, np.ndarray):
        image = value
    else:
        from PIL import Image
        with Image.open(value) as opened:
            image = np.asarray(opened.convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    return np.asarray(image)


def infer_sam(config):
    config = _config(config)
    boxes = np.asarray(_get(config, "boxes_xyxy", []), dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        raise ValueError("SAM inference requires at least one external proposal box")
    checkpoint = Path(_get(config, "checkpoint")).resolve()
    student = Path(_get(config, "student_checkpoint")).resolve()
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("state_format") != "urap.sam_decoder_checkpoint.v1" or state.get("weights_scope") != "sam_mask_decoder":
        raise ValueError("checkpoint is not a reloadable URAP SAM decoder checkpoint")
    if state.get("adapter_identity", {}).get("adapter_graph_version") != ADAPTER_GRAPH_VERSION:
        raise ValueError("decoder checkpoint predates the inference-parity adapter graph and must be retrained")
    if state.get("student_checkpoint_sha256") != _sha256(student):
        raise ValueError("trained decoder belongs to a different official SAM2 base checkpoint")
    builder = _get(config, "_model_builder")
    if builder is None:
        from sam2.build_sam import build_sam2
        builder = build_sam2
    reference = _reference(config)
    model = builder(_get(config, "student_config", reference.TINY_CONFIG), str(student),
                    device=_get(config, "device", "cuda"), mode="eval")
    model.sam_mask_decoder.load_state_dict(state["model"], strict=True); model.eval()
    factory = _get(config, "_predictor_factory")
    if factory is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        factory = SAM2ImagePredictor
    predictor = factory(model); image = _load_image(_get(config, "image")); predictor.set_image(image)
    results = []
    for box in boxes:
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=False)
        results.append({"proposal_xyxy": box.tolist(), "mask": np.asarray(masks[0], dtype=bool),
                        "predicted_mask_quality": float(np.asarray(scores).reshape(-1)[0]),
                        "score_semantics": "sam_predicted_mask_quality_not_drone_confidence"})
    return results
