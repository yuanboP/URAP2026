"""Validated manifest loading and portable RGB+motion training samples."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _inside(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{description} is outside data_root: {path}") from exc
    return resolved


def load_manifest(data_root, manifest, expected_split=None, require_motion=False, limit=0):
    """Load JSONL records while enforcing the release's zero-based frame contract."""
    root = Path(data_root).resolve()
    manifest_path = _inside(root / manifest, root, "manifest") if not Path(manifest).is_absolute() else _inside(Path(manifest), root, "manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    records, seen, seen_frames, split_by_sha = [], set(), set(), {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prefix = f"{manifest_path}:{line_number}"
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{prefix}: invalid sample_id")
            if sample_id in seen:
                raise ValueError(f"{prefix}: duplicate sample_id {sample_id}")
            seen.add(sample_id)
            sha = row.get("source_sha256")
            if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
                raise ValueError(f"{prefix}: invalid source_sha256")
            if expected_split is not None and row.get("split") != expected_split:
                raise ValueError(f"{prefix}: expected split {expected_split!r}")
            split = row.get("split")
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"{prefix}: invalid split")
            if split_by_sha.setdefault(sha, split) != split:
                raise ValueError(f"{prefix}: source SHA appears in multiple splits")
            frame_id = row.get("source_frame_id")
            frame_count = row.get("video_frame_count")
            if not isinstance(frame_id, int) or not isinstance(frame_count, int) or not 0 <= frame_id < frame_count:
                raise ValueError(f"{prefix}: source_frame_id outside video boundary")
            width, height = row.get("source_width"), row.get("source_height")
            if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
                raise ValueError(f"{prefix}: invalid source dimensions")
            boxes = row.get("bbox_xyxy", [])
            classes = row.get("class_id", [])
            if len(boxes) != len(classes):
                raise ValueError(f"{prefix}: bbox/class count mismatch")
            if any(type(class_id) is not int or class_id != 0 for class_id in classes):
                raise ValueError(f"{prefix}: unsupported class_id; only drone class 0 is valid")
            if not boxes and row.get("role") != "negative_certified":
                raise ValueError(f"{prefix}: uncertified empty annotation")
            frame_key = (sha, frame_id)
            if frame_key in seen_frames:
                raise ValueError(f"{prefix}: duplicate source frame")
            seen_frames.add(frame_key)
            for box in boxes:
                if len(box) != 4 or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in box):
                    raise ValueError(f"{prefix}: invalid bbox")
                x1, y1, x2, y2 = map(float, box)
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"{prefix}: bbox outside source bounds")
            eligible = row.get("motion_eligible") is True
            neighbors = row.get("neighbor_frame_ids")
            if eligible:
                expected = [frame_id - 1, frame_id, frame_id + 1]
                if neighbors != expected or expected[0] < 0 or expected[-1] >= frame_count:
                    raise ValueError(f"{prefix}: invalid neighbor_frame_ids")
            if require_motion and not eligible:
                raise ValueError(f"{prefix}: motion required but record is ineligible")
            source = Path(str(row.get("source_video_path", "")))
            candidates = [root / source, manifest_path.parent / source]
            resolved = None
            for candidate in candidates:
                try:
                    contained = _inside(candidate, root, "source_video_path")
                except ValueError:
                    continue
                if contained == root / "videos" / f"{sha}.mp4":
                    resolved = contained
                    break
            if resolved is None:
                raise ValueError(f"{prefix}: source_video_path is outside data_root or does not match SHA")
            row = dict(row)
            row["_video_path"] = str(resolved)
            records.append(row)
            if limit and len(records) >= int(limit):
                break
    return records


def _decode_frames(path: Path, frame_ids, expected_shape):
    cv2.setNumThreads(1)

    def open_capture():
        value = cv2.VideoCapture(str(path))
        if not value.isOpened():
            value.release()
            raise RuntimeError(f"cannot open source video: {path}")
        return value

    def checked_read(value, frame_id, seek):
        if seek and not value.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id)):
            return None, {"reason": "seek_rejected", "actual_frame_id": None}
        ok, frame = value.read()
        reported = value.get(cv2.CAP_PROP_POS_FRAMES)
        actual = int(round(reported)) - 1 if reported is not None and np.isfinite(reported) else None
        if not ok or frame is None:
            return None, {"reason": "read_failed", "actual_frame_id": actual}
        if actual != int(frame_id):
            return None, {"reason": "position_mismatch", "actual_frame_id": actual}
        if tuple(frame.shape[:2]) != tuple(expected_shape):
            return None, {"reason": "shape_mismatch", "actual_frame_id": actual,
                          "actual_shape": list(frame.shape[:2]), "expected_shape": list(expected_shape)}
        return frame, None

    capture = open_capture(); frames = []
    contiguous = all(frame_ids[index] == frame_ids[0] + index for index in range(len(frame_ids)))
    try:
        for index, frame_id in enumerate(frame_ids):
            frame, error = checked_read(capture, frame_id, index == 0 or not contiguous)
            if error is not None:
                initial = error; capture.release(); capture = None; failures = []
                for attempt in range(1, 3):
                    fresh = open_capture()
                    frame, retry_error = checked_read(fresh, frame_id, True)
                    if retry_error is None:
                        capture = fresh
                        break
                    failures.append({"attempt": attempt, **retry_error}); fresh.release()
                if frame is None:
                    raise RuntimeError(f"exact decode failed: requested={frame_id}, initial={initial}, retries={failures}, video={path}")
            frames.append(frame)
    finally:
        if capture is not None:
            capture.release()
    return frames


@lru_cache(maxsize=1)
def _motion_function():
    here = Path(__file__).resolve().parents[1]
    source = here / "docs" / "training-prompt" / "reference_implementation" / "yolomg_parallax_robust_difference.py"
    spec = importlib.util.spec_from_file_location("_drone_training_motion_reference", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load motion reference: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compensated_difference


_MOTION_DEFAULTS = {
    "local_grid_cols": 36, "local_grid_rows": 21, "local_neighbors": 18,
    "local_radius": 190.0, "local_max_shift": 18.0,
    "local_apply_threshold": 0.18, "track_scale": 0.5,
}


class FrameDataset(Dataset):
    def __init__(self, records, data_root, size=640, augment=False, seed=17,
                 with_motion=True, cache_dir=None, cache_gib=10, motion_config=None):
        self.records = list(records)
        self.data_root = Path(data_root).resolve()
        self.size = (int(size), int(size)) if isinstance(size, int) else tuple(map(int, size))
        self.augment = bool(augment)
        self.seed = int(seed)
        self.with_motion = bool(with_motion)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_gib = float(cache_gib)
        self.motion_config = {**_MOTION_DEFAULTS, **(motion_config or {})}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.records)

    def _motion(self, record, frames):
        if not self.with_motion:
            return np.zeros(frames[0].shape[:2], np.float32), False, "disabled"
        if record.get("motion_eligible") is not True or record.get("neighbor_frame_ids") is None:
            return np.zeros(frames[0].shape[:2], np.float32), True, "boundary_or_ineligible"
        cache_path = None
        if self.cache_dir:
            signature = json.dumps(self.motion_config, sort_keys=True, separators=(",", ":"))
            key = hashlib.sha256(f'{record["source_sha256"]}:{record["source_frame_id"]}:{signature}'.encode()).hexdigest()
            cache_path = self.cache_dir / f"{key}.npz"
            if cache_path.is_file():
                try:
                    with np.load(cache_path, allow_pickle=False) as cached:
                        motion = cached["motion"].astype(np.float32)
                    if motion.shape == frames[1].shape[:2]:
                        cache_path.touch()
                        return motion, False, None
                except (OSError, ValueError, KeyError):
                    cache_path.unlink(missing_ok=True)
        try:
            gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
            difference, valid, _ = _motion_function()(gray[0], gray[1], gray[2], SimpleNamespace(**self.motion_config), True)
            difference = difference.astype(np.float32)
            difference[valid == 0] = 0.0
            motion = difference / 255.0
            if cache_path is not None:
                cap = max(0, int(self.cache_gib * 1024 ** 3))
                if motion.nbytes <= cap:
                    try:
                        with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as handle:
                            temporary = Path(handle.name)
                        try:
                            np.savez_compressed(temporary, motion=motion.astype(np.float32))
                            needed = temporary.stat().st_size
                            entries = sorted(self.cache_dir.glob("*.npz"), key=lambda path: path.stat().st_mtime)
                            total = sum(path.stat().st_size for path in entries)
                            while entries and total + needed > cap:
                                victim = entries.pop(0); total -= victim.stat().st_size; victim.unlink(missing_ok=True)
                            if needed <= cap:
                                temporary.replace(cache_path)
                        finally:
                            temporary.unlink(missing_ok=True)
                    except OSError:
                        pass  # Cache is optional; computed motion remains authoritative.
            return motion, False, None
        except Exception as exc:
            return np.zeros(frames[1].shape[:2], np.float32), True, f"compensation_failed:{type(exc).__name__}"

    def __getitem__(self, item):
        if isinstance(item, tuple):
            index, augmentation_seed = int(item[0]), int(item[1])
        else:
            index, augmentation_seed = int(item), self.seed + int(item)
        record = self.records[index]
        frame_id = int(record["source_frame_id"])
        triplet = record.get("motion_eligible") is True and record.get("neighbor_frame_ids") is not None and self.with_motion
        ids = list(record["neighbor_frame_ids"]) if triplet else [frame_id]
        decoded = _decode_frames(Path(record["_video_path"]), ids,
                                 (int(record["source_height"]), int(record["source_width"])))
        rgb_bgr = decoded[1] if triplet else decoded[0]
        motion, fallback, reason = self._motion(record, decoded if triplet else [rgb_bgr])
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        boxes = np.asarray(record.get("bbox_xyxy", []), dtype=np.float32).reshape(-1, 4)
        classes = np.asarray(record.get("class_id", []), dtype=np.float32).reshape(-1, 1)
        height, width = rgb.shape[:2]
        if self.augment:
            rng = np.random.default_rng(augmentation_seed)
            scale = float(rng.uniform(0.9, 1.1)); tx = float(rng.uniform(-0.08, 0.08) * width); ty = float(rng.uniform(-0.08, 0.08) * height)
            matrix = np.asarray([[scale, 0, tx + width * (1 - scale) / 2], [0, scale, ty + height * (1 - scale) / 2]], np.float32)
            rgb = cv2.warpAffine(rgb, matrix, (width, height), borderValue=(114 / 255,) * 3)
            motion = cv2.warpAffine(motion, matrix, (width, height), borderValue=0)
            if len(boxes):
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + matrix[0, 2]
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + matrix[1, 2]
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
            if rng.random() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1]); motion = np.ascontiguousarray(motion[:, ::-1])
                if len(boxes):
                    old_x1 = boxes[:, 0].copy(); boxes[:, 0] = width - boxes[:, 2]; boxes[:, 2] = width - old_x1
            if len(boxes):
                keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
                boxes, classes = boxes[keep], classes[keep]
        target_h, target_w = self.size
        ratio = min(target_w / width, target_h / height)
        resized_w, resized_h = int(round(width * ratio)), int(round(height * ratio))
        left, top = (target_w - resized_w) // 2, (target_h - resized_h) // 2
        canvas = np.full((target_h, target_w, 4), 114 / 255, np.float32); canvas[..., 3] = 0
        canvas[top:top+resized_h, left:left+resized_w, :3] = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas[top:top+resized_h, left:left+resized_w, 3] = cv2.resize(motion, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio + left; boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + top
            xywh = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2 / target_w, (boxes[:, 1] + boxes[:, 3]) / 2 / target_h, (boxes[:, 2] - boxes[:, 0]) / target_w, (boxes[:, 3] - boxes[:, 1]) / target_h)).astype(np.float32)
        else:
            xywh = np.empty((0, 4), np.float32)
        metadata = {"sample_id": record["sample_id"], "source_domain": record.get("source_domain"),
                    "sequence_id": record.get("sequence_id"), "source_sha256": record["source_sha256"],
                    "source_frame_id": frame_id, "split": record.get("split"), "role": record.get("role"),
                    "original_size": (height, width), "ratio": ratio,
                    "pad": (left, top), "boxes_complete_for_frame": record.get("boxes_complete_for_frame"),
                    "bbox_xyxy": [list(map(float, box)) for box in record.get("bbox_xyxy", [])],
                    "motion_fallback": fallback, "motion_reason": reason}
        return {"img": torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))),
                "bboxes": torch.from_numpy(xywh), "cls": torch.from_numpy(classes.astype(np.float32)),
                "metadata": metadata}


def collate_samples(samples):
    return {"img": torch.stack([s["img"] for s in samples]),
            "cls": torch.cat([s["cls"] for s in samples], dim=0),
            "bboxes": torch.cat([s["bboxes"] for s in samples], dim=0),
            "batch_idx": torch.cat([torch.full((len(s["bboxes"]),), i, dtype=torch.long) for i, s in enumerate(samples)]),
            "metadata": [s["metadata"] for s in samples]}
