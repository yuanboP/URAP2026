"""Matched deterministic augmentation for native RGB, motion, validity, and boxes."""
from __future__ import annotations

import cv2
import numpy as np


_TARGET_INTERVALS = ((4., 16.), (16., 32.), (32., 64.), (64., 128.))


def _full_spec(record, rng, kind, size):
    w, h = float(record["source_width"]), float(record["source_height"])
    boxes = [list(map(float, b)) for b in record.get("bbox_xyxy", [])]
    return {"original_gt_xyxy": boxes, "crop_xyxy": [0., 0., w, h], "flip": bool(rng.integers(2)),
            "kind": kind, "target_bin": "full_frame", "requested_target_short_side": None,
            "effective_target_short_side": _effective_short(boxes, w, h, size), "fallback_reason": None}


def _effective_short(boxes, crop_w, crop_h, size):
    if not boxes:
        return None
    arr = np.asarray(boxes, np.float64)
    return float(np.min(np.minimum(arr[:, 2]-arr[:, 0], arr[:, 3]-arr[:, 1])) * size / max(crop_w, crop_h))


def _positive_crop(record, rng, size):
    w, h = float(record["source_width"]), float(record["source_height"])
    boxes = np.asarray(record["bbox_xyxy"], np.float64).reshape(-1, 4)
    interval_id = int(rng.integers(len(_TARGET_INTERVALS)))
    low, high = _TARGET_INTERVALS[interval_id]
    target = float(rng.uniform(low, high))
    short = float(np.min(np.minimum(boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1])))
    desired_long = short * size / target
    union = np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()])
    union_w, union_h = union[2]-union[0], union[3]-union[1]
    min_long = max(w, h) / 8.0
    aspect = w / h
    crop_w = max(union_w, min_long if aspect >= 1 else min_long * aspect,
                 desired_long if aspect >= 1 else desired_long * aspect)
    crop_h = max(union_h, min_long / aspect if aspect >= 1 else min_long,
                 desired_long / aspect if aspect >= 1 else desired_long)
    factor = max(union_w / crop_w, union_h / crop_h, 1.0)
    crop_w, crop_h = min(w, crop_w * factor), min(h, crop_h * factor)
    center = np.array([(union[0]+union[2])/2, (union[1]+union[3])/2])
    jitter = (rng.random(2)-.5) * np.maximum(0., np.array([crop_w-union_w, crop_h-union_h])) * .7
    center += jitter
    x1 = np.clip(center[0]-crop_w/2, max(0., union[2]-crop_w), min(union[0], w-crop_w))
    y1 = np.clip(center[1]-crop_h/2, max(0., union[3]-crop_h), min(union[1], h-crop_h))
    crop = [float(np.floor(x1)), float(np.floor(y1)), float(np.ceil(x1+crop_w)), float(np.ceil(y1+crop_h))]
    crop[0], crop[1] = max(0., crop[0]), max(0., crop[1])
    crop[2], crop[3] = min(w, crop[2]), min(h, crop[3])
    crop_w, crop_h = crop[2]-crop[0], crop[3]-crop[1]
    effective = _effective_short(boxes.tolist(), crop_w, crop_h, size)
    reason = None if low <= effective <= high else "GT union, image bounds, or 8x zoom clamp limited requested scale"
    return {"original_gt_xyxy": boxes.tolist(), "crop_xyxy": crop, "flip": bool(rng.integers(2)),
            "kind": "positive", "target_bin": f"{int(low)}-{int(high)}",
            "requested_target_short_side": target, "effective_target_short_side": effective,
            "fallback_reason": reason}


def _intersects_any(rect, forbidden):
    if not len(forbidden):
        return False
    x1, y1, x2, y2 = rect
    return bool(np.any((forbidden[:, 0] < x2) & (forbidden[:, 2] > x1) &
                       (forbidden[:, 1] < y2) & (forbidden[:, 3] > y1)))


def _region_negative(record, rng, size):
    domain = record.get("source_domain", record.get("dataset"))
    if domain != "ARD100":
        raise ValueError("region-negative views are ARD100 only")
    w, h = float(record["source_width"]), float(record["source_height"])
    boxes = np.asarray(record.get("bbox_xyxy", []), np.float64).reshape(-1, 4)
    if not len(boxes):
        raise ValueError("region-negative view requires positive ARD GT")
    margins = np.maximum(16., .5*np.maximum(boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]))
    forbidden = boxes.copy()
    forbidden[:, [0, 1]] -= margins[:, None]; forbidden[:, [2, 3]] += margins[:, None]
    forbidden[:, [0, 2]] = np.clip(forbidden[:, [0, 2]], 0, w)
    forbidden[:, [1, 3]] = np.clip(forbidden[:, [1, 3]], 0, h)
    candidates = []
    for frac in (.5, .35, .25, .18):
        cw, ch = w*frac, h*frac
        for ax, ay in ((0, 0), (1, 0), (0, 1), (1, 1), (.5, .5)):
            candidates.append([ax*(w-cw), ay*(h-ch), ax*(w-cw)+cw, ay*(h-ch)+ch])
    for _ in range(32):
        frac = float(rng.uniform(.15, .5)); cw, ch = w*frac, h*frac
        x1, y1 = float(rng.uniform(0, w-cw)), float(rng.uniform(0, h-ch))
        candidates.append([x1, y1, x1+cw, y1+ch])
    # Exhaustive boundary-aligned corner fallbacks. An axis-aligned maximal
    # disjoint corner rectangle has an edge on a forbidden-box boundary.
    xs = sorted(set([0., w] + forbidden[:, 0].tolist() + forbidden[:, 2].tolist()))
    ys = sorted(set([0., h] + forbidden[:, 1].tolist() + forbidden[:, 3].tolist()))
    for x in xs:
        for y in ys:
            candidates.extend(([0., 0., x, y], [x, 0., w, y],
                               [0., y, x, h], [x, y, w, h]))
    candidates = [[float(np.floor(r[0])), float(np.floor(r[1])), float(np.ceil(r[2])), float(np.ceil(r[3]))]
                  for r in candidates]
    safe = [r for r in candidates if r[2] > r[0] and r[3] > r[1] and not _intersects_any(r, forbidden)]
    if not safe:
        raise ValueError(f"no safe region-negative crop for sample {record.get('sample_id', '<unknown>')}")
    largest = max(safe, key=lambda r: (r[2]-r[0])*(r[3]-r[1]))
    return {"original_gt_xyxy": boxes.tolist(), "crop_xyxy": list(map(float, largest)),
            "flip": bool(rng.integers(2)), "kind": "region_negative", "target_bin": "gt_disjoint",
            "requested_target_short_side": None, "effective_target_short_side": None,
            "fallback_reason": None, "gt_disjoint": True,
            "safety_margin_px": margins.tolist()}


def make_view_spec(record, view_kind, aug_seed, size=1280):
    rng = np.random.default_rng(int(aug_seed))
    role = record.get("role")
    if int(view_kind) == 2:
        return _region_negative(record, rng, size)
    if role == "positive":
        if not record.get("bbox_xyxy"):
            raise ValueError("positive record has no GT boxes")
        if rng.random() < .5:
            return _full_spec(record, rng, "positive", size)
        return _positive_crop(record, rng, size)
    if role == "negative_certified":
        spec = _full_spec(record, rng, "negative_certified", size)
        if rng.random() < .5:
            w, h = float(record["source_width"]), float(record["source_height"])
            frac = float(rng.uniform(.5, 1.)); cw, ch = w*frac, h*frac
            x1, y1 = float(rng.uniform(0, w-cw)), float(rng.uniform(0, h-ch))
            spec["crop_xyxy"] = [float(np.floor(x1)), float(np.floor(y1)),
                                  float(np.ceil(x1+cw)), float(np.ceil(y1+ch))]
            spec["target_bin"] = "certified_crop"
        return spec
    raise ValueError(f"unsupported record role: {role!r}")


def apply_view(rgb, motion, valid, boxes_xyxy, spec, size=1280):
    rgb = np.asarray(rgb); motion = np.asarray(motion); valid = np.asarray(valid)
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be uint8 HWC")
    if motion.dtype != np.float32 or motion.shape != rgb.shape[:2] or valid.shape != rgb.shape[:2]:
        raise ValueError("motion must be float32 native HW and valid must match native HW")
    x1, y1, x2, y2 = map(float, spec["crop_xyxy"])
    h, w = rgb.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("crop_xyxy lies outside native image")
    if spec["kind"] == "positive":
        original = np.asarray(spec["original_gt_xyxy"], np.float32).reshape(-1, 4)
        if not len(original) or np.any(original[:, :2] < [x1, y1]) or np.any(original[:, 2:] > [x2, y2]):
            raise AssertionError("positive crop does not fully contain every original GT box")
    elif spec["kind"] == "region_negative":
        if _intersects_any([x1, y1, x2, y2], boxes):
            raise AssertionError("region-negative crop intersects source GT")
        boxes = np.empty((0, 4), np.float32)
    elif spec["kind"] == "negative_certified":
        if len(boxes):
            raise AssertionError("certified negative source contains boxes")
    if any(value != int(value) for value in (x1, y1, x2, y2)):
        raise ValueError("crop_xyxy must use integer pixel boundaries")
    ix1, iy1, ix2, iy2 = map(int, (x1, y1, x2, y2))
    crop_rgb = rgb[iy1:iy2, ix1:ix2]
    safe_motion = motion.copy()
    safe_motion[~valid.astype(bool)] = 0.0
    crop_motion = safe_motion[iy1:iy2, ix1:ix2]
    crop_valid = valid[iy1:iy2, ix1:ix2]
    cw, ch = x2-x1, y2-y1
    scale = min(size/cw, size/ch)
    rw, rh = max(1, int(round(cw*scale))), max(1, int(round(ch*scale)))
    resized_rgb = cv2.resize(crop_rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    resized_motion = cv2.resize(crop_motion, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    resized_valid = cv2.resize(crop_valid.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
    resized_motion[resized_valid == 0] = 0.0
    left, top = (size-rw)//2, (size-rh)//2
    rgb_out = np.full((size, size, 3), 114, np.uint8)
    motion_out = np.zeros((size, size), np.float32); valid_out = np.zeros((size, size), np.uint8)
    rgb_out[top:top+rh, left:left+rw] = resized_rgb
    motion_out[top:top+rh, left:left+rw] = resized_motion
    valid_out[top:top+rh, left:left+rw] = resized_valid
    scale_x, scale_y = rw/cw, rh/ch
    if len(boxes):
        boxes[:, [0, 2]] = (boxes[:, [0, 2]]-x1)*scale_x + left
        boxes[:, [1, 3]] = (boxes[:, [1, 3]]-y1)*scale_y + top
    if spec.get("flip", False):
        rgb_out = rgb_out[:, ::-1].copy(); motion_out = motion_out[:, ::-1].copy(); valid_out = valid_out[:, ::-1].copy()
        if len(boxes):
            old_x1 = boxes[:, 0].copy(); boxes[:, 0] = size-boxes[:, 2]; boxes[:, 2] = size-old_x1
    if len(boxes):
        xywh = np.column_stack(((boxes[:, 0]+boxes[:, 2])*.5/size,
                                (boxes[:, 1]+boxes[:, 3])*.5/size,
                                (boxes[:, 2]-boxes[:, 0])/size,
                                (boxes[:, 3]-boxes[:, 1])/size)).astype(np.float32)
    else:
        xywh = np.empty((0, 4), np.float32)
    image = np.concatenate((rgb_out.astype(np.float32).transpose(2, 0, 1)/255., motion_out[None]), axis=0)
    return {"img": image.astype(np.float32, copy=False), "bboxes": xywh,
            "cls": np.zeros((len(xywh), 1), np.float32), "valid": valid_out.astype(bool),
            "metadata": {**spec, "resize_scale": float(scale), "scale_xy": [float(scale_x), float(scale_y)],
                         "resized_wh": [rw, rh],
                         "pad_ltrb": [left, top, size-rw-left, size-rh-top]}}
