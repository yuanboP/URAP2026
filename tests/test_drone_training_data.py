import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import drone_training.data as data_module
from drone_training.data import FrameDataset, collate_samples, load_manifest


SHA = "a" * 64


def _record(**changes):
    row = {
        "sample_id": "sample-1",
        "source_sha256": SHA,
        "source_video_path": f"videos/{SHA}.mp4",
        "source_frame_id": 1,
        "source_annotation_frame_id": 2,
        "source_width": 64,
        "source_height": 32,
        "video_frame_count": 3,
        "split": "train",
        "motion_eligible": True,
        "neighbor_frame_ids": [0, 1, 2],
        "bbox_xyxy": [[8.0, 4.0, 24.0, 20.0]],
        "class_id": [0],
        "source_domain": "test",
        "role": "positive",
        "sequence_id": "sequence-1",
        "boxes_complete_for_frame": True,
    }
    row.update(changes)
    return row


def _write_manifest(root: Path, rows):
    path = root / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _write_video(root: Path):
    path = root / "videos" / f"{SHA}.mp4"
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 32))
    assert writer.isOpened()
    for value in (20, 80, 140):
        frame = np.full((32, 64, 3), value, np.uint8)
        frame[:, :16, 2] = 255
        writer.write(frame)
    writer.release()
    return path


def test_load_manifest_preserves_canonical_frame_and_validates(tmp_path):
    _write_video(tmp_path)
    manifest = _write_manifest(tmp_path, [_record()])
    rows = load_manifest(tmp_path, manifest.name, expected_split="train", require_motion=True)
    assert rows[0]["source_frame_id"] == 1
    assert rows[0]["source_annotation_frame_id"] == 2
    assert rows[0]["_video_path"] == str((tmp_path / "videos" / f"{SHA}.mp4").resolve())


@pytest.mark.parametrize(
    "rows,error",
    [
        ([_record(), _record()], "duplicate sample_id"),
        ([_record(bbox_xyxy=[[0, 0, 65, 4]])], "outside source bounds"),
        ([_record(neighbor_frame_ids=[1, 2, 3])], "neighbor_frame_ids"),
        ([_record(source_video_path="../../../escape.mp4")], "outside data_root"),
    ],
)
def test_load_manifest_rejects_corrupt_rows(tmp_path, rows, error):
    manifest = _write_manifest(tmp_path, rows)
    with pytest.raises(ValueError, match=error):
        load_manifest(tmp_path, manifest.name, require_motion=True)


def test_frame_dataset_returns_rgbd_boxes_and_collates(tmp_path):
    _write_video(tmp_path)
    rows = load_manifest(tmp_path, _write_manifest(tmp_path, [_record()]).name, require_motion=True)
    sample = FrameDataset(rows, tmp_path, size=64, augment=False, with_motion=False)[0]
    assert sample["img"].shape == (4, 64, 64)
    assert sample["img"].dtype == torch.float32
    assert torch.all(sample["img"][3] == 0)
    assert sample["bboxes"].shape == (1, 4)
    assert sample["cls"].shape == (1, 1)
    assert sample["metadata"]["source_frame_id"] == 1
    assert sample["metadata"]["source_domain"] == "test"
    assert sample["metadata"]["sequence_id"] == "sequence-1"
    assert sample["metadata"]["boxes_complete_for_frame"] is True
    assert sample["metadata"]["bbox_xyxy"] == [[8.0, 4.0, 24.0, 20.0]]
    assert sample["metadata"]["split"] == "train"
    assert sample["metadata"]["role"] == "positive"
    assert "motion_reason" in sample["metadata"]
    batch = collate_samples([sample, sample])
    assert batch["img"].shape == (2, 4, 64, 64)
    assert batch["batch_idx"].tolist() == [0, 1]
    assert len(batch["metadata"]) == 2


def test_boundary_frame_has_explicit_zero_motion_fallback(tmp_path):
    _write_video(tmp_path)
    row = _record(source_frame_id=0, neighbor_frame_ids=None, motion_eligible=False)
    rows = load_manifest(tmp_path, _write_manifest(tmp_path, [row]).name)
    sample = FrameDataset(rows, tmp_path, size=64, augment=False, with_motion=True)[0]
    assert torch.count_nonzero(sample["img"][3]) == 0
    assert sample["metadata"]["motion_fallback"] is True
    assert sample["metadata"]["motion_reason"] == "boundary_or_ineligible"


def test_tuple_seed_makes_augmentation_stateless(tmp_path):
    _write_video(tmp_path)
    rows = load_manifest(tmp_path, _write_manifest(tmp_path, [_record()]).name)
    dataset = FrameDataset(rows, tmp_path, size=64, augment=True, seed=9, with_motion=False)
    first = dataset[(0, 1234)]
    second = dataset[(0, 1234)]
    assert torch.equal(first["img"], second["img"])
    assert torch.equal(first["bboxes"], second["bboxes"])


def test_motion_cache_avoids_recomputation(tmp_path, monkeypatch):
    _write_video(tmp_path)
    rows = load_manifest(tmp_path, _write_manifest(tmp_path, [_record()]).name)
    calls = []

    def fake_motion():
        def calculate(previous, reference, following, args, local):
            calls.append(1)
            return np.full(reference.shape, 51, np.float32), np.full(reference.shape, 255, np.uint8), {}
        return calculate

    monkeypatch.setattr(data_module, "_motion_function", fake_motion)
    dataset = FrameDataset(rows, tmp_path, size=64, with_motion=True, cache_dir=tmp_path / "cache", cache_gib=0.001)
    first = dataset[0]
    second = dataset[0]
    assert len(calls) == 1
    assert torch.equal(first["img"][3], second["img"][3])
    assert list((tmp_path / "cache").glob("*.npz"))


@pytest.mark.parametrize(
    "rows,error",
    [
        ([_record(class_id=[1])], "unsupported class_id"),
        ([_record(bbox_xyxy=[], class_id=[], role="unknown")], "uncertified empty"),
        ([_record(sample_id="a"), _record(sample_id="b")], "duplicate source frame"),
        ([_record(sample_id="a"), _record(sample_id="b", source_frame_id=2, neighbor_frame_ids=[1, 2, 3], video_frame_count=4, split="validation")], "multiple splits"),
    ],
)
def test_load_manifest_rejects_unsafe_annotation_semantics(tmp_path, rows, error):
    manifest = _write_manifest(tmp_path, rows)
    with pytest.raises(ValueError, match=error):
        load_manifest(tmp_path, manifest.name)


def test_frame_dataset_rejects_decoded_shape_mismatch(tmp_path):
    _write_video(tmp_path)
    row = _record(source_width=65)
    records = load_manifest(tmp_path, _write_manifest(tmp_path, [row]).name)
    with pytest.raises(RuntimeError, match="exact decode failed.*shape_mismatch"):
        FrameDataset(records, tmp_path, size=64, with_motion=False)[0]


def test_frame_dataset_rejects_decoder_position_mismatch(tmp_path, monkeypatch):
    _write_video(tmp_path)
    records = load_manifest(tmp_path, _write_manifest(tmp_path, [_record()]).name)
    real_capture = cv2.VideoCapture

    class WrongPositionCapture:
        def __init__(self, path):
            self.inner = real_capture(path)

        def isOpened(self): return self.inner.isOpened()
        def release(self): return self.inner.release()
        def set(self, prop, value): return self.inner.set(prop, value)
        def read(self): return self.inner.read()
        def get(self, prop):
            value = self.inner.get(prop)
            return value + 1 if prop == cv2.CAP_PROP_POS_FRAMES else value

    monkeypatch.setattr(data_module.cv2, "VideoCapture", WrongPositionCapture)
    with pytest.raises(RuntimeError, match="exact decode failed.*position_mismatch"):
        FrameDataset(records, tmp_path, size=64, with_motion=False)[0]
