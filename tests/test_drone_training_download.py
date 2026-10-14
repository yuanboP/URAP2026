import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from drone_training.download import download_dataset


SHA = hashlib.sha256(b"video bytes").hexdigest()


def _archive(path: Path, *, unsafe=False):
    row = {
        "sample_id": "x",
        "source_sha256": SHA,
        "source_video_path": f"videos/{SHA}.mp4",
        "source_frame_id": 1,
        "source_width": 10,
        "source_height": 10,
        "video_frame_count": 3,
        "split": "train",
        "motion_eligible": True,
        "neighbor_frame_ids": [0, 1, 2],
        "bbox_xyxy": [],
        "class_id": [],
        "source_domain": "A",
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.txt" if unsafe else "release_v3_curated/manifests/flow_train_common.jsonl", json.dumps(row) + "\n")
        if not unsafe:
            validation = dict(row, sample_id="v", split="validation", source_domain="B")
            archive.writestr("release_v3_curated/manifests/flow_validation.jsonl", json.dumps(validation) + "\n")


def test_download_verifies_archive_extracts_and_fetches_selected_video(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    archive = remote / "metadata.zip"
    _archive(archive)
    (remote / f"{SHA}.mp4").write_bytes(b"video bytes")

    requests = []

    def fake_download(*, filename, local_dir, **kwargs):
        requests.append({"filename": filename, **kwargs})
        source = archive if filename == "release_v3_curated/metadata.zip" else remote / Path(filename).name
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return str(target)

    monkeypatch.setattr("drone_training.download.hf_hub_download", fake_download)
    receipt = download_dataset(
        tmp_path / "data",
        smoke=True,
        train_limit=1,
        val_limit=1,
        metadata_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        metadata_bytes=archive.stat().st_size,
    )
    assert receipt["videos_downloaded"] == 1
    assert receipt["revision"]
    assert (tmp_path / "data" / "videos" / f"{SHA}.mp4").read_bytes() == b"video bytes"
    assert Path(receipt["train_manifest"]).is_file()
    assert Path(receipt["validation_manifest"]).is_file()
    assert requests[0]["filename"] == "release_v3_curated/metadata.zip"
    assert all(request["repo_type"] == "dataset" for request in requests)
    assert all(request["revision"] == receipt["revision"] for request in requests)
    assert Path(receipt["progress_jsonl"]).is_file()


def test_download_rejects_zip_traversal(tmp_path, monkeypatch):
    archive = tmp_path / "bad.zip"
    _archive(archive, unsafe=True)

    def fake_download(*, filename, local_dir, **_):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read_bytes())
        return str(target)

    monkeypatch.setattr("drone_training.download.hf_hub_download", fake_download)
    with pytest.raises(ValueError, match="unsafe archive member"):
        download_dataset(tmp_path / "data", metadata_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), metadata_bytes=archive.stat().st_size)
