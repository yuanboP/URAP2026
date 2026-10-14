"""Pinned, integrity-checked downloader for the curated public dataset release."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "Tyler0918/urap2026-drone-training"
DEFAULT_REVISION = "aaf817f51dd670579a8e2c30a5db4930cf5eeb7f"
METADATA_BYTES = 78_781_652
METADATA_SHA256 = "58d4dc1084ba1281d121c93db4186e27491edd3192db9384da3771b95a3d12ac"
RELEASE = "release_v3_curated"


def _hash(path: Path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, expected_sha: str, expected_bytes=None):
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise ValueError(f"size mismatch for {path}: expected {expected_bytes}, got {path.stat().st_size}")
    actual = _hash(path)
    if actual != expected_sha:
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected_sha}, got {actual}")


def _safe_extract(archive_path: Path, root: Path):
    root = root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"unsafe archive member: {member.filename}") from exc
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
            with archive.open(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
                target.flush(); os.fsync(target.fileno())
            temporary.replace(destination)


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _verify_archive_members(release: Path):
    index = release / "metadata_archive_members.jsonl"
    if not index.is_file():
        return 0
    checked = 0
    for row in _read_jsonl(index):
        relative = Path(row["archive_member"])
        candidate = (release.parent / relative).resolve()
        try:
            candidate.relative_to(release.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"unsafe metadata member path: {relative}") from exc
        if not candidate.is_file():
            raise ValueError(f"metadata archive member missing: {relative}")
        _verified(candidate, row["sha256"], row.get("bytes"))
        checked += 1
    return checked


def _balanced_first(records, limit):
    """Stable round-robin over domains; include a reviewed negative when present."""
    if not limit:
        return list(records)
    groups = {}
    for row in records:
        groups.setdefault(str(row.get("source_domain", row.get("dataset", "unknown"))), []).append(row)
    chosen = []
    depth = 0
    keys = sorted(groups)
    while len(chosen) < int(limit) and any(depth < len(groups[key]) for key in keys):
        for key in keys:
            if depth < len(groups[key]) and len(chosen) < int(limit):
                chosen.append(groups[key][depth])
        depth += 1
    negatives = [row for row in records if "negative" in str(row.get("role", "")).lower() and row.get("reviewed") is not False]
    if negatives and chosen and not any("negative" in str(row.get("role", "")).lower() for row in chosen):
        chosen[-1] = negatives[0]
    return chosen


def _write_subset(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _progress(path: Path, event, **fields):
    payload = {"time": time.time(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def download_dataset(data_root, repo_id=DEFAULT_REPO_ID, revision=DEFAULT_REVISION,
                     smoke=False, train_limit=8, val_limit=4, workers=4,
                     metadata_sha256=METADATA_SHA256, metadata_bytes=METADATA_BYTES):
    """Download metadata plus only videos referenced by the selected manifests."""
    del workers  # huggingface_hub handles resumable file transfers internally.
    root = Path(data_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "download_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    _progress(progress_path, "metadata_start", repo_id=repo_id, revision=revision)
    archive = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                   filename=f"{RELEASE}/metadata.zip", revision=revision,
                                   local_dir=str(root)))
    _verified(archive, metadata_sha256, metadata_bytes)
    _progress(progress_path, "metadata_verified", bytes=archive.stat().st_size, sha256=metadata_sha256)
    _safe_extract(archive, root)
    release = root / RELEASE
    metadata_members_verified = _verify_archive_members(release)
    train_source = release / "manifests" / "flow_train_common.jsonl"
    validation_source = release / "manifests" / "flow_validation.jsonl"
    if not train_source.is_file() or not validation_source.is_file():
        raise ValueError("metadata archive is missing canonical flow manifests")
    train_rows, validation_rows = _read_jsonl(train_source), _read_jsonl(validation_source)
    if smoke:
        train_rows = _balanced_first(train_rows, train_limit)
        validation_rows = _balanced_first(validation_rows, val_limit)
        train_manifest = root / "smoke" / "manifests" / "flow_train_common.jsonl"
        validation_manifest = root / "smoke" / "manifests" / "flow_validation.jsonl"
        _write_subset(train_manifest, train_rows); _write_subset(validation_manifest, validation_rows)
    else:
        train_manifest, validation_manifest = train_source, validation_source
    asset_manifest = release / "asset_manifest.jsonl"
    assets = ([row for row in _read_jsonl(asset_manifest) if row.get("retained") is not False]
              if asset_manifest.is_file() else [])
    expected_bytes = {row["sha256"]: row.get("bytes") for row in assets}
    if smoke:
        videos = {row["source_sha256"] for row in train_rows + validation_rows}
    else:
        if not assets:
            raise ValueError("metadata archive is missing or has empty asset_manifest.jsonl")
        videos = {row["sha256"] for row in assets}
    video_root = root / "videos"; video_root.mkdir(exist_ok=True)
    downloaded = 0
    for position, sha in enumerate(sorted(videos), 1):
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(f"invalid video SHA in manifest: {sha!r}")
        destination = video_root / f"{sha}.mp4"
        byte_count = expected_bytes.get(sha)
        valid_existing = destination.is_file()
        if valid_existing and byte_count is not None:
            valid_existing = destination.stat().st_size == int(byte_count)
        if valid_existing:
            valid_existing = _hash(destination) == sha
        if not valid_existing:
            fetched = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                           filename=f"videos/{sha}.mp4", revision=revision,
                                           local_dir=str(root)))
            if fetched.resolve() != destination.resolve():
                shutil.copy2(fetched, destination)
        _verified(destination, sha, byte_count)
        downloaded += 1
        _progress(progress_path, "video_verified", done=position, total=len(videos),
                  sha256=sha, bytes=destination.stat().st_size)
        print(f"[download] verified video {position}/{len(videos)} {sha[:12]}", flush=True)
    receipt = {"repo_id": repo_id, "revision": revision, "smoke": bool(smoke),
               "train_records": len(train_rows), "validation_records": len(validation_rows),
               "videos_downloaded": downloaded, "train_manifest": str(train_manifest.resolve()),
               "validation_manifest": str(validation_manifest.resolve()), "metadata_sha256": metadata_sha256}
    receipt["metadata_members_verified"] = metadata_members_verified
    receipt["progress_jsonl"] = str(progress_path)
    receipt_path = root / ("download_receipt_smoke.json" if smoke else "download_receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt"] = str(receipt_path)
    _progress(progress_path, "complete", videos_downloaded=downloaded, receipt=str(receipt_path))
    return receipt
