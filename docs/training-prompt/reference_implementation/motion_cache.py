"""Lossless, bounded cache for native-resolution motion residuals only."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import struct
import time
import uuid
import zlib
from pathlib import Path

import numpy as np

CODEC_VERSION = "zlib6-u16-packbits-v1"


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a full lowercase SHA256")
    return value


def build_cache_key(*, source_sha256, frame_ids, flow_sha256, params_sha256,
                    rng_seed=17, codec_version=CODEC_VERSION):
    payload = {"source_sha256": _sha(source_sha256, "source_sha256"),
               "frame_ids": [int(x) for x in frame_ids],
               "flow_sha256": _sha(flow_sha256, "flow_sha256"),
               "params_sha256": _sha(params_sha256, "params_sha256"),
               "rng_seed": int(rng_seed), "codec_version": str(codec_version)}
    if len(payload["frame_ids"]) != 3:
        raise ValueError("frame_ids must contain real t-1,t,t+1 ids")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def encode_motion_blob(motion2, valid):
    motion2 = np.ascontiguousarray(motion2, dtype=np.uint16)
    valid = np.ascontiguousarray(valid, dtype=np.uint8)
    if motion2.shape != valid.shape or motion2.ndim != 2:
        raise ValueError("motion2 and valid must be equal-shape 2D arrays")
    nonzero = np.unique(valid[valid != 0])
    if len(nonzero) > 1:
        raise ValueError("valid mask must have zero plus at most one nonzero value for lossless packbits")
    valid_on = int(nonzero[0]) if len(nonzero) else 1
    if motion2.size and int(motion2.max()) > 510:
        raise ValueError("D*2 must stay in uint16 range 0..510")
    packed_valid = np.packbits(valid.reshape(-1) != 0, bitorder="little").tobytes()
    header = json.dumps({"shape": motion2.shape, "codec": CODEC_VERSION,
                         "valid_bytes": len(packed_valid), "valid_on": valid_on}, separators=(",", ":")).encode()
    raw = struct.pack("<I", len(header)) + header + motion2.tobytes() + packed_valid
    blob = zlib.compress(raw, level=6)
    return blob, hashlib.sha256(blob).hexdigest()


def decode_motion_blob(blob, checksum=None):
    if checksum is not None and hashlib.sha256(blob).hexdigest() != checksum:
        raise ValueError("motion blob checksum mismatch")
    raw = zlib.decompress(blob)
    size = struct.unpack("<I", raw[:4])[0]
    header = json.loads(raw[4:4+size])
    if header.get("codec") != CODEC_VERSION:
        raise ValueError("unsupported motion codec")
    shape = tuple(int(x) for x in header["shape"]); count = int(np.prod(shape)); offset = 4 + size
    motion2 = np.frombuffer(raw, dtype=np.uint16, count=count, offset=offset).reshape(shape).copy()
    packed = np.frombuffer(raw, dtype=np.uint8, count=int(header["valid_bytes"]), offset=offset + count * 2)
    valid = (np.unpackbits(packed, bitorder="little", count=count).reshape(shape).astype(np.uint8)
             * np.uint8(header["valid_on"]))
    if offset + count * 2 + int(header["valid_bytes"]) != len(raw):
        raise ValueError("motion blob length mismatch")
    return motion2, valid


class MotionCache:
    """Single-writer SQLite index plus atomic derived blob files."""
    def __init__(self, root, cache_gib=55, min_free_gib=20, *, max_bytes=None, min_free_bytes=None, read_only=False):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.blob_root = (self.root / "blobs").resolve(); self.blob_root.mkdir(exist_ok=True)
        self.max_bytes = int(max_bytes if max_bytes is not None else cache_gib * 1024**3)
        self.min_free_bytes = int(min_free_bytes if min_free_bytes is not None else min_free_gib * 1024**3)
        self.read_only = bool(read_only); self.db = self.root / "index.sqlite3"
        self.connection = sqlite3.connect(self.db)
        self.connection.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY,path TEXT UNIQUE,bytes INTEGER,checksum TEXT,shape TEXT,signature TEXT,atime REAL)")
        self.connection.commit()
        self.stats = {"lookups": 0, "hits": 0, "misses": 0, "writes": 0, "evictions": 0, "bypasses": 0}

    def _inside(self, path):
        path = Path(path).resolve()
        try: path.relative_to(self.blob_root)
        except ValueError as exc: raise RuntimeError(f"refusing cache path outside root: {path}") from exc
        return path

    def lookup(self, key, *, count_request=True):
        if count_request: self.stats["lookups"] += 1
        row = self.connection.execute("SELECT path,checksum,shape,signature FROM entries WHERE key=?", (key,)).fetchone()
        if row is None:
            if count_request: self.stats["misses"] += 1
            return None
        path = self._inside(row[0])
        if not path.is_file():
            if not self.read_only:
                self.connection.execute("DELETE FROM entries WHERE key=?", (key,)); self.connection.commit()
            if count_request: self.stats["misses"] += 1
            return None
        if not self.read_only:
            self.connection.execute("UPDATE entries SET atime=? WHERE key=?", (time.time(), key)); self.connection.commit()
        if count_request: self.stats["hits"] += 1
        return {"path": path, "checksum": row[1], "shape": tuple(json.loads(row[2])), "signature": row[3]}

    def read(self, key):
        entry = self.lookup(key)
        if entry is None: return None
        motion2, valid = decode_motion_blob(entry["path"].read_bytes(), entry["checksum"])
        return motion2, valid, entry

    def total_bytes(self):
        return int(self.connection.execute("SELECT COALESCE(SUM(bytes),0) FROM entries").fetchone()[0])

    def _evict(self, needed, protected=()):
        protected = {str(self._inside(path)) for path in protected}
        while self.total_bytes() + needed > self.max_bytes or shutil.disk_usage(self.root).free - needed < self.min_free_bytes:
            rows = self.connection.execute("SELECT key,path FROM entries ORDER BY atime ASC").fetchall()
            victim = next(((key, path) for key, path in rows if str(self._inside(path)) not in protected), None)
            if victim is None: raise OSError("cache/disk guard cannot be satisfied without evicting protected entries")
            key, path = victim; path = self._inside(path)
            if path.exists(): path.unlink()
            self.connection.execute("DELETE FROM entries WHERE key=?", (key,)); self.connection.commit()
            self.stats["evictions"] += 1

    def put_blob(self, key, blob, checksum, shape, signature, protected=()):
        if self.read_only: raise PermissionError("worker/read-only cache cannot write")
        if not re.fullmatch(r"[0-9a-f]{64}", key): raise ValueError("invalid cache key")
        if hashlib.sha256(blob).hexdigest() != checksum: raise ValueError("blob checksum mismatch")
        if len(blob) > self.max_bytes: raise OSError("single cache entry exceeds hard cache cap")
        decode_motion_blob(blob, checksum)
        existing = self.lookup(key, count_request=False)
        if existing is not None: return existing
        self._evict(len(blob), protected)
        final = self._inside(self.blob_root / f"{key}.mcz")
        temporary = self._inside(self.blob_root / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(blob); handle.flush(); os.fsync(handle.fileno())
            temporary.replace(final)
        finally:
            if temporary.exists(): temporary.unlink()
        self.connection.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?)", (key,str(final),len(blob),checksum,json.dumps(list(shape)),str(signature),time.time()))
        self.connection.commit()
        self.stats["writes"] += 1
        return {"path": final, "checksum": checksum, "shape": tuple(shape), "signature": str(signature)}

    def record_bypass(self):
        self.stats["bypasses"] = self.stats.get("bypasses", 0) + 1

    def cache_stats(self):
        return {**self.stats, "total_bytes": self.total_bytes(), "entry_count": int(self.connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]), "max_bytes": self.max_bytes, "min_free_bytes": self.min_free_bytes, "codec_version": CODEC_VERSION}
