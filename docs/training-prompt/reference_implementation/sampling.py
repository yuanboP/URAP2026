"""Deterministic, model-independent epoch sampling for the full-data experiment."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


_BIN_NAMES = ("lt4", "4-8", "8-16", "16-32", "ge32")


def _video(record):
    return str(record.get("source_video_path") or record.get("sequence_id") or record.get("source_domain"))


def _size_bin(record, size=1280):
    boxes = np.asarray(record.get("bbox_xyxy", []), dtype=np.float64).reshape(-1, 4)
    if not len(boxes):
        return "empty"
    scale = size / max(float(record["source_width"]), float(record["source_height"]))
    short = float(np.min(np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])) * scale)
    return _BIN_NAMES[np.searchsorted([4, 8, 16, 32], short, side="right")]


def _balanced_indices(indices, records, count, rng, use_bins=False):
    if count <= 0:
        return []
    by_video = defaultdict(list)
    for i in indices:
        by_video[_video(records[i])].append(i)
    if not by_video:
        raise ValueError("requested supplemental samples but no eligible records exist")
    pools = {}
    for video, members in by_video.items():
        if use_bins:
            groups = defaultdict(list)
            for i in members:
                groups[_size_bin(records[i])].append(i)
            shuffled = {key: list(rng.permutation(value)) for key, value in groups.items()}
            keys = sorted(shuffled, key=lambda key: (_BIN_NAMES.index(key) if key in _BIN_NAMES else 99, key))
            # Repeat bins, not their populations: rare native-size bins receive
            # the same exposure as common bins. Samples within each bin cycle
            # through seeded permutations.
            per_bin_cursor = Counter()
            ordered = []
            target_len = max(count, len(members))
            for turn in range(target_len):
                key = keys[turn % len(keys)]
                pool = shuffled[key]
                cycle, offset = divmod(per_bin_cursor[key], len(pool))
                if offset == 0 and cycle:
                    pool[:] = rng.permutation(pool).tolist()
                ordered.append(pool[offset])
                per_bin_cursor[key] += 1
            pools[video] = ordered
        else:
            pools[video] = list(rng.permutation(members))
    videos = list(rng.permutation(sorted(pools)))
    cursors = Counter()
    chosen = []
    for turn in range(count):
        video = videos[turn % len(videos)]
        pool = pools[video]
        chosen.append(pool[cursors[video] % len(pool)])
        cursors[video] += 1
        if turn % len(videos) == len(videos) - 1:
            videos = list(rng.permutation(videos))
    return chosen


def build_epoch_plan(records, epoch, seed=17, positive_replay_fraction=.10,
                     region_negative_fraction=.10, certified_negative_fraction=.05):
    """Return shuffled sample indices, augmentation seeds, view kinds, and audit data.

    Every inventory row is included once. Supplemental selection depends only on
    immutable source metadata and seeded permutations, never model results.
    """
    records = list(records)
    sample_ids = [record.get("sample_id") for record in records]
    if any(sample_id is None for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("inventory sample_id values must be present and unique")
    if any(value < 0 for value in (positive_replay_fraction, region_negative_fraction,
                                   certified_negative_fraction)):
        raise ValueError("sampling fractions must be nonnegative")
    positives = [i for i, r in enumerate(records) if r.get("role") == "positive"]
    negatives = [i for i, r in enumerate(records) if r.get("role") == "negative_certified"]
    ard_positives = [i for i in positives if records[i].get("source_domain", records[i].get("dataset")) == "ARD100"]
    npos = len(positives)
    replay_n = math.ceil(positive_replay_fraction * npos)
    region_n = math.ceil(region_negative_fraction * npos)
    certified_target = math.ceil(certified_negative_fraction * npos)
    certified_repeat_n = max(0, certified_target - len(negatives))
    root = np.random.SeedSequence([int(seed), int(epoch)])
    select_ss, aug_ss, shuffle_ss = root.spawn(3)
    select_rng = np.random.default_rng(select_ss)
    replay = _balanced_indices(positives, records, replay_n, select_rng, use_bins=True)
    regions = _balanced_indices(ard_positives, records, region_n, select_rng) if region_n else []
    negative_repeats = _balanced_indices(negatives, records, certified_repeat_n, select_rng) if certified_repeat_n else []
    index = np.asarray(list(range(len(records))) + replay + regions + negative_repeats, dtype=np.int64)
    kinds = np.asarray([0] * len(records) + [1] * replay_n + [2] * region_n + [0] * certified_repeat_n,
                       dtype=np.uint8)
    is_base = np.asarray([True] * len(records) + [False] * (replay_n + region_n + certified_repeat_n),
                         dtype=np.bool_)
    aug_rng = np.random.default_rng(aug_ss)
    aug_seed = aug_rng.integers(0, np.iinfo(np.int64).max, size=len(index), dtype=np.int64)
    order = np.random.default_rng(shuffle_ss).permutation(len(index))
    index, kinds, aug_seed, is_base = index[order], kinds[order], aug_seed[order], is_base[order]
    multiplicity = np.bincount(index, minlength=len(records)).astype(np.int64)
    base_multiplicity = np.bincount(index[is_base], minlength=len(records)).astype(np.int64)
    if not np.all(base_multiplicity == 1):
        raise AssertionError("internal error: base coverage is not exactly once per inventory record")
    exposure = Counter()
    for i, kind in zip(index.tolist(), kinds.tolist()):
        role = "region_negative" if kind == 2 else records[i].get("role", "unknown")
        exposure[(role, _video(records[i]), _size_bin(records[i]))] += 1
    audit = {
        "base_coverage": {"count": int(is_base.sum()), "unique_count": int(np.count_nonzero(base_multiplicity)),
                          "all_once": bool(np.all(base_multiplicity == 1)), "sample_ids_unique": True},
        "base_multiplicity": base_multiplicity.tolist(),
        "multiplicity": multiplicity.tolist(),
        "multiplicity_by_sample_id": {str(record.get("sample_id", i)): int(multiplicity[i])
                                      for i, record in enumerate(records)},
        "supplemental_counts": {"positive_replay": replay_n, "region_negative": region_n,
                                "certified_negative_repeat": certified_repeat_n},
        "exposure": [{"role": key[0], "video_id": key[1], "size_bin": key[2], "count": value}
                     for key, value in sorted(exposure.items())],
        "policy": "source metadata only; video/size round-robin; no model-adaptive mining",
    }
    audit["role_exposure"] = dict(sorted(Counter(
        "region_negative" if kind == 2 else records[i].get("role", "unknown")
        for i, kind in zip(index.tolist(), kinds.tolist())).items()))
    audit["video_exposure"] = dict(sorted(Counter(_video(records[i]) for i in index.tolist()).items()))
    audit["bin_exposure"] = dict(sorted(Counter(_size_bin(records[i]) for i in index.tolist()).items()))
    return {"sample_index": index, "aug_seed": aug_seed, "view_kind": kinds,
            "is_base": is_base, "audit": audit}
