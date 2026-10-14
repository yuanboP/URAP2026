# Data contract

## Pinned public release

Use Hugging Face dataset `Tyler0918/urap2026-drone-training` at immutable revision `aaf817f51dd670579a8e2c30a5db4930cf5eeb7f`. The CLI downloads `release_v3_curated/metadata.zip`, verifies its 78,781,652 bytes and SHA256 `58d4dc1084ba1281d121c93db4186e27491edd3192db9384da3771b95a3d12ac`, safely extracts it, verifies indexed members, and downloads only selected video objects by SHA.

| Source family | Retained videos | Role |
|---|---:|---|
| ARD100 | 100 | Train, validation, and test on disjoint videos |
| Historical DJI | 4 | Train, validation, and test on disjoint videos |
| NewDJI | 22 | Curated subset of the historical 28-video catalog |
| Total | 126 | Mixed-source held-out-video release |

The retained videos total 96,376,789,379 bytes. This is payload size, not total workspace capacity. Metadata extraction, a bounded motion cache, pseudo masks, weights, checkpoints, predictions, and evaluation outputs need additional space. Choose explicit caps and monitor free space.

The original NewDJI catalog contains 28 videos. Six were excluded by curation. Preserve that historical catalog and its provenance, but do not describe it as the current 126-video payload. Anti-UAV and NPS are not distributed in this release and are not part of the default formal evaluation.

## Canonical records

| Manifest | Records | Intended use |
|---|---:|---|
| `flow_train_common.jsonl` | 115,675 | Matched YOLO A/B training; every row has a valid triplet |
| `detection_train_complete.jsonl` | 115,749 | Complete curated training inventory |
| `validation.jsonl` | 26,424 | All curated validation detection records |
| `flow_validation.jsonl` | 26,399 | Motion-eligible source validation used by matched RGB/RGBD evaluation |
| `test.jsonl` | 76,257 | All curated test detection records |
| `flow_test.jsonl` | 76,211 | Motion-eligible formal test after all settings are frozen |
| `sam_positive_train.jsonl` | 114,025 | Positive-only box prompts for pseudo masks |

Canonical decoder indices are zero-based. `source_annotation_frame_id` and `source_frame_index_base` preserve annotation provenance and must never be subtracted from `source_frame_id`. Motion rows require `[t-1,t,t+1]` exactly and must remain within video bounds.

Each record preserves sample ID, source domain, sequence, source SHA, path, dimensions, frame count/rate, split, role, box coordinates, class IDs, motion eligibility, and completeness/provenance fields. `bbox_xyxy` is in original pixels and class `0` is drone.

An empty record is usable only when `role` is `negative_certified`. Unreviewed empty NewDJI annotations are unknown and excluded from negative supervision. A positive box does not certify all background regions on an incomplete frame.

## Partition meaning

Train, validation, and test are separated by source-video SHA. This prevents neighboring-frame and derivative leakage. Because every split contains a mixture of source families, the default result measures held-out videos within a mixed ARD/DJI corpus. It must not be labeled strict cross-domain generalization.

Validation selects checkpoints, thresholds, and allowed protocol settings. Test is read only after those decisions are frozen. Any external-domain study needs its own preregistered manifest, provenance, licensing check, and exclusion from training/tuning.

## Full and smoke downloads

The default `download` command materializes all 126 retained videos and canonical metadata. `--smoke --train-limit 8 --val-limit 4` writes deterministic balanced subset manifests outside the canonical directory. The smoke selection spans available source domains and includes a certified negative when possible. Smoke data verify integration only and cannot support performance conclusions.

Inspect `download_receipt.json` or `download_receipt_smoke.json` and `download_progress.jsonl`. A path existing is insufficient: hashes, bytes when published, manifest validation, exact decoder position, and native shape must pass.

## Motion and cache data

The fourth channel is raw numerical compensated difference normalized by 255. Invalid warp pixels are zero. Boundary/ineligible rows and compensation failures use D=0 with an explicit reason; RGB decode failure is fatal. The cache is derived, bounded, atomic, and disposable. It does not replace source verification.

## Weight provenance

Use one identical official YOLO11m initialization for both detector arms, official SAM 2.1 Large as pseudo-mask teacher, and official SAM 2.1 Tiny as the adapted student. Verify the weight receipt and SAM source commit before training. Never publish credentials, private source paths, or unlicensed source archives.
