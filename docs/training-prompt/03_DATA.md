# Data handoff for the new training project

Snapshot date: **2026-10-11**. This document records the dataset inventory known on that date. It does not guarantee that any cloud object remains available today.

## What is and is not in this repository

This public Git branch contains reference code, documentation, and inventory files. It does **not** contain raw videos, raw annotations, generated manifests, model checkpoints, or presentation exports. The project owner delivers the authorized original annotations, labels, and dataset assets separately. Mount those assets under the logical roots below before running preparation code. GitHub alone is insufficient to reproduce the datasets.

The annotation archive is referenced by the portable ASCII alias `external_assets/annotations/annotations_2607.zip`. Its original snapshot metadata is:

- 424,042 bytes
- SHA-256 `dcfcb9c1a79ceecf39b127e57abe03ba13ec73c602571147687267b450f69e4a`
- 28 JSON files
- 14,276 annotation records, including 13,373 drone boxes and 918 empty-box records

Verify the archive size and hash after receipt.

## NewDJI28

The authorized Google Drive discovery folder was:

- Folder: https://drive.google.com/drive/folders/1N1TRgMt608bJXheu3Y9W6V5qVEX9QEai
- Folder ID: `1N1TRgMt608bJXheu3Y9W6V5qVEX9QEai`

The 2026-10-11 catalog recorded 28 raw MP4 files totaling **79,333,410,303 bytes (73.885 GiB)** and 28 compensated-difference visualization MP4 files. Those visualizations are diagnostic media; they are not defined numerical motion tensors and must not replace raw-video decoding or motion-feature generation.

Use the individual Drive file IDs in `new_video_inventory.csv`, then verify each received file against the catalog byte count and any recorded SHA-256. A folder listing is only a discovery aid and may drift.

The frozen capture-date split is:

| Capture group | Split |
|---|---|
| 20260626 and 20260706 | train |
| 20260702 | validation |
| 20260704 | test |

Positive-record totals are train **6,967** across 11 videos, validation **3,893** across 9 videos, and test **2,498** across 5 videos, for **13,358** positive records. The 2026-10-11 audit found **6,956** triplet-eligible training rows. Treat this as a planning count and re-audit it when generating fresh manifests. Three additional videos (003, 013, and 016) have zero positive annotations and are excluded from those positive-row counts.

Use the complete audited training split for the fresh project. If a smoke-test subset is needed, generate it anew from the training split and record its sampling method and seed; do not inherit an earlier reduced selection.

Only 222 annotations were marked reviewed, all from video 007 and one capture date. Treat precision and recall on these labels as provisional. The 918 empty-box records from 007 have unresolved semantics; do not treat them as certified negatives. Do not treat the three zero-positive videos as negative videos without a separate label audit.

A second authorized historical discovery folder is https://drive.google.com/drive/folders/1kFVI2KfZ7JmRKXoLFEuzEZ42chu47bV2 (ID `1kFVI2KfZ7JmRKXoLFEuzEZ42chu47bV2`). Deduplicate by source-video identity, capture group, byte size, and SHA-256 before combining it with NewDJI28.

## Legacy and public-domain assets

The full source-training planning inventory contains **107,561** rows across 53 videos: **105,957 ARD100** rows and **1,604 DJI** rows. The inventory marks 107,032 rows positive and 529 as certified negative; verify those annotation semantics during T02. Adding all 6,956 eligible NewDJI28 training rows gives a planning total of **114,517**. Recompute this total from the fresh manifest after deduplication, boundary checks, and label review; it is not a fixed quota or a completed-training count.

`legacy_video_inventory.csv` identifies the exact 67 legacy videos previously selected: 53 training videos and a 14-video source-validation candidate pool, totaling **18,819,068,978 bytes** before any content deduplication. The legacy training planning count is **107,561** records and must be re-audited when fresh manifests are generated. Rebuild validation from all audited eligible frames in the 14-video pool. Inventory paths are portable logical external-asset paths.

Additional portable logical roots may be mounted as:

- `external_assets/public/ARD100` (correct dataset name: ARD100)
- `external_assets/public/Anti-UAV300`
- `external_assets/public/Drone-detection-dataset`
- `external_assets/public/AOT_train_adapt_pool_v2`
- `external_assets/public/TransVisDrone_NPS`
- `external_assets/public/TriCross-D2D`

These datasets were used during prior development and must not be described as untouched independent test sets. Preserve their licenses, original annotations, class mappings, and official split files in the separately delivered asset package. A small prepared cross-domain subset cannot substitute for a complete benchmark.

## Leakage controls for the fresh experiment

1. Freeze split membership by capture group or original video before extracting frames, flow, residuals, crops, tiles, pseudo-masks, or augmentations.
2. Keep every derivative of a source video in the same split.
3. Use `(dataset, capture_group, source_video_id)` as the preferred split key, and assert that source-video IDs do not overlap across train, validation, and test.
4. Use validation only under a predeclared selection policy. Do not use test labels to tune thresholds, epochs, prompts, motion methods, or hard-case selection.
5. Establish a new run contract with fixed code revision, official pretrained initialization, seeds, manifest hashes, update budget, image size, optimizer, thresholds, and evaluation manifests.
6. Start every new model from the official generic pretrained weights listed in `weights_inventory.csv`, and record those exact initialization identities.

## Receipt checklist

Before materialization or training:

- obtain the separately delivered authorized asset package;
- verify `annotations_2607.zip` against its byte count and SHA-256;
- verify every raw video byte count and every available SHA-256 in both video inventories;
- confirm that all manifest paths resolve relative to the configured logical roots;
- audit split disjointness at source-video level;
- record any unavailable Drive object as unavailable on the receipt date rather than rewriting the 2026-10-11 snapshot;
- keep access tokens, cookies, credentials, and private sharing URLs out of logs and Git.

`experiment_config.template.yaml` is a labeled specification template. It is non-executable until the T00/T04 adapter validates and maps it into the actual training entry points. Do not launch a GPU job from the template alone.
