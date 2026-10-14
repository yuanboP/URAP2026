# Tiny-drone training and evaluation package

This package provides a runnable matched experiment for tiny-drone detection. Arm A is YOLO11m-P2 with RGB input. Arm B uses the same architecture, initialization, sample order, augmentation seeds, optimizer schedule, and evaluator, with one additional numerical camera-motion-compensated residual channel. It also provides a SAM 2.1 Large pseudo-mask teacher workflow and SAM 2.1 Tiny mask-decoder adaptation.

Start with [00_QUICKSTART.md](00_QUICKSTART.md). It is the command source of truth for installation, pinned data and weight downloads, full and smoke training, recovery, calibration, COCO evaluation, paired comparison, and SAM commands. Use `python -m drone_training COMMAND --help` for current CLI options.

## Implementation boundary

The `drone_training` package implements pinned dataset download and verification, canonical manifest loading, exact frame decode, RGBD construction, deterministic paired augmentation, bounded motion caching, matched YOLO A/B training and recovery, original-coordinate COCO evaluation, validation-only threshold calibration, paired video bootstrap comparison, SAM pseudo-mask generation, SAM Tiny decoder training/inference, and an optional proposal-conditioned SAM feature head.

Implementation availability is not a successful experiment. Formal full-data training, frozen validation selection, untouched test evaluation, and scientific review must still be completed. [Actual training/recovery verification](09_ENGINEERING_VERIFICATION.md) covers YOLO and SAM2 on real samples; it is engineering evidence, not a final accuracy result. T12–T14—tracking integration, loss/reacquisition controls, and sequential tracking evaluation—remain unimplemented follow-on work.

## Public curated dataset

The pinned release is `Tyler0918/urap2026-drone-training` at revision `aaf817f51dd670579a8e2c30a5db4930cf5eeb7f`.

- 126 retained videos: 100 ARD100, 4 historical DJI, and 22 NewDJI videos
- 96,376,789,379 retained video bytes, approximately 96.4 GB decimal
- 115,749 training records; 115,675 motion-eligible common-frame training records
- 26,424 validation records and 76,257 test records
- 114,025 positive training records for SAM pseudo-mask generation

This is a mixed-source held-out-video evaluation, not a strict cross-domain benchmark. Training, validation, and test contain disjoint source videos but mixtures of ARD100 and DJI-family material. Anti-UAV and NPS are not in this Hugging Face release. The historical 28-video NewDJI catalog remains provenance; the current curated payload retains 22 of those videos and combines them with ARD100 and historical DJI sources.

The 96.4 GB figure describes retained videos, not total disk required. Budget additional bounded space for metadata extraction, motion cache, pseudo masks, checkpoints, predictions, and evaluation artifacts. Smoke data are for integration checks only.

## Scientific rules

- Keep source videos disjoint across train, validation, and test.
- Use canonical zero-based decoder indices without subtracting annotation provenance offsets.
- Feed numerical compensated residuals, never colorized motion visualizations.
- Preserve RGB and record explicit zero-motion fallback on boundary or compensation failure.
- Match A/B presentation order, augmentation, updates, initialization, and evaluation.
- Select thresholds on validation and freeze them before test.
- Treat empty frames as negatives only when certified.
- Use predicted proposals for SAM evaluation; ground-truth prompts are oracle diagnostics only.
- Do not claim segmentation accuracy from box labels or standalone detection from a prompt-conditioned mask decoder.

The research contract remains in [02_ARCHITECTURE_AND_EXPERIMENTS.md](02_ARCHITECTURE_AND_EXPERIMENTS.md). Files under `reference_implementation/` are immutable published inputs used by runnable adapters.
