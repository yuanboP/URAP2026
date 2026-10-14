# Training and evaluation task plan

`Implemented` means runnable code and tests exist. It does not mean the formal run is complete or that a gain was measured. [02_ARCHITECTURE_AND_EXPERIMENTS.md](02_ARCHITECTURE_AND_EXPERIMENTS.md) remains the binding research contract.

## T00 — Reproducible environment

**Status:** implemented workflow; formal environment still to freeze. **Priority:** P0.

Use `doctor` and [00_QUICKSTART.md](00_QUICKSTART.md). Record software versions, official SAM 2 commit, hardware, free disk, repository commit, and dependency lock. Acceptance requires CUDA forward/backward, CPU exact decode, one RGBD sample, one YOLO inference, one SAM inference, deterministic seeds, and verified artifact writes on the run host.

## T01 — Acquire and verify pinned assets

**Status:** downloader and receipts implemented; full acquisition remains a prerequisite. **Priority:** P0.

Download the pinned curated release and official weights. Preserve archive/member hashes, video bytes/SHA256, dimensions, frame counts, provenance, and revision. Acceptance requires 126 retained videos and every selected asset to verify and decode. Unreviewed empty annotations never become negatives.

## T02 — Freeze and audit partitions

**Status:** curated video-level partitions exist; independent pre-run audit required. **Priority:** P0.

Audit SHA and capture-group isolation across train, validation, and test. Report domains, videos, frames, boxes, certified negatives, completeness, and size bins. This is mixed-source held-out-video evaluation, not strict cross-domain holdout. Acceptance requires zero source-video overlap, no neighboring-frame leakage, valid canonical indices/boxes, and no test-informed selection.

## T03 — Freeze the motion residual

**Status:** implemented in `drone_training.data`. **Priority:** P0.

Use the published numerical residual and default local parameters: grid 36×21, 18 neighbors, radius 190, maximum shift 18, threshold 0.18, tracking scale 0.5. Acceptance requires exact position/shape decode checks, synchronized RGB/D/box transforms, inverse coordinates, bounded cache, D=0 with a reason on failure/boundary, and no visualization-based input.

## T04 — Unified A/B entrypoint

**Status:** implemented; formal training incomplete. **Priority:** P0.

Use `train-yolo`, `evaluate`, `calibrate`, and `compare`. A/B share initialization, P2 topology, RGB weights, presentations, augmentation seeds, effective batch, optimizer schedule, update count, validation schedule, and evaluator. B alone receives a zero-initialized fourth input slice. Acceptance requires strict checkpoint identities, independent best checkpoints, atomic recovery, original-coordinate predictions, COCO AP, validation-frozen thresholds, and paired video comparison.

## T05 — Smoke and interruption drills

**Status:** bounded real GPU training/recovery and reload verified; repeat on the execution host. See [verification](09_ENGINEERING_VERIFICATION.md). **Priority:** P0.

Run the explicit smoke commands, check finite losses/gradients and both arms, interrupt after a checkpoint, and resume with identical immutable inputs. Smoke is integration evidence only. Acceptance requires deterministic A/B presentation identity and readable receipts, logs, and checkpoints.

## T06 — Formal matched YOLO experiment

**Status:** not completed. **Priority:** P0.

Run all 115,675 common training frames for the frozen default of 10 epochs; validation and test are excluded from optimization. Calibrate arms independently on validation and evaluate untouched test. Acceptance includes complete lineage; no silent drops; matched budgets; AP50, AP50–95, operating metrics, FP/frame, domain/size slices, failures, throughput, and paired video bootstrap metrics. Negative or inconclusive findings are valid.

## T07 — SAM pseudo masks and Tiny decoder

**Status:** runnable implementation exists; formal 5-epoch run incomplete. **Priority:** P0.

Generate pseudo masks only from positive training boxes with official SAM 2.1 Large. Train the SAM 2.1 Tiny mask decoder with immutable base identity and recovery. Acceptance requires acceptance/rejection accounting, train-only lineage, no validation/test leakage, and predicted-box inference. Do not claim standalone detection or true mask accuracy without independent mask truth.

## T08 — Optional proposal-conditioned SAM head

**Status:** implementation exists; leakage-safe candidates and formal evaluation remain. **Priority:** P1.

Generate proposals out of fold by source video or from a separately held-out training partition. Only certified-negative or exhaustive training frames supply background labels; incomplete-frame unmatched regions are ignored. Acceptance requires documented proposal lineage, IoU ≥0.5 positives, IoU <0.3 negatives, an ignore band, class/regression losses, validation calibration, and comparison on identical automatic proposals.

## T09 — Lock and evaluate models

**Status:** evaluator implemented; formal locked evaluation incomplete. **Priority:** P0.

Freeze checkpoints, manifests, code/config hashes, and validation thresholds before test. Acceptance requires immutable test inputs, COCO artifacts, original-coordinate boxes, duplicate/missing/decode accounting, per-sequence outputs, and no test-selected setting.

## T10 — Controlled ablations

**Status:** proposed. **Priority:** P1. Run frozen one-factor comparisons for RGB/RGBD, compensation, fallback, normalization/suppression, SAM refinement, and the optional head with matched budgets.

## T11 — Deliver reproducible evidence

**Status:** implementation package present; formal result bundle incomplete. **Priority:** P0. Deliver environment, receipts, hashes, configs, checkpoints, logs, predictions, calibration, COCO results, paired comparison, failures, and [06_RESULT_TEMPLATE.md](06_RESULT_TEMPLATE.md).

## T12–T14 — Tracking follow-on

**Status:** unimplemented. T12 integrates frozen detector/refiner outputs, T13 adds loss/reacquisition and memory safety, and T14 performs causal sequential evaluation. Follow [08_TRACKING_INTEGRATION.md](08_TRACKING_INTEGRATION.md); the current CLI does not deliver these stages.
