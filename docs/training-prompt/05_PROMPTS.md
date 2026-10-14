# Runnable formal-training prompt

Copy the following prompt to the engineer or agent executing the experiment. The runnable implementation already exists; do not ask them to reimplement it.

```text
Run the formal tiny-drone experiment in this repository using docs/training-prompt/00_QUICKSTART.md as the command source of truth and docs/training-prompt/02_ARCHITECTURE_AND_EXPERIMENTS.md as the frozen research contract.

Use the existing drone_training package. Do not rewrite the immutable files under docs/training-prompt/reference_implementation/. First run doctor, record the exact repository commit, environment, CUDA/GPU/VRAM, free disk, dependency versions, official SAM 2 commit, and weight receipts.

Download and verify the pinned Hugging Face dataset Tyler0918/urap2026-drone-training at revision aaf817f51dd670579a8e2c30a5db4930cf5eeb7f. The current release has 126 videos: 100 ARD100, 4 historical DJI, and 22 NewDJI. It is a mixed-source held-out-video dataset, not strict cross-domain evaluation. Anti-UAV and NPS are absent. Preserve the historical 28-video NewDJI inventory only as provenance. Verify that source-video SHA sets do not overlap across train, validation, and test.

Before a long run, execute the explicit smoke download and smoke training/recovery drill. Confirm exact frame decode, nonzero numerical D on moving eligible frames, D=0 reasons on boundary/failure, synchronized RGB/D/boxes, finite losses/gradients, updates in both arms, atomic checkpoints, and deterministic resume. Do not report smoke performance as a result.

Run the matched YOLO experiment with the provided train-yolo entrypoint. Arm A is RGB YOLO11m-P2. Arm B is the identical model with the same RGB initialization and a zero-initialized fourth motion-channel slice. Both arms must see the same 115,675 common training records on each full pass, augmentation seeds, effective batch, optimizer schedule, updates, validation schedule, and evaluator. Use the formal default of 10 epochs; do not replace it with a short --steps run. Save last.pt and independent best_A.pt/best_B.pt.

Evaluate both arms on all 26,399 motion-eligible records in flow_validation.jsonl. The broader detection validation manifest has 26,424 records and is a separately labeled boundary/no-motion evaluation. Select checkpoints and calibrate each threshold on validation only. Freeze checkpoint/config/manifest/calibration hashes. Then evaluate all 76,211 untouched motion-eligible records in flow_test.jsonl in original coordinates; the broader detection test manifest has 76,257 records. Export original-coordinate prediction JSONL plus provenance and official COCO metrics, preserve low-score predictions for AP, and record decode/fallback/missing/duplicate counts. Compare A/B with paired source-video bootstrap operating metrics. Report held-out-video performance, not strict cross-domain performance. Negative or inconclusive results are valid.

Run the provided SAM workflow separately. Generate training-only pseudo masks for all 114,025 positive training records with official SAM 2.1 Large. Inspect accepted/rejected masks, then train the official SAM 2.1 Tiny mask decoder for the formal default of 5 epochs. Evaluate it only as a box-prompted refiner on automatic detector proposals. Do not claim standalone drone detection or true segmentation accuracy without independent mask ground truth.

The proposal-conditioned SAM feature classification/regression head is optional. Use it only if its candidate JSONL was generated out of fold by source video or from a separately held-out training partition. Background labels require certified negatives or exhaustive training frames; unmatched proposals on incomplete frames are ignored. Use IoU >= 0.5 positives, IoU < 0.3 negatives, and ignore the middle band. Compare on identical automatic proposals with and without the head.

Track every command, receipt, data/config/code hash, checkpoint, training log, validation decision, prediction artifact, calibration, COCO result, failure, and runtime measure. Fill 06_RESULT_TEMPLATE.md without guaranteeing improvement. Stop at T11. T12-T14 tracking integration, reacquisition/memory controls, and sequential tracking evaluation are not implemented and must not be claimed.
```

## Review prompts

Use these narrower prompts only after the main run exists.

### Independent data review

```text
Independently verify the pinned revision, archive/member/media hashes, 126-video inventory, canonical counts, zero-based frame contract, certified-negative semantics, and source-video split isolation. Confirm the release is mixed-source held-out-video evaluation. Produce a signed audit without editing training artifacts.
```

### Independent evaluation review

```text
Verify that A/B checkpoints came from matched training, thresholds were selected only on validation, test predictions cover identical sample identities, boxes are mapped to original coordinates, and COCO/operating metrics can be reproduced. Check that paired resampling is by source video and that no AP confidence interval is claimed without a valid AP resampling implementation.
```

### SAM review

```text
Verify pseudo masks use positive training rows only, SAM base-weight identity is frozen, only the Tiny mask decoder was adapted by default, automatic proposals are used for evaluation, mask quality is not described as drone confidence, and any proposal head uses leakage-safe training candidates plus certified background supervision.
```
