# Execution prompts

These English prompts can be assigned to a teammate or an executing AI agent. They describe a brand-new training program. Start YOLO from standard generic `yolo11m.pt`, SAM teacher from official SAM2.1 Hiera Large, and SAM student from official SAM2.1 Hiera Tiny. Checkpoint recovery is allowed only within a new run created by this program.

The repository's `reference_implementation/` tree is reference material, not a complete product. Inspect the real parser and tests before running a command. The unified fair A/B trainer, SAM classification/regression head, and temporal evaluation path are implementation tasks; do not fabricate flags or claim that they already run.

## Master prompt: T00–T14

```text
Execute the drone-detection training program in docs/training-prompt from T00 through T14, honoring task dependencies and acceptance criteria.

This is brand-new training. Initialize both YOLO variants from the same standard generic Ultralytics yolo11m.pt. Initialize SAM with the official SAM2.1 Hiera Large teacher and Hiera Tiny student checkpoints. Create new optimizer state and output artifacts for each experiment. Resume only a checkpoint produced by the same new immutable run contract.

Treat reference_implementation/ as code evidence and reusable building blocks. It is not a turnkey system. Inspect every parser before issuing commands. T04 must implement and test the unified matched A/B entry point. T08 must implement and test the SAM proposal-conditioned drone/background classifier and box regressor. T14 must implement the complete-video temporal evaluator. Do not invent executable commands for pending work.

Obtain the datasets externally; raw videos, label archives, manifests containing sensitive local paths, generated caches, pseudo-masks, and weights are not stored in GitHub. Verify every received asset against its inventory, preserve source IDs and hashes, and report missing or corrupt assets before training.

Complete T00 environment and public-weight verification; T01 external data intake; T02 capture-group split and annotation audit; T03 numerical motion-channel verification; T04 unified A/B implementation; T05 smoke and interruption/recovery test; T06 matched A/B training; T07 SAM Large-to-Tiny weak mask adaptation; T08 SAM candidate classification and box-regression head; T09 frozen fair evaluation; T10 ablations; T11 reproducible delivery; T12 video API integration; T13 bounded recovery/memory; and T14 temporal evaluation.

For A versus B, change only the input defined by the experiment: A receives RGB and B receives RGB plus the frozen numerical compensated residual D. Use the same P2 topology, compatible transferred pretrained parameters, samples, presentation order, augmentations, effective batch, optimizer, schedule, maximum optimizer steps, validation events, checkpoint rule, seed, and evaluation inputs. Initialize B's added input-channel weights deterministically to zero and record the transfer report. Equal optimizer steps is the primary training budget; also report sample coverage.

Split by capture group or source video before deriving triplets, residuals, crops, pseudo-masks, or proposals. Never random-split neighboring frames. Exclude unreviewed empty annotations from negative supervision. Use background labels only when a frame is confirmed exhaustively annotated. Preserve annotation completeness and provenance in every derived manifest.

Use the numerical raw D/255 channel produced by the reviewed motion code. Never substitute a colorized residual video or per-frame visualization scaling. Apply geometric augmentation identically to RGB, D, and boxes. Log D=0 fallback events. Record that the symmetric residual requires a future frame.

Use training boxes only to prompt the SAM2.1 Large teacher for weak pseudo-masks. Train the Tiny decoder from those accepted pseudo-masks and compare it with the same frozen Tiny base under identical automatic YOLO proposals. At evaluation, SAM receives no ground-truth prompts. Its mask-quality prediction is not drone-class confidence.

T08 must train a real proposal-conditioned drone/background classification and box-regression head from SAM RGB features. Define positive, negative, and ignore matching before training. Generate realistic out-of-fold or separately held-out training proposals. Unknown candidates in incompletely annotated frames are ignore, never automatic background. The inference API must require only RGB/video and automatic proposals.

Freeze checkpoints, manifests, thresholds, NMS, proposals, and failure fallback before test evaluation. Select thresholds only on source validation. Evaluate all systems on identical eligible frames and retain raw predictions. Report per-domain and per-video AP50, AP50-95, precision, recall, F1, FP/frame, target-size recall, proposal recall@K, paired video-level confidence intervals, and stage/end-to-end latency. Label custom AP accurately; use official COCO evaluation before calling a metric COCO AP. Box-only ground truth does not support mask Dice, mask IoU, or J&F claims.

T12 must expose deterministic full-video inference with timestamps and original-coordinate outputs and connect `DetectionCandidate` records to the existing `ConstantVelocityTracker`. T13 must add bounded loss, reacquisition, and memory controls with explicit state, expiry, reset, and confidence rules. T14 must compare detector-only, tracker, optional SAM2 memory, and Action Bank arms on the same complete videos. Do not tune temporal rules on test videos.

Maintain a run contract, hashes, environment lock, logs, atomic checkpoints, optimizer/RNG/sampler state, sample coverage, raw predictions, metrics, and failure examples. Report a job as running only after verifying its process, advancing progress, and GPU signal. Negative or null results are valid. Never alter the protocol after seeing test results to manufacture an improvement.
```

## Prompt: T00–T03 foundations

```text
Own T00 through T03.

Create a reproducible CUDA/Python environment and record installed versions and hardware. Acquire standard generic yolo11m.pt plus official SAM2.1 Hiera Large and Tiny checkpoints, calculate SHA-256, load each once, and save provenance. These are the only pretrained starting points.

Intake the external videos and annotations without placing binaries in Git. Verify file size, SHA-256, decode, frame count, resolution, frame-index convention, class mapping, and box bounds. Build canonical records with source and capture-group lineage. Produce explicit missing, corrupt, ambiguous, and excluded lists.

Freeze train, source-validation, and test partitions by capture group or source video before generating derivatives. Prove zero overlap. Summarize each domain by videos, eligible frames, boxes, reviewed negative frames, annotation completeness, and target-size bins. Do not treat an empty record or zero-label video as background unless annotation completeness has been reviewed.

Validate the reference numerical motion path. Confirm previous and next frames align to the center frame, invalid warp areas become zero, model input is clip(D/255, 0, 1), and RGB/D/boxes share every geometric transformation. Save 20-50 representative overlays plus numerical distributions and failure rates. Freeze the motion config and code hash. Record the one-frame-lookahead requirement.

Deliver environment and weight audits, canonical manifests, split and annotation audits, motion configuration, visual checks, and a clear list of blockers. Do not begin model training.
```

## Prompt: T04–T06 matched YOLO training

```text
Own T04 through T06. You are implementing a new matched experiment, not continuing an earlier trained model.

Study the corresponding files under reference_implementation/, then build a single tested training entry point for RGB-only YOLO11m-P2 A and RGB+D YOLO11m-P2 B. Start both from the same standard generic yolo11m.pt and emit a key-by-key transfer report. Initialize B's added channel to zero. Do not add fictional flags to a reference script; publish the real command only after the parser exists.

Freeze one presentation plan so A and B see identical sample IDs, order, augmentation decisions, image size, effective batch size, optimizer, learning-rate schedule, maximum optimizer steps, validation events, and checkpoint-selection rule. A must not read D. B must use the frozen numerical D/255 channel. Log motion failures and sample/domain coverage.

Add meaningful tests for channel construction, synchronized transformations, weight transfer, presentation equality, immutable run identity, atomic checkpoint saving, and exact recovery of optimizer, scheduler, scaler, RNG, sampler cursor, and step count.

Run a 20-100-step representative smoke test for each arm. Inspect inputs and predictions, verify finite losses, interrupt both runs, resume them, and compare against uninterrupted controls. Fix every P0 data or recovery failure before formal training.

Train the matched A/B arms to the same preregistered optimizer-step limit. An extension is permitted only by a rule frozen before test evaluation and applied equally to both arms. Select best checkpoints by the same source-validation rule. Save last and best separately, retain optimizer state, and report step count plus unique/repeated sample coverage. Do not inspect test results during training or checkpoint selection.
```

## Prompt: T07 SAM2.1 mask adaptation

```text
Own T07.

Use the official SAM2.1 Hiera Large checkpoint as teacher and official Hiera Tiny checkpoint as both the frozen baseline and adapted student's base. Adapt the reviewed sam_adapt.py reference to the repository and verify its actual pseudo and train parsers before execution.

Give the Large teacher only training RGB and training human boxes as prompts. Generate weak pseudo-masks, preserving sample, video, domain, box, teacher checkpoint hash, quality fields, and accept/reject/failure reason. Audit acceptance by domain and target size so tiny objects are not silently removed. Never generate training supervision from validation or test labels.

Train the Tiny mask decoder from accepted pseudo-masks while keeping the rest frozen for the baseline experiment. Save complete configuration and decoder weights together with the required Tiny base identity. Recovery may use only a checkpoint from this same new student run.

Evaluate frozen and adapted Tiny with exactly the same automatic detector proposals, proposal order, cap, RGB inputs, box conversion, and fallback. Never use a ground-truth prompt in the main result. Label any ground-truth-prompt diagnostic as oracle. Do not interpret mask quality as drone probability or claim a semantic detector has been trained.
```

## Prompt: T08 SAM candidate head

```text
Own T08. This capability is pending and requires implementation.

Build a proposal-conditioned drone/background classification head and bounding-box regression head over SAM2 RGB features. Start with the SAM image encoder frozen. Define the ROI/box feature, losses, positive/negative/ignore matching, sampling, score calibration, and original-coordinate box decoding in a reviewed experiment contract.

Generate training proposals out of fold by source video, or from a separately frozen proposal-training partition, so the head sees realistic detector errors without leakage. Use background targets only in exhaustively annotated training frames or reviewed negative frames. Treat unknown candidates in incomplete frames as ignore.

Implement tests and an inference smoke path that receives only RGB/video and automatic candidates and returns drone confidence, corrected xyxy, and optional mask. It must not read ground truth. Compare classification-only, regression-only, combined head, and the unmodified proposal baseline. Keep encoder adapters or LoRA as a later isolated ablation.

Publish the real CLI and reproducible command only after the code, parser, tests, and no-ground-truth smoke test pass. Until then, state explicitly that the repository contains no executable T08 trainer.
```

## Prompt: T09–T11 independent evaluation and delivery

```text
Own T09 through T11 independently from model training decisions.

Freeze model hashes, test manifests, source-validation threshold selection, NMS, candidate cap, SAM fallback, metric definitions, and comparison arms before opening test labels. The required arms are new-run A, new-run B, B plus frozen Tiny, B plus adapted Tiny, and B plus the T08 candidate head when available. Use identical eligible frames and automatic proposal inputs.

Audit the reference evaluator before reuse. Adapt or replace interfaces that encode a different comparison. Preserve raw low-score predictions for AP. If the implementation is a custom 101-point single-class AP, name it accurately. Add official COCO evaluation and a controlled consistency fixture before reporting COCO AP.

Report per domain and per video: TP, FP, FN, precision, recall, F1, FP/frame, AP50, AP50-95, predefined target-size recall, proposal recall@K, fallback/failure counts, p50/p95 stage latency, total latency including motion and future-frame wait, GPU memory, and hardware. Compute paired confidence intervals by video or capture group. Do not treat adjacent frames as independent bootstrap units.

Separate proposal misses, detector scoring errors, SAM refinement regressions, head classification rejections, box-correction effects, and decode failures. Never tune a threshold or prompt after viewing test results. Preserve negative, zero, and degraded results.

For T10, run only preregistered single-factor ablations with equal optimizer-step budgets where retraining is required. For T11, deliver code, real commands, environment lock, public-weight provenance, data/split hashes, model identities, configs, raw predictions, metrics, confidence intervals, failure examples, and limitations. Raw data and large binaries remain external. Confirm that another person can reproduce one-video inference and the offline metric calculation from the delivered materials.
```

## Prompt: T12–T14 video integration, recovery, and temporal evaluation

```text
Own T12 through T14 after the selected frame-level checkpoints and thresholds are frozen.

For T12, implement a deterministic video inference API that preserves timestamps, frame indices, source identity, score, and original-image coordinates. Integrate decode, the numerical motion channel, YOLO, optional SAM refinement/head, NMS, conversion to the documented DetectionCandidate schema, the existing ConstantVelocityTracker, and structured audit JSONL output. Define first/last-frame behavior and expose the symmetric motion channel's one-frame lookahead. Keep optional SAM2 video memory and Action Bank behind explicit configuration, and retain a valid detector-only fallback when D or SAM fails. Produce an annotated review video only as a diagnostic rendering. Measure each pipeline stage and total latency.

For T13, define tentative, confirmed, lost, and reacquiring states around frozen frame detections. Predefine association, innovation/distance and scale gates, confidence update, maximum gap, memory expiry, scene-cut and camera-motion reset, and conditions for emitting a recovered box. Require detector-supported confirmation after loss. Prevent stale or low-confidence SAM2 memory from recursively writing predictions back into memory. Reset at every video boundary and log every transition, memory write, suppression, reset, and fallback. Tune or train only with training and source-validation videos. Preserve raw frame detections so temporal effects can be audited separately.

For T14, implement complete-video temporal evaluation; do not fabricate a command before its parser and tests exist. Compare frozen detector-only, ConstantVelocityTracker, optional SAM2 memory, and Action Bank with one component changed at a time on identical videos and thresholds. Report frame detection metrics, continuity, recovered gaps or time to reacquire, false persistence/false-track duration, reviewed-empty-video false alarms, and p50/p95 end-to-end latency. Report MOTA, IDF1, HOTA, or identity switches only when the dataset has consistent reviewed identities; otherwise use clearly labeled continuity/fragmentation proxies. Use videos or capture groups as statistical units. Do not tune memory rules on test sequences or omit sequences where a temporal arm fails.
```

## Status-report prompt

```text
Report only verified state.

task_id / run_id:
state: RUNNING | QUEUED | STOPPED | FAILED | COMPLETE
process or scheduler id; start time; latest log timestamp:
epoch; optimizer step / planned optimizer steps; sample and domain coverage:
latest checkpoint path; step; timestamp; SHA-256; load-check result:
GPU utilization and memory; CPU/I/O observations; free workspace capacity:
acceptance checks completed:
current error, affected outputs, and next action:
run class: smoke | matched training | SAM adaptation | evaluation | temporal evaluation

Use RUNNING only after checking that the process exists, progress is advancing, and GPU activity is consistent with the job. If any check fails, report STOPPED or FAILED and give the last confirmed progress. A command's successful exit is not by itself evidence of scientific completion.
```
