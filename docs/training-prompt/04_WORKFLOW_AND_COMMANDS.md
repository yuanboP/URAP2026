# Execution workflow

[00_QUICKSTART.md](00_QUICKSTART.md) is the canonical copyable command guide. This file defines ordering, evidence, and failure handling.

## 1. Preflight

Run `python -m drone_training doctor`, record the repository commit and environment, inspect free disk, and download verified official weights. A successful import report is not proof that data or training are ready.

Run the explicit 8/4 smoke workflow before a long run. Confirm exact decoding, nonzero D on moving eligible frames, certified-negative handling, A/B forward/backward, finite losses, artifact writes, interruption, and resume. Smoke outputs are not scientific results.

## 2. Data acquisition

The default download obtains the pinned metadata and all 126 retained videos. It can resume Hugging Face transfers and verifies archive, indexed metadata members, and media SHA/bytes. Follow progress through `download_progress.jsonl`; retain the receipt.

The full payload is approximately 96.4 GB. Add explicit bounded budgets for derived motion cache, pseudo masks, checkpoints, and evaluation outputs. Do not interpret 96.4 GB as the total disk requirement.

Audit that train, validation, and test SHA sets are disjoint. The dataset is mixed ARD/DJI with held-out videos, not strict cross-domain. Anti-UAV/NPS require separate acquisition and protocols because they are absent from this release.

## 3. Formal YOLO A/B training

Use the full default `train-yolo` command from the quickstart: 115,675 common training records and 10 epochs. Do not set a small `--steps` budget for a formal run. Leave validation outside optimizer updates.

The trainer writes atomic `last.pt`, independent `best_A.pt`/`best_B.pt`, frozen configuration/model reports, JSONL progress, validation artifacts, and a summary. Resume only with the identical data identities and immutable configuration. If a process stops, report it as stopped; verify checkpoint integrity before restarting.

Arm A and B must consume the identical RGB presentation and target. Arm B receives the corresponding D channel. No test data, validation labels, or different augmentation may enter one arm's training path.

## 4. Validation, calibration, and test

Evaluate both arms on full validation, select checkpoints by the preregistered metric, and calibrate thresholds independently. Freeze checkpoint hashes, calibration JSON, evaluator settings, and manifests before opening test.

Evaluate test with the frozen thresholds. Predictions and ground truth remain in original coordinates. Keep low-score predictions needed for AP. Record decode failures, duplicates, missing frames, and incomplete-annotation caveats. COCO AP is a point estimate; the paired source-video bootstrap applies to operating metrics unless a separate valid AP resampling method is implemented.

Use `compare` only after A/B predictions cover identical sample identities. Report overall, per-domain, per-sequence, size-bin, positive/certified-negative, and motion-fallback slices. The mixed corpus supports held-out-video claims only.

## 5. SAM paths

Generate pseudo masks from all 114,025 positive training rows with the Large teacher, inspect rejection counts, and train the Tiny mask decoder for the formal default of 5 epochs. SAM inference is box-prompted; mask quality score is not drone confidence. Evaluate automatic predicted proposals and label ground-truth-box prompting as oracle-only.

The optional proposal head is separate from mask-decoder adaptation. Its candidate JSONL must come from out-of-fold source-video predictions or a separately held-out training partition. Only certified negatives or exhaustive training frames may label proposal background. Incomplete-frame unmatched proposals are ignored.

## 6. Recovery and audit

- Preserve `last.pt`, run config, logs, and data/weight receipts before resuming.
- Reject a resume when model, manifest identity, seed, batch/update plan, or other immutable settings differ.
- Keep validation calibration and test execution in distinct directories.
- Do not silently skip decode, motion, label, or evaluator failures.
- Store large derived artifacts outside Git and retain content hashes and generation commands.
- Keep `reference_implementation/` unchanged; adapters import it as the published baseline.

## 7. Completion boundary

T00–T11 complete only when their acceptance evidence is present. Available CLI commands do not imply completed formal runs. T12–T14 remain separate unimplemented tracking work and must follow [08_TRACKING_INTEGRATION.md](08_TRACKING_INTEGRATION.md).
