# Delivery index

## Version-controlled implementation

- `drone_training/`: CLI, validated RGBD data adapter, bounded cache, model construction, matched trainer, evaluation/calibration/comparison, SAM decoder workflow, and optional proposal head.
- `tests/test_drone_training_*.py`: unit and integration coverage for the portable package.
- `requirements-training.txt`: training dependencies.
- `docs/training-prompt/00_QUICKSTART.md`: canonical commands.
- `docs/training-prompt/01_TASKS.md`: status and acceptance plan.
- `docs/training-prompt/02_ARCHITECTURE_AND_EXPERIMENTS.md`: frozen research contract.
- `docs/training-prompt/03_DATA.md`: pinned release and annotation contract.
- `docs/training-prompt/04_WORKFLOW_AND_COMMANDS.md`: execution/evidence workflow.
- `docs/training-prompt/05_PROMPTS.md`: copyable formal-run prompt.
- `docs/training-prompt/06_RESULT_TEMPLATE.md`: final report structure.
- `docs/training-prompt/08_TRACKING_INTEGRATION.md`: unimplemented T12–T14 follow-on.
- `docs/training-prompt/09_ENGINEERING_VERIFICATION.md`: actual bounded training/recovery evidence and limits.
- `docs/training-prompt/reference_implementation/`: immutable published components used by adapters.

## Data acquisition artifacts

Expected under the chosen external data root:

- `release_v3_curated/metadata.zip`
- `release_v3_curated/` verified extracted metadata and canonical manifests
- `videos/<sha256>.mp4` for selected or full retained media
- `download_receipt.json` for full data, or `download_receipt_smoke.json` for smoke
- `download_progress.jsonl`
- optional `smoke/manifests/flow_train_common.jsonl` and `flow_validation.jsonl`

The full release contains 126 retained videos and approximately 96.4 GB of video bytes. A delivery must also state configured caps and actual usage for motion cache, pseudo masks, checkpoints, and other derived outputs.

## Weight artifacts

- `yolo11m.pt` with receipt
- official `sam2.1_hiera_large.pt` teacher with receipt
- official `sam2.1_hiera_tiny.pt` student base with receipt
- official SAM 2 source commit identity

## YOLO run artifacts

- frozen run configuration and model report
- `last.pt`, `best_A.pt`, and `best_B.pt`
- `training_log.jsonl` and summary
- validation artifacts for each arm
- interruption/resume evidence
- exact train/validation manifest identities and data/code/weight hashes

## Evaluation artifacts

- original-coordinate validation predictions for A and B
- independent validation calibration JSON for A and B
- frozen test predictions for A and B
- original-coordinate prediction JSONL, bound provenance sidecars, and official COCO AP reports
- paired comparison with source-video bootstrap operating metrics
- domain, sequence, size, certified-negative, and motion-fallback slices
- decode/missing/duplicate/fallback failure accounting
- runtime and peak memory measurements

## SAM artifacts

- pseudo-mask manifest, accepted/rejected counts, and generation identity
- Tiny decoder `last.pt`/best checkpoint, logs, and immutable base identity
- prompt-conditioned inference outputs on automatic proposals
- clearly labeled oracle diagnostics, if any
- for the optional head: candidate lineage, partition proof, supervision counts, checkpoints, calibration, and identical-proposal comparison

## Final evidence package

- completed [06_RESULT_TEMPLATE.md](06_RESULT_TEMPLATE.md)
- environment and dependency record
- command transcript and repository commit
- data and weight receipts
- split and annotation audit
- all frozen configs and hashes
- positive, negative, and inconclusive findings
- limitations: mixed-source held-out-video scope, incomplete-mask truth, prompt-conditioned SAM semantics, and any provisional annotation slices

Do not include raw credentials, private paths, unlicensed source archives, or large binaries in Git. Implementation presence is not a success claim. T12–T14 tracking artifacts are absent until the separate tracking plan is implemented and evaluated.
