# Training quickstart

This package runs a matched experiment. Arm A receives RGB and ignores motion. Arm B receives the same RGB presentation plus a zero-initialized fourth motion channel. Both start from one YOLO11m-to-P2 transfer and receive the same shuffled samples, augmentation seeds, and optimizer-step count.

These commands target a Linux NVIDIA GPU workstation. Python 3.11 or 3.12 is recommended for dependency compatibility. The validated project stack used PyTorch `2.11.0+cu128`, torchvision `0.26.0`, Ultralytics `8.4.51`, and pycocotools `2.0.11`; that does not validate every Python and driver combination.

## Install

```bash
git clone --branch training-prompt https://github.com/yuanboP/URAP2026.git
cd URAP2026
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-training.txt

# Install official SAM 2 after torch. This avoids build isolation fetching a
# second large torch wheel and disables the optional CUDA extension.
SAM2_BUILD_CUDA=0 .venv/bin/python -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'

.venv/bin/python -m drone_training doctor
```

`doctor` reports the environment it actually finds. `ready: true` only means required imports succeeded. It does not claim data, weights, or a completed run exist.

```bash
export DATA="$HOME/urap2026/data"
export WEIGHTS="$HOME/urap2026/weights"
export RUNS="$HOME/urap2026/runs"
mkdir -p "$DATA" "$WEIGHTS" "$RUNS"
```

The data location needs room for the approximately 100 GB release. Keep the motion cache at 10 GiB unless the workstation has extra capacity.

## Data and weights

The default is the formal, pinned full release. Downloads are hash-verified and produce receipts.

```bash
.venv/bin/python -m drone_training download --data-root "$DATA"
.venv/bin/python -m drone_training weights --out "$WEIGHTS"
```

The common-frame training manifest contains 115,675 records. Inspect `download_receipt.json`, `download_progress.jsonl`, and `weights_receipt.json` before a long run.

Smoke data is a separate explicit download:

```bash
export SMOKE_DATA="$HOME/urap2026/smoke-data"
.venv/bin/python -m drone_training download --data-root "$SMOKE_DATA" \
  --smoke --train-limit 8 --val-limit 4
```

## Matched YOLO training

On a new machine, run the four-step smoke/recovery sequence below **before** the formal command. Use different output directories for smoke and formal runs. A and B are trained together by one command.

The formal plan uses all common frames for ten epochs at 1280 pixels. `batch=4` and `accumulation=2` are the planned settings. If memory is insufficient, reduce batch and increase accumulation by the same factor. Omit `--steps` so the trainer derives the full optimizer-step count.

When `--val-every` is omitted, validation runs once per derived training pass and again at the final step. Use `--val-every 0` only when intentionally disabling validation, such as a local diagnostic that must not select best checkpoints.

```bash
.venv/bin/python -m drone_training train-yolo \
  --data-root "$DATA" --weights "$WEIGHTS/yolo11m.pt" \
  --out "$RUNS/yolo_ab_full" --epochs 10 --size 1280 \
  --batch 4 --accumulation 2 --workers 4 --device cuda \
  --cache-dir "$DATA/motion-cache" --cache-gib 10
```

Resume with the same immutable schedule:

```bash
.venv/bin/python -m drone_training train-yolo \
  --data-root "$DATA" --weights "$WEIGHTS/yolo11m.pt" \
  --out "$RUNS/yolo_ab_full" --epochs 10 --size 1280 \
  --batch 4 --accumulation 2 --workers 4 --device cuda \
  --cache-dir "$DATA/motion-cache" --cache-gib 10 --resume
```

This smoke sequence tests interruption and resume. Both invocations retain the same four-step schedule; `--stop-after 2` only interrupts the first.

```bash
.venv/bin/python -m drone_training train-yolo \
  --data-root "$SMOKE_DATA" --weights "$WEIGHTS/yolo11m.pt" \
  --train-manifest smoke/manifests/flow_train_common.jsonl \
  --val-manifest smoke/manifests/flow_validation.jsonl \
  --out "$RUNS/yolo_ab_smoke" --size 256 --batch 2 --accumulation 1 \
  --workers 0 --steps 4 --val-every 2 --stop-after 2 --device cuda

.venv/bin/python -m drone_training train-yolo \
  --data-root "$SMOKE_DATA" --weights "$WEIGHTS/yolo11m.pt" \
  --train-manifest smoke/manifests/flow_train_common.jsonl \
  --val-manifest smoke/manifests/flow_validation.jsonl \
  --out "$RUNS/yolo_ab_smoke" --size 256 --batch 2 --accumulation 1 \
  --workers 0 --steps 4 --val-every 2 --device cuda --resume
```

The run directory contains `last.pt`, independently selected `best_A.pt` and `best_B.pt`, configuration and model reports, logs, validation artifacts, and `summary.json`. A stopped run remains partial until the summary says `complete: true`.

A fresh invocation refuses an output directory containing managed run evidence. Use `--resume` for the same immutable experiment or choose a new `--out`; the trainer never silently overwrites a prior completed or partial run.

## Validation, test, and paired comparison

Calibrate each arm independently from its selected checkpoint. A base weight is unnecessary because evaluation strictly loads the complete saved arm state.

```bash
mkdir -p "$RUNS/eval"
.venv/bin/python -m drone_training evaluate \
  --data-root "$DATA" --manifest release_v3_curated/manifests/flow_validation.jsonl \
  --split validation --checkpoint "$RUNS/yolo_ab_full/best_A.pt" --arm A \
  --out "$RUNS/eval/A_validation" --size 1280 --device cuda

.venv/bin/python -m drone_training calibrate \
  --predictions "$RUNS/eval/A_validation/predictions.jsonl" \
  --out "$RUNS/eval/A_threshold.json"
```

Repeat for B. Then use the exact canonical test manifest and each arm's frozen validation threshold:

```bash
.venv/bin/python -m drone_training evaluate \
  --data-root "$DATA" --manifest release_v3_curated/manifests/flow_test.jsonl \
  --split test --checkpoint "$RUNS/yolo_ab_full/best_A.pt" --arm A \
  --threshold-file "$RUNS/eval/A_threshold.json" \
  --out "$RUNS/eval/A_test" --size 1280 --device cuda

.venv/bin/python -m drone_training compare \
  --a-predictions "$RUNS/eval/A_test/predictions.jsonl" \
  --b-predictions "$RUNS/eval/B_test/predictions.jsonl" \
  --a-calibration "$RUNS/eval/A_threshold.json" \
  --b-calibration "$RUNS/eval/B_threshold.json" \
  --out "$RUNS/eval/paired_comparison.json" --resamples 2000 --seed 17
```

The comparison gives official AP point estimates and a paired video bootstrap for operating metrics. It does not claim AP confidence intervals.

## SAM adaptation

With the default `--manifest release_v3_curated/manifests/sam_positive_train.jsonl` and `--limit 0`, pseudo-mask generation visits all 114,025 positive training-frame records and their boxes, including boundary frames. It writes `pseudo_manifest.jsonl`; the accepted mask count depends on the fixed teacher-quality filters. Run the bounded SAM smoke commands below first on a new machine.

```bash
.venv/bin/python -m drone_training sam-pseudo \
  --data-root "$DATA" --out "$RUNS/sam/pseudo" \
  --teacher-checkpoint "$WEIGHTS/sam2.1_hiera_large.pt" --device cuda

.venv/bin/python -m drone_training sam-train \
  --pseudo-manifest "$RUNS/sam/pseudo/pseudo_manifest.jsonl" \
  --student-checkpoint "$WEIGHTS/sam2.1_hiera_tiny.pt" \
  --out "$RUNS/sam/decoder" --epochs 5 --device cuda
```

SAM training always writes `last.pt` periodically. `best.pt` appears only after a completed epoch improves loss. Resume from the guaranteed `last.pt`:

```bash
.venv/bin/python -m drone_training sam-train \
  --pseudo-manifest "$RUNS/sam/pseudo/pseudo_manifest.jsonl" \
  --student-checkpoint "$WEIGHTS/sam2.1_hiera_tiny.pt" \
  --out "$RUNS/sam/decoder" --epochs 5 --device cuda \
  --resume "$RUNS/sam/decoder/last.pt"
```

Actual bounded SAM smoke commands:

```bash
.venv/bin/python -m drone_training sam-pseudo \
  --data-root "$SMOKE_DATA" \
  --manifest "$SMOKE_DATA/smoke/manifests/flow_train_common.jsonl" \
  --out "$RUNS/sam-smoke/pseudo" --limit 8 \
  --teacher-checkpoint "$WEIGHTS/sam2.1_hiera_large.pt" --device cuda

.venv/bin/python -m drone_training sam-train \
  --pseudo-manifest "$RUNS/sam-smoke/pseudo/pseudo_manifest.jsonl" \
  --student-checkpoint "$WEIGHTS/sam2.1_hiera_tiny.pt" \
  --out "$RUNS/sam-smoke/decoder" --epochs 1 --stop-after 1 --device cuda
```

Resume the stopped smoke checkpoint with the same one-epoch schedule and without `--stop-after`:

```bash
.venv/bin/python -m drone_training sam-train \
  --pseudo-manifest "$RUNS/sam-smoke/pseudo/pseudo_manifest.jsonl" \
  --student-checkpoint "$WEIGHTS/sam2.1_hiera_tiny.pt" \
  --out "$RUNS/sam-smoke/decoder" --epochs 1 --device cuda \
  --resume "$RUNS/sam-smoke/decoder/last.pt"
```

`--stop-after` is a fresh-run smoke bound and cannot be combined with resume. If no pseudo masks pass, inspect the rejection records; do not silently relax the formal quality filters.

SAM inference is prompt-conditioned and requires a JSON array of original-coordinate `xyxy` proposals. It writes JSON metadata and a compressed `.masks.npz`.

```bash
.venv/bin/python -m drone_training sam-infer \
  --image frame.png --boxes proposals.json \
  --checkpoint "$RUNS/sam/decoder/last.pt" \
  --student-checkpoint "$WEIGHTS/sam2.1_hiera_tiny.pt" \
  --out "$RUNS/sam/inference.json" --device cuda
```

## Optional proposal head

The formal head defaults to five epochs and batch size 32. Candidates must contain matched positives and certified negatives.

```bash
.venv/bin/python -m drone_training sam-head-train \
  --candidate-jsonl candidates.jsonl --data-root "$DATA" \
  --student-checkpoint "$WEIGHTS/sam2.1_hiera_tiny.pt" \
  --out "$RUNS/sam/proposal-head" --epochs 5 --batch 32 --device cuda

.venv/bin/python -m drone_training sam-head-infer \
  --image frame.png --proposals proposals.json \
  --checkpoint "$RUNS/sam/proposal-head/last.pt" \
  --student-checkpoint "$WEIGHTS/sam2.1_hiera_tiny.pt" \
  --out "$RUNS/sam/scored-proposals.json" --device cuda
```

Head smoke interruption/resume retains the same four-step schedule:

```bash
.venv/bin/python -m drone_training sam-head-train \
  --candidate-jsonl candidates.jsonl --data-root "$DATA" \
  --student-checkpoint "$WEIGHTS/sam2.1_hiera_tiny.pt" \
  --out "$RUNS/sam/head-smoke" --steps 4 --batch 2 --stop-after 2 --device cuda

.venv/bin/python -m drone_training sam-head-train \
  --candidate-jsonl candidates.jsonl --data-root "$DATA" \
  --student-checkpoint "$WEIGHTS/sam2.1_hiera_tiny.pt" \
  --out "$RUNS/sam/head-smoke" --steps 4 --batch 2 --device cuda --resume
```

Use `python -m drone_training COMMAND --help` for every option. These commands use local storage and compute and do not start paid cloud jobs. Command exit alone is not proof of completion; inspect receipts, checkpoint cursors, progress files, and summaries.
