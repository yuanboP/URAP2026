# URAP2026

URAP2026 is the lightweight research workspace for UAV small-object detection experiments.

The **fresh training project** has executable [full-data training commands](docs/training-prompt/00_QUICKSTART.md) and a [copyable colleague prompt](docs/training-prompt/05_PROMPTS.md). Run `python -m drone_training --help` for pinned Hugging Face data import, matched RGB/RGBD YOLO11m-P2 training, SAM2 Tiny decoder adaptation, recovery and evaluation. [Training tasks and scope](docs/training-prompt/README.md) distinguish these model-training paths from the separate video-tracking work.

The current active prototype is **QSTR-DroneDet**: a quality-aware, two-stage tiny drone detection and recognition scaffold. It separates candidate generation from recognition so experiments can diagnose whether a miss came from proposal failure, recognition failure, motion alignment failure, feature loss, or weak single-frame evidence.

This repository intentionally tracks:

- QSTR-DroneDet source code in `qstr_dronedet/`
- portable model training and evaluation in `drone_training/`
- synthetic/unit tests in `tests/`
- empty data templates in `data_templates/`
- experiment and reproduction notes in `doc/`
- project reports in `reports/`
- advanced optical-flow package in `optical_flow_advanced/`
- Windows/PowerShell and Python runner scripts in `tools/`
- repository operating rules in `AGENTS.md`

It intentionally does not track:

- raw datasets, extracted videos, frames, or annotations
- model weights/checkpoints
- generated artifacts and training outputs
- third-party cloned paper repositories under `papers/` and related local folders
- Python virtual environments

## Repository Layout

```text
qstr_dronedet/      QSTR-DroneDet Python package
tests/              Synthetic tests and CLI smoke tests
data_templates/     CSV templates for real video recording and annotation
doc/                Reproduction notes and experimental logs
reports/            Professor-facing summaries and paper/dataset notes
optical_flow_advanced/  Unified motion-compensation methods and documentation
tools/              PowerShell/Python runners for repeatable experiments
```

## Key Local Dependencies

The full local workspace uses external code/data checked out or downloaded beside this repo:

- ESOD official code: `papers/ESOD`
- TransVisDrone code: `papers/TransVisDrone`
- NPS/Dogfight annotations/code: `datasets/Drone-Detection`
- Li-TETC / Fast-and-Robust UAV-to-UAV baseline: `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking`
- ARD100 data: `D:\URAP_datasets\ARD100`
- AOT data: `D:\URAP_datasets\AOT`

These paths are local workspace conventions, not git-tracked assets.

## Current Focus

The current QSTR experiment line combines:

1. Stage A class-agnostic tiny-object proposal generation with YOLO-P2, motion proposals, tracker proposals, and optional fallback detector proposals.
2. Stage B object recognition using image crops, feature ROI crops, temporal tubes, motion/alignment metadata, and tracker metadata.
3. Motion quality logging through `q_H` / alignment quality so bad homography or fast ego-motion does not blindly dominate fusion.
4. Separate stable and hard-recovery inference profiles:
   - stable/default: lower false positives, old YOLO-P2 primary detector, no verified objectness boost.
   - hard-recovery: optional fallback detector plus stricter gate for tiny / low-objectness recovery cases.
5. Frozen held-out Anti-UAV sequence checks before deciding whether to scale training.

Useful entry points:

- `python -m qstr_dronedet.cli --help`
- `python optical_flow_advanced/run.py list`
- advanced optical-flow guide: `optical_flow_advanced/README.md`
- annotation web/API deployment: `doc/annotation_monorepo_deployment.md`
- `tools/run_qstr_stable_profile.ps1`
- `tools/run_qstr_hard_recovery_profile.ps1`
- `tools/run_qstr_frozen10_profile_benchmark.ps1`
- `tools/run_qstr_tests.ps1`
- `doc/qstr_sanity_static_fast_tests_2026_05_20.md`
- `doc/qstr_real_video_data_protocol.md`
- `doc/official_datasets_and_metrics.md`
- `doc/per_video_window_accuracy_curves.md`
- `doc/paper_window_accuracy_status_2026_05_22.md`
- `data_templates/paper_window_accuracy_runs.example.json`
- `tools/pull_paper_repos.py`
- `tools/inventory_external_window_accuracy_sources.py`
- `tools/download_nps_videos.py`
- `tools/inventory_aicrowd_lfs_weights.py`
- `tools/download_aicrowd_lfs_weights.py`
- `tools/build_yolomg_test_images_dataset.py`
- `tools/build_paper_window_accuracy_smoke.py`
- `tools/prepare_visdrone_yolo.py`
- `tools/audit_paper_window_accuracy_readiness.py`
- `tools/audit_paper_window_accuracy_goal.py`
- `tools/write_paper_window_accuracy_gap_report.py`
- `tools/run_paper_window_accuracy_pipeline.py`
- `tools/discover_paper_window_accuracy_runs.py`
- `tools/build_window_accuracy_dashboard.py`
- `tools/prepare_aicrowd_nps_flight_dirs.py`
- `tools/run_edtc_tracker_window_accuracy.py`
- `tools/start_edtc_tracker_window_accuracy_detached.ps1`
- `tools/monitor_edtc_tracker_window_accuracy.ps1`
- `tools/run_yolo_eval_window_accuracy.py`
- `tools/start_yolo_eval_window_accuracy_detached.ps1`
- `tools/monitor_yolo_eval_window_accuracy.ps1`
- `tools/start_winner_v022_nps_val_detached.ps1`
- `tools/monitor_winner_v022_nps_val.ps1`
- `doc/progress_report_for_professor.md`

Older ESOD / TransVisDrone / YOLOMG reproduction notes remain in `doc/` and `tools/` for baseline comparison.

## Quick Checks

From the repository root:

```powershell
python -m qstr_dronedet.cli --help
pytest tests -q
```

Run one QSTR profile on a video:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_stable_profile.ps1 `
  -Video D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\raw_videos\test\visible\20190925_111757_1_5\visible.mp4 `
  -Out runs\profiles\stable_example `
  -Device 0
```

Run the frozen held-out benchmark wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_frozen10_profile_benchmark.ps1 `
  -OutRoot runs\profiles\frozen10_profile_eval `
  -Device 0
```

These commands expect local datasets and model weights to exist on the machine. The repo stores scripts and code only, not the datasets or checkpoints.

## Tracklet-Level Recovery MVP

The hard-tiny recovery direction now includes a supervised tracklet classifier. It aggregates `infer` diagnostics by `track_id`, converts each tracklet into objectness, Stage B, fallback, drift, validation, and box-size features, then predicts `tracklet_is_drone`.

```powershell
python -m qstr_dronedet.cli build-tracklet-dataset --diagnostics runs\...\diagnostics.jsonl --gt-csv D:\datasets\...\qstr_real_boxes.csv --out runs\...\tracklets
python -m qstr_dronedet.cli train-tracklet-classifier --csv runs\...\tracklets\tracklets.csv --out runs\...\tracklet_mlp.pt
python -m qstr_dronedet.cli eval-tracklet-classifier --csv runs\...\tracklets\tracklets.csv --weights runs\...\tracklet_mlp.pt --out runs\...\tracklet_eval
```

This is an MVP, not a frozen benchmark claim. Fit it on train/adaptation sequences, then evaluate once on frozen held-out sequences.

## Detached Long Runs

Long-running jobs must be launched through the `tools/start_*_detached.ps1` scripts and monitored by matching `tools/monitor_*` scripts. Status reports should be based on PID, output timestamps, progress counters, and GPU signal when relevant.
