# Workflow and verified commands

This document defines a **brand-new training program**. Every learned component starts from a generic public pretrained model: standard Ultralytics `yolo11m.pt` for YOLO and the official SAM2.1 Hiera Large and Hiera Tiny checkpoints for SAM. Interruption recovery uses a checkpoint from the same new run contract.

The files under `docs/training-prompt/reference_implementation/` are a small, flat set of component sources with unchanged logic and normalized line endings. They contain useful model, motion, data-stream, SAM adaptation, training-support, and metric logic, but they are not a finished turnkey training system. In particular, the unified fair A/B trainer, the SAM candidate classification/regression head, the final fair evaluator, and the temporal evaluation runner must be implemented and tested in T04, T08, T09, and T14. Do not invent command-line flags for those pending tools.

## 1. Repository layout and run artifacts

Run all commands from the repository root. Keep datasets and generated artifacts outside Git; store only public schemas, small non-sensitive audit summaries, and code in the repository. Generated dataset manifests remain external. Commands containing `<work-root>` are templates: replace that placeholder with the absolute experiment directory before running them.

```text
<repo>/
  docs/training-prompt/
  docs/training-prompt/reference_implementation/
    fusion_model.py
    yolo11m_p2.yaml
    sam_adapt.py
    data_stream.py
    motion_cache.py
    augment_full.py
    sampling.py
    precision_policy.py
    training_support.py
    metrics.py
    yolomg_parallax_robust_difference.py
  data_templates/

<work-root>/
  raw/                       # original videos and annotations
  manifests/                 # immutable source manifests and derived split manifests
  cache/                     # reproducible, bounded RGB/D/SAM caches
  weights/                   # downloaded public pretrained checkpoints
  runs/<run_id>/             # configs, logs, checkpoints, predictions, and metrics
  audits/                    # hashes, split checks, environment capture, and readback reports
```

Never commit raw videos, extracted frames, label archives, generated masks, cache tensors, or model checkpoints. A receiving teammate must obtain the external dataset separately and verify it against the supplied inventory before training.

Each run directory must contain a machine-readable contract with the code commit, dirty diff status, input manifest SHA-256, split identity, starting-weight SHA-256, model variant, seed, image size, augmentation configuration, optimizer and scheduler, effective batch size, maximum optimizer steps, validation schedule, checkpoint rule, and metric protocol. Treat this contract as immutable after the first optimizer step.

## 2. Environment and public starting weights

Choose a CUDA-compatible PyTorch build for the available GPU, then lock the versions actually installed. Record at least Python, PyTorch, CUDA runtime, torchvision, Ultralytics, OpenCV, NumPy, the SAM2 commit, GPU model, and GPU memory.

Install SAM2 from the official repository at the revision declared by the supplied adaptation component. If that revision changes, update and validate the implementation and its recorded provenance together.

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout --detach 2b90b9f5ceec907a1c18123530e92e794ad901a4
python -m pip install -e third_party/sam2
```

Download and hash the three public starting checkpoints:

- Ultralytics `yolo11m.pt`, obtained through the official Ultralytics model-loading flow.
- SAM2.1 Hiera Large: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`.
- SAM2.1 Hiera Tiny: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`.

Use `configs/sam2.1/sam2.1_hiera_l.yaml` with Large and `configs/sam2.1/sam2.1_hiera_t.yaml` with Tiny. Record file hashes before use.

```bash
python - <<'PY'
import hashlib
from pathlib import Path

for path in map(Path, [
    "<work-root>/weights/yolo11m.pt",
    "<work-root>/weights/sam2.1_hiera_large.pt",
    "<work-root>/weights/sam2.1_hiera_tiny.pt",
]):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    print(h.hexdigest(), path)
PY
```

The YOLO reference model uses `docs/training-prompt/reference_implementation/yolo11m_p2.yaml` together with `fusion_model.py`. The implementation task must transfer compatible shared parameters from standard `yolo11m.pt` into both A and B. A receives RGB. B receives RGB plus one numerical motion-residual channel; initialize only B's added channel weights to zero. Save a transfer report listing loaded, reshaped, initialized, and rejected keys.

## 3. Data intake and split freeze

T00 and T01 begin with external dataset intake. Preserve original bytes and annotations, calculate SHA-256, inspect native frame count and resolution, and retain the source video ID and capture group. Convert annotations into a canonical manifest only after the source files pass validation.

Split by capture group or source video before creating frame triplets, crops, motion residuals, pseudo-masks, or proposals. Assert that the train, validation, and test sets have no source-video or capture-group overlap. Unknown or unreviewed empty annotations are not negatives. Use a frame as background supervision only when annotation completeness has been reviewed explicitly.

The canonical records should preserve at least `sample_id`, `dataset`, `capture_group`, `source_video_id`, `source_video_path`, `source_sha256`, center-frame index and index base, neighboring-frame indices, native width and height, `bbox_xyxy`, `class_id`, split, annotation completeness, and review state. Derived manifests must retain lineage to the original record.

Before training, produce:

1. an asset audit with bytes, SHA-256, decode result, frame count, and dimensions;
2. a split audit with per-domain videos, frames, boxes, reviewed negative frames, and target-size bins;
3. an overlap report proving zero shared capture groups and videos across splits;
4. visual checks of boxes and triplets across domains and size bins;
5. an annotation audit identifying incomplete, ambiguous, or excluded records.

## 4. Numerical motion channel

T03 validates the reference logic in:

- `docs/training-prompt/reference_implementation/yolomg_parallax_robust_difference.py`
- `docs/training-prompt/reference_implementation/data_stream.py`
- `docs/training-prompt/reference_implementation/motion_cache.py`

For each center frame, align the previous and next frames to the center, compute the numerical symmetric residual, set invalid warp regions to zero, and feed `clip(D / 255, 0, 1)` as B's fourth channel. Do not use a colorized visualization, a rendered video, or per-frame display normalization as model input. Apply every geometric transformation identically to RGB, D, and boxes. When motion computation fails, retain RGB, set D to zero, and log the failure.

Freeze the motion configuration and implementation hash before T04. Save representative RGB/D/box overlays and numerical distribution summaries, including failure rates by domain. Because the symmetric construction uses the next frame, the system has one-frame lookahead; latency and temporal evaluation must include it.

## 5. T04 unified A/B implementation boundary

The published component sources include model construction, sampling, augmentation, data streaming, precision, and checkpoint helpers. They do not include a unified trainer. T04 must deliver a new tested entry point that:

- constructs A and B from the same standard `yolo11m.pt` and the same P2 topology;
- consumes the canonical manifest and a frozen sample-presentation plan;
- gives A and B identical sample IDs, order, geometric and photometric augmentation decisions, image size, effective batch size, optimizer, schedule, maximum optimizer steps, validation points, and checkpoint-selection rule;
- supplies D only to B and initializes B's added input weights deterministically;
- writes coverage, model-transfer, optimizer-step, validation, and checkpoint records;
- resumes atomically within the same new run contract, including model, optimizer, scheduler, scaler, RNG, sampler cursor, and completed-step state;
- rejects a resume if the immutable contract identity differs.

No executable command is published here for that entry point because it does not yet exist. T04 is complete only after its real `--help`, smoke command, tests, and recovery demonstration are committed.

## 6. Smoke tests, matched training, and recovery

T05 exercises every path on a small, representative subset. Run 20–100 optimizer steps for A and B, verify identical sample presentation, inspect RGB/D/box overlays, confirm finite losses, and save a checkpoint. Interrupt the process deliberately, resume it, and compare the next sample IDs, optimizer-step count, RNG state, and scheduler position with an uninterrupted control.

T06 launches the matched main experiment only after T05 passes. The primary comparison uses equal **optimizer-step budgets**, equal effective batch size, and the same validation opportunities. Record sample coverage as well as steps so repeated sampling is visible. A run may extend to another preregistered step limit only if the rule is applied to both arms without consulting test results. Choose each arm's best checkpoint using the same source-validation rule.

Recovery is limited to checkpoints produced by these new runs. Keep `last` and `best` distinct, write checkpoints atomically, and verify their hashes and loadability. A resumed run keeps the original run ID and contract; a changed manifest, split, model, seed, or optimizer creates a new run.

## 7. SAM2.1 pseudo-mask and Tiny decoder adaptation

The CLI in `docs/training-prompt/reference_implementation/sam_adapt.py` has been inspected and provides the verified `pseudo` and `train` subcommands below. It remains reference code: first verify its local imports, run `--help`, and prove a one-sample load test in the target environment.

Teacher smoke run:

```bash
python -u docs/training-prompt/reference_implementation/sam_adapt.py pseudo \
  --manifest <work-root>/manifests/sam_train.jsonl \
  --output <work-root>/runs/sam_teacher_smoke \
  --teacher-checkpoint <work-root>/weights/sam2.1_hiera_large.pt \
  --teacher-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --max-samples 16 \
  --selection-order balanced \
  --max-seconds 600 \
  --device cuda
```

Tiny student smoke run:

```bash
python -u docs/training-prompt/reference_implementation/sam_adapt.py train \
  --pseudo-manifest <work-root>/runs/sam_teacher_smoke/pseudo_manifest.jsonl \
  --output <work-root>/runs/sam_student_smoke \
  --student-checkpoint <work-root>/weights/sam2.1_hiera_tiny.pt \
  --student-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --epochs 1 \
  --learning-rate 0.00001 \
  --checkpoint-every-steps 25 \
  --max-seconds 600 \
  --device cuda
```

The teacher receives training-set human boxes as prompts and produces weak pseudo-mask supervision. Audit acceptance and rejection by domain, video, and target size. Validation and test use fixed automatic detector candidates, never ground-truth prompts. The teacher's predicted mask quality is not a drone-class probability. Compare adapted Tiny with frozen Tiny using the same Tiny base checkpoint, proposals, candidate ordering, and fallback rules.

After smoke acceptance, use separate formal output directories and include every eligible training box with `--max-samples 0`. The following reference run uses the frozen teacher filters and five student epochs; select and record the final epoch/validation policy before the formal run. Five epochs are a starting setting, not evidence of convergence.

```bash
python -u docs/training-prompt/reference_implementation/sam_adapt.py pseudo \
  --manifest <work-root>/manifests/sam_train.jsonl \
  --output <work-root>/runs/sam_teacher_full \
  --teacher-checkpoint <work-root>/weights/sam2.1_hiera_large.pt \
  --teacher-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --max-samples 0 --selection-order balanced \
  --min-score 0.70 --min-crop 256 --box-scale 6 --device cuda

python -u docs/training-prompt/reference_implementation/sam_adapt.py train \
  --pseudo-manifest <work-root>/runs/sam_teacher_full/pseudo_manifest.jsonl \
  --output <work-root>/runs/sam_student_full \
  --student-checkpoint <work-root>/weights/sam2.1_hiera_tiny.pt \
  --student-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --epochs 5 --learning-rate 0.00001 \
  --checkpoint-every-steps 25 --device cuda
```

The student trains on accepted teacher masks, so report both the requested training-box count and the accepted-mask count. A capped or partly processed teacher run is not full-data adaptation. The reference student uses one 1024-by-1024 crop per update, AdamW, and pseudo-mask BCE/Dice plus predicted-IoU supervision; video memory remains frozen.

The reference `train` command accepts `--resume <checkpoint>` for recovery inside the same new student run. Preserve and check its run identity before using it. Its `best` artifact reflects the implemented training-loss rule unless an independent validation rule is added and tested; label it accordingly.

## 8. T08 candidate classification and box-regression head

The current reference code adapts the SAM mask decoder. It does not implement drone/background classification or bounding-box regression. T08 must add a proposal-conditioned head over SAM2 RGB features, define positive/negative/ignore matching, train only from training data, and provide inference that accepts RGB/video plus automatic candidates and returns drone confidence, corrected original-image `xyxy`, and an optional mask.

Use background targets only in reviewed, exhaustively annotated training frames. Generate training candidates using out-of-fold detectors or a separately frozen proposal-training split so the head learns realistic noisy proposals. Start with the SAM image encoder frozen; any adapter or encoder tuning is a separate ablation. Publish no executable T08 command until the implementation, parser, tests, and no-ground-truth inference smoke test exist.

## 9. Evaluation boundary

`docs/training-prompt/reference_implementation/metrics.py` contains reusable metric logic, but the published references do not include a complete fair-evaluation CLI. T09 must build and test that evaluator; do not issue a made-up evaluation command.

T09 must implement or adapt an evaluator for frozen new-run checkpoints. Evaluate A, B, B plus frozen Tiny, B plus adapted Tiny, and—after T08—B plus the candidate head on exactly the same eligible frames. Preserve low-score predictions for AP. Select operating thresholds on source validation, freeze them, and apply them unchanged to test domains. Report original-coordinate AP50, AP50–95, precision, recall, F1, FP/frame, target-size recall, proposal recall at fixed K, per-video/per-domain metrics, paired video-level bootstrap intervals, and stage plus end-to-end latency.

The reference metric helper performs single-class matching and calls Ultralytics `ap_per_class`. It also reports fixed-confidence-0.25 precision/recall; candidate limits implied by its output field names are caller assumptions, not enforced inside the helper. T09 must add explicit threshold calibration, confidence filtering, max-detection handling, and original-coordinate evaluation. If results are called COCO AP, add the official COCO evaluation path and check its behavior on controlled fixtures. With box-only labels, do not report ground-truth mask Dice, mask IoU, or J&F.

Never use test labels to choose thresholds, prompts, epochs, motion settings, head rules, or failure cases. SAM receives the same frozen automatic proposals in every comparison. Count proposal misses, SAM failures, fallback usage, classification rejections, box corrections, and decode failures separately.

## 10. T12–T14 video integration and temporal evaluation

T12 integrates the selected frozen image-level system into a video API and the repository's `ConstantVelocityTracker`. Convert detections into the documented `DetectionCandidate` schema while preserving frame timestamps, source, score, and original coordinates. Expose the one-frame-lookahead requirement, define first/last-frame behavior, batch and cache safely, and produce deterministic per-frame audit JSONL plus an annotated review video. The review rendering is diagnostic output and never model input. Optional SAM2 video memory and Action Bank arms stay behind explicit configuration; detector-only fallback must remain valid when D or SAM fails.

T13 adds bounded loss, reacquisition, and memory controls around frozen detections. Define tentative, confirmed, lost, and reacquiring states; freeze missed-frame, distance/innovation, scale-change, detector-confidence, and memory-confidence gates on sequence-level validation. Reset deterministically at video boundaries, require detector-supported confirmation after loss, and prevent low-confidence memory output from recursively updating memory. Log every memory write, suppression, reset, transition, and fallback. The image-level A/B and SAM conclusions remain based on frozen pre-memory outputs.

T14 evaluates temporal behavior on complete videos. Compare frozen detector-only, `ConstantVelocityTracker`, optional SAM2 memory, and Action Bank with one component changed at a time. Always report frame detection metrics, continuity, gap recovery or time-to-reacquire, false persistence/false-track duration, and end-to-end latency. Report MOTA, IDF1, HOTA, or identity switches only when consistent reviewed identities exist; otherwise label fragmentation measures as proxies. Use video-level confidence intervals, include reviewed empty videos when calculating false alarms, and do not tune memory on test sequences.

## 11. Monitoring and completion evidence

For any long run, save the launch command, process or scheduler identifier, start time, stdout/stderr paths, run contract, and checkpoint paths. A status report may say `RUNNING` only after verifying the process, a moving progress signal, and GPU activity consistent with the job. Report `STOPPED` or `FAILED` explicitly when those checks fail.

Each status update should include:

```text
task_id / run_id:
state: RUNNING | QUEUED | STOPPED | FAILED | COMPLETE
process or scheduler id; start time; latest log timestamp:
epoch; optimizer step / planned optimizer steps; samples and domain coverage:
latest checkpoint path; step; timestamp; SHA-256; load-check result:
GPU utilization and memory; CPU/I/O observations; free workspace capacity:
completed acceptance checks; current error and impact; next action:
run class: smoke | matched training | SAM adaptation | evaluation | temporal evaluation
```

Completion means the artifact exists, loads, has a verified identity, and satisfies its task acceptance checks. A successful command exit alone is not evidence that a model is scientifically validated.
