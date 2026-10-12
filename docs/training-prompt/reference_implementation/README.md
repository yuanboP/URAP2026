# Training component reference source

These components provide the model and data operations needed to implement the fresh experiment. They are a selected source subset, not a complete trainer, evaluator, or video-tracking application. Use the tasks in [01_TASKS.md](../01_TASKS.md) to complete the missing interfaces.

## Components

| File | Purpose |
|---|---|
| [fusion_model.py](fusion_model.py) | RGB-only A, input-fused RGBD B, optional P2-feature fusion C, and shared-weight transfer |
| [yolo11m_p2.yaml](yolo11m_p2.yaml) | YOLO11m topology with P2/P3/P4/P5 detection scales |
| [yolomg_parallax_robust_difference.py](yolomg_parallax_robust_difference.py) | Global compensation, local parallax correction, and numerical residual operations |
| [data_stream.py](data_stream.py) | Ordered triplet decoding and bounded numerical motion streaming |
| [motion_cache.py](motion_cache.py) | Content-aware motion cache and residual codec |
| [augment_full.py](augment_full.py) | Synchronized transforms for RGB, motion, validity masks, and boxes |
| [sampling.py](sampling.py) | Deterministic full-coverage sample plans with explicit replay |
| [precision_policy.py](precision_policy.py) | Local precision policy for the detector input stem |
| [training_support.py](training_support.py) | Small schedule, serialization, and run-identity helpers |
| [metrics.py](metrics.py) | One-to-one matching and Ultralytics AP helper; not a full formal evaluator |
| [sam_adapt.py](sam_adapt.py) | Training-box teacher prompts and decoder-only SAM2.1 student adaptation |

[source_manifest.csv](source_manifest.csv) records each source and published SHA-256 identity. Only line endings and final newlines are normalized; the component logic is unchanged. These components have not been turned into a new unified training launcher by this documentation change.

## Integration contracts to implement

- Keep this directory together when importing its flat modules. `data_stream.py` imports `motion_cache`; place the directory on the application's Python import path or package it explicitly as part of T04.
- The stream expects `neighbor_frame_ids` in canonical zero-based decoder coordinates. `source_frame_index_base` records annotation provenance and is not subtracted again by the worker.
- The residual is encoded as `motion2 = round(2 * D)` in a lossless integer cache, then normalized as `motion2 / 510`. This represents the same D/255 model input. Do not feed an encoded cache array without decoding and normalization.
- Pass the exact residual script path and frozen parameters in the stream's signature. Its functions compute the training input; its rendered-video CLI is only a visualization tool.
- Map each fresh canonical manifest to the actual component fields, including dimensions, source IDs, `role`, and annotation-completeness flags. Generated negative crops require verified exhaustive annotations; disable or ignore them for partially annotated frames. A positive box alone does not certify that another region is empty.
- T03/T04 must handle compensation/decoding failures explicitly. The presence of a helper does not establish that the complete RGB fallback and coverage accounting have been integrated.
- `metrics.py` expects prediction and target coordinates in the same space; its documented reference input is network-input pixels. T09 must evaluate the formal experiment in original-image coordinates, with consistent inverse transforms.
- The metric helper calls Ultralytics `ap_per_class`, includes fixed confidence-0.25 precision/recall, and reports a candidate-recall field whose name assumes a caller-side confidence floor and max detections. It does **not** itself enforce that candidate floor or cap. T09 must implement explicit operating-point calibration, detection limits, and standard COCO evaluation when claimed.
- `sam_adapt.py` trains only `sam_mask_decoder.*`; it provides neither a drone semantic detector nor a video-memory training loop. Its declared official SAM2 commit is `2b90b9f5ceec907a1c18123530e92e794ad901a4`. Use that dependency revision or explicitly update and revalidate both the implementation and provenance.
- The SAM command's `best.pt` reflects complete-epoch pseudo-mask training loss. Add independent validation-based selection before calling an artifact the best detection model. Verify resume identity at the orchestration layer before using a checkpoint from the same fresh run.

## Verified interface checks

From the repository root in an environment with NumPy installed:

```bash
python docs/training-prompt/reference_implementation/sam_adapt.py pseudo --help
python docs/training-prompt/reference_implementation/sam_adapt.py train --help
```

These help commands exercise argument parsing only. They do not allocate a training GPU, validate SAM/CUDA installation, load model weights, or establish model performance. Follow T00/T05 for actual environment and training smoke checks.
