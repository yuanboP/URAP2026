# Training project files and external assets

## Git-tracked project specification

| File or directory | Purpose |
|---|---|
| [README.md](README.md) | Scope, reading order, and implementation status |
| [01_TASKS.md](01_TASKS.md) and [task_board.csv](task_board.csv) | Assignable tasks with dependencies and acceptance criteria |
| [02_ARCHITECTURE_AND_EXPERIMENTS.md](02_ARCHITECTURE_AND_EXPERIMENTS.md) | Model definitions, ablations, and fair comparisons |
| [03_DATA.md](03_DATA.md) and dataset CSV inventories | Data sources, split rules, annotation limitations, and file identities |
| [04_WORKFLOW_AND_COMMANDS.md](04_WORKFLOW_AND_COMMANDS.md) | Fresh execution order and actual component commands |
| [05_PROMPTS.md](05_PROMPTS.md) | Copyable instructions for implementing engineers or agents |
| [06_RESULT_TEMPLATE.md](06_RESULT_TEMPLATE.md) | Empty detection, ablation, and tracking report tables |
| [08_TRACKING_INTEGRATION.md](08_TRACKING_INTEGRATION.md) | Follow-on video integration and sequential evaluation |
| [experiment_config.template.yaml](experiment_config.template.yaml) | Experiment configuration specification; requires implementation |
| [reference_implementation/](reference_implementation/README.md) | Selected model, residual, sampling, cache, and SAM adaptation components |
| [reference_implementation/source_manifest.csv](reference_implementation/source_manifest.csv) | Component byte sizes and SHA-256 hashes |

These files define a new experiment. The component source provides implementation building blocks, not a turnkey end-to-end training or tracking executable. T04, T08, T09, and T12-T14 identify the required integration and evaluation work.

## External assets required before training

1. **Original NewDJI28 videos:** use [new_video_inventory.csv](new_video_inventory.csv) for exact filenames, Drive file IDs, byte counts, split membership, and available hashes.
2. **Original drone boxes:** obtain the annotation archive from the project owner. Its portable alias is `annotations_2607.zip`, its size is 424,042 bytes, and its SHA-256 is `dcfcb9c1a79ceecf39b127e57abe03ba13ec73c602571147687267b450f69e4a`. Renaming the archive does not alter its content hash. Preserve the annotation JSON bytes when reading it.
3. **ARD100 and DJI source assets:** [legacy_video_inventory.csv](legacy_video_inventory.csv) identifies 53 training and 14 validation source videos. Obtain original labels, class mappings, and source split definitions along with the videos.
4. **External evaluation domains:** obtain Anti-UAV/NPS original videos, complete annotations, and official split definitions. Any optional additional domain needs the same provenance and label audit.
5. **Generic base weights:** acquire the YOLO11m and official SAM2.1 Large/Tiny bases listed in [weights_inventory.csv](weights_inventory.csv). Record download URL, bytes, and SHA-256. Both detector arms start from the same generic source checkpoint and matched initialization seed.

The repository does not contain videos, annotation payloads, generated frame manifests, pseudo-masks, crops, or model weight binaries. Shared Drive access is separate from GitHub access. Use the inventory to acquire the data, then build and freeze manifests for this experiment under T01-T02; private source manifests can be used as auditable inputs, not as unexamined evidence of completeness.

## Portable data layout

```text
external_assets/
  annotations/annotations_2607.zip
  newdji28/raw/<original_video_name>.MP4
  legacy/<domain>/<content-addressed-video-path>
  public/<dataset>/<original-dataset-layout>
workspace/
  manifests/             # new frozen train, validation, and test definitions
  models/                # official generic initialization weights
  cache/                 # regenerable numerical residuals and decoded samples
  runs/<run_id>/         # new checkpoints, predictions, and logs
  reports/               # measured results and audits
```

Inventory paths are logical asset names. Resolve them once against a data root and keep the resulting manifest, content hashes, and path mapping in the run record. Do not silently replace a missing video with another clip or with a motion-visualization video.

## Receipt and reproducibility checks

- Verify byte counts and available SHA-256 hashes before decoding; compute and record missing hashes after full receipt.
- Check native video dimensions, frame count, frame-index base, and annotation coordinate conventions.
- Keep each video, neighboring frames, residuals, crops, and pseudo-masks within its source split.
- Review evaluation annotations, including target absence. Unreviewed empty records are not certified negatives.
- Store raw assets and new-run checkpoints durably; keep regenerable caches bounded.
- Record the exact repository commit and dependency versions used to build the trainer and evaluator.

The listed hashes establish file identity, not annotation correctness or model quality. Fresh model initialization does not remove prior human exposure to a dataset; strict external claims still require an appropriate protocol.

## Final model handoff

Deliver trained weights with their base dependencies, model and tracker configurations, frozen manifests, environment lock, code revision, raw predictions, calibration files, per-video metrics, and a ground-truth-free inference command. Supply a checkpoint interruption/recovery readback for the new run. Complete [06_RESULT_TEMPLATE.md](06_RESULT_TEMPLATE.md), including failures and limitations, before declaring the experiment complete.
