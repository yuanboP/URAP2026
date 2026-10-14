# Engineering verification — 2026-10-14

This report establishes that the supplied entry points execute real training, save recoverable models and reload them. It does **not** establish full-data accuracy, improved recall, cross-domain generalization or video tracking performance.

Machine-readable evidence: [verification/2026-10-14_smoke.json](verification/2026-10-14_smoke.json). Code SHA256 values in that file identify the implementation exercised below.

## Actual runtime

Windows, Python 3.14, NVIDIA RTX 4070 Ti SUPER (16 GB), PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.51, pycocotools 2.0.11, and official SAM2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`. The documented Linux installation was dependency-checked; an H100/Linux training run was not performed in this verification.

## Checks performed

| Component | Actual check | Result |
|---|---|---|
| Dataset import | Pinned public HF metadata retrieval, archive/member verification, seven locally available videos checked by bytes/SHA | Passed; 62 indexed metadata members verified |
| RGBD samples | Eight training and four validation frames across ARD100, historical DJI and NewDJI; certified negative in each partition | All 12 D channels nonzero; zero motion fallbacks; no source-video overlap |
| Matched YOLO | Real YOLO11m-P2 A/B, 256px, batch 2, accumulation 1, BF16, four optimizer steps, validation every two steps | Finite paired updates; `last.pt`, `best_A.pt`, `best_B.pt` written |
| YOLO recovery | Stop at step 2, resume to step 4, compare to independent uninterrupted four-step run | Same presentation digest; maximum tensor difference 0.0 in both arms |
| YOLO weight update | Compare saved states at steps 2 and 4 | 802 state tensors changed in each arm |
| Reload and evaluation | Reload each selected checkpoint through the CLI; infer four validation frames; calibrate; execute paired comparison | All commands exited successfully |
| SAM teacher | Official Large teacher on seven real positive training box prompts, unchanged quality filters | Six accepted pseudo-masks; one rejection |
| SAM decoder | Official Tiny, corrected predictor-compatible feature graph, train stop at step 1 and resume to step 6 | Finite losses; 121/131 decoder state tensors changed from base |
| SAM reload | CLI reload of final decoder plus exact official Tiny base | Mask output produced; checkpoint source hash matches tested source |
| Optional SAM head | CPU CLI stop at step 2, resume to step 4, reuse feature cache and reload for inference | Finite scores and clipped boxes; 8/8 tensors changed; source hash matches |
| Automated tests | Seven `test_drone_training_*.py` modules | 61 tests passed |
| Immutable references | Recompute all published reference sizes and hashes | All 11 unchanged |

The optional-head fixture uses artificial proposals/labels with actual SAM features and is an engineering test only. The SAM inference check uses an explicitly supplied box; it is an oracle-style API/reload diagnostic, not an automatic detector evaluation. Four-step YOLO outputs provide no useful accuracy estimate. The paired smoke comparison uses validation frames solely to test the comparison command; it is not a held-out test result.

## Important fixes exercised

- The downloader requests the dataset repository and the real nested metadata archive path.
- The data adapter validates zero-based frame positions and native image shapes.
- Predictions entirely in letterbox padding are removed after clipping back to the original image.
- CUDA AMP uses BF16 on supported GPUs; unsupported devices use explicit FP32. Both arms abort together on non-finite losses/gradients.
- Recovery deserializes RNG byte tensors on CPU before restoring the CPU and CUDA RNG states.
- SAM decoder training includes the frozen `no_mem_embed` used by official image inference. Older incompatible adapter checkpoints are rejected.
- Full-set permutation sampling and a constant-size presentation hash avoid replacement sampling and quadratic replay bookkeeping.
- Formal validation defaults to once per training pass plus the final step. Fresh runs reject existing managed run artifacts.
- Calibration binds the arm, checkpoint, data/code identities and inference settings; swapped or stale calibration is rejected.

## Work still required on the training host

Download the full release, repeat the smoke check on the target GPU, run the declared full-data schedules, inspect pseudo-mask acceptance, calibrate on source validation and evaluate untouched test data. Save all actual model and evaluation artifacts. No full-data run, formal SAM refinement comparison, H100 throughput result, strict external-domain evaluation or T12-T14 tracker result is claimed here.

The formal commands use all 115,675 common-motion training frames for YOLO and all 114,025 positive training-frame records as candidates for SAM pseudo-mask generation. Validation/test and rejected pseudo-masks do not receive training updates. The 96.4 GB video payload needs additional storage for metadata, caches, pseudo-masks and checkpoints.
