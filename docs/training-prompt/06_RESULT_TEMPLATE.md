# Detection and tracking experiment report

Use `NOT_RUN`, `PROVISIONAL`, `MISSING_DATA`, or `NOT_APPLICABLE` where appropriate. Do not replace missing measurements with zero. All result cells below are intentionally empty.

## Experiment identity

| Field | Recorded value |
|---|---|
| Run ID and task ID | |
| Repository commit and source hashes | |
| Model architecture and generic initialization hashes | |
| Train, validation, and test manifest hashes | |
| Capture-group/source-video overlap audit | |
| Annotation review status and dataset exposure record | |
| Seeds, input size, microbatch, and accumulation | |
| Optimizer, learning-rate schedule, and precision policy | |
| Maximum updates, stopping rule, and actual sample coverage | |
| Checkpoint selection and threshold calibration rules | |
| Evaluator, max detections, area bins, and ignore handling | |
| Hardware, dependency lock, and measured training duration | |
| Future-frame lookahead and complete inference latency | |
| SAM prompt source, trainable modules, and memory policy | |

## Detection results

Record one row per system, dataset, split, and seed. Report the sample counts actually evaluated.

| System | Domain and split | Videos / frames / GT boxes | AP50 | AP50-95 | Precision | Recall | F1 | FP/frame | Tiny recall | Frozen threshold | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: RGB | | | | | | | | | | | NOT_RUN |
| B: RGB + D input | | | | | | | | | | | NOT_RUN |
| B-FS: frozen SAM2 Tiny | | | | | | | | | | | NOT_RUN |
| B-AS: adapted SAM2 decoder | | | | | | | | | | | NOT_RUN |
| B-DH: SAM2 candidate detection head | | | | | | | | | | | NOT_RUN |

Also report TP/FP/FN, score floor, NMS IoU, max detections, candidate K, and the definition of the tiny-object size bins. Distinguish native-resolution object size from resized network-input size. Provide per-video results, per-domain macro averages, and pooled micro summaries.

State whether AP is standard COCO evaluation or a custom single-class implementation. A matching metric name does not establish identical ignore, interpolation, area, or max-detection semantics. Record the actual test FP/frame attained with validation-frozen thresholds.

## Paired effects and ablations

| Comparison | Changed component | Recall delta and paired video-level 95% CI | AP50-95 delta | FP/frame delta | p95 latency delta | Seed agreement | Conclusion |
|---|---|---|---|---|---|---|---|
| B minus A | Input motion residual | | | | | | |
| B-AS minus B-FS | SAM2 decoder adaptation | | | | | | |
| B-DH minus B | SAM2 candidate head | | | | | | |
| Local + global minus global-only residual | Local parallax compensation | | | | | | |
| Suppressed minus unsuppressed residual | Texture suppression | | | | | | |

Resample complete videos or capture groups for uncertainty estimates; adjacent frames are not independent replicates. System-level gains with extra SAM computation must also report their extra compute and latency.

## Sequential tracking results

Use complete held-out sequences and identical initialization protocols. Automated tracking starts from detector predictions. Any ground-truth-initialized single-object experiment must be a separate row with that initialization explicitly stated.

| System | Domain / videos | Initialization | SOT success AUC / normalized precision | Track coverage | Loss events | Reacquisition delay | False tracks | IDF1 / HOTA if labeled | End-to-end FPS / p95 latency | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| A + common tracker | | automatic | | | | | | | | NOT_RUN |
| B + common tracker | | automatic | | | | | | | | NOT_RUN |
| V0: locked B detector only | | automatic | | | | | | | | NOT_RUN |
| V1: same B + constant-velocity tracker | | automatic | | | | | | | | NOT_RUN |
| V2: V1 + SAM2 memory | | automatic | | | | | | | | NOT_RUN |
| V3: V2 + Action Bank | | automatic | | | | | | | | NOT_RUN |

The A/B tracker pair isolates the input-motion contribution. V0-V3 then add one temporal component at a time while holding B fixed; reuse the B tracker measurements for V1 when the complete configurations match. V0 has no persistent identity, so mark identity-based tracking columns `NOT_APPLICABLE` and retain its frame-level detection measurements.

Define match IoU, visibility/absence labels, tolerated gaps, loss events, false tracks, and reacquisition latency before testing. MOT identity metrics require persistent target IDs and complete multi-object annotations. Do not infer them from isolated positive boxes. Record initialization failures and never remove those sequences from the automatic-system aggregate.

## Failure analysis

- No proposal overlaps the drone: count by scale, domain, and scene.
- A correct proposal exists but semantic classification or temporal selection rejects it.
- SAM refinement improves or damages an originally correct box; report both directions.
- Mask failure triggers the defined detector-box fallback.
- Birds, aircraft, leaves, clouds, structures, camera shake, rolling shutter, occlusion, or hovering cause errors.
- Compensation failure, zero-residual fallback, decoding failure, or cache errors affect coverage.
- Tiny-object pseudo-mask acceptance or quality varies across domains.
- Tracker drifts, switches identity, updates memory with a distractor, or fails to reacquire.

Include representative successes and failures from every domain and size bin. A visually appealing demonstration is not a substitute for a complete evaluation.

## Final delivery checklist

- [ ] Freshly trained YOLO A/B best and last checkpoints with full configuration.
- [ ] SAM2 decoder/head checkpoints and exact base-weight dependencies.
- [ ] Resume checkpoints from these runs, containing optimizer, scheduler when used, RNG, step, epoch, sampler cursor, and run identity.
- [ ] Asset hashes, frozen split manifests, dependency lock, and source revision.
- [ ] Validation calibration, frozen test protocol, raw predictions, per-video metrics, and statistical scripts.
- [ ] Sample coverage, throughput, memory use, and full-pipeline latency measurements.
- [ ] Ground-truth-free inference command and a continuous-video demonstration.
- [ ] T12-T14 tracker configuration and sequential evaluation, if delivering the tracking extension.
- [ ] Independent model loading and metric readback.

## Questions the final report must answer

1. Does input motion improve recall on the team's data, at what false-positive rate and latency?
2. How does each system perform separately on ARD100, DJI, Anti-UAV, and NPS? Which tests meet strict domain-holdout conditions?
3. What do motion compensation, pretrained SAM2, and SAM2 adaptation each contribute?
4. Can the complete system maintain and reacquire tiny-drone tracks, and under which visibility conditions does it fail?
5. Which system is supported for deployment by these measurements? If added complexity has no reliable benefit, retain the simpler measured baseline.
