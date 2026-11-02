# v0.11: controlled 2D micro-pattern detection

The added micro-pattern supervision **did not establish a primary improvement**. TEST localized recall was **11/256 (4.30%)** for the micro-aware CNN versus **14/256 (5.47%)** for the same CNN trained without the candidate micro-negative match objective. Both had **11/256 false positives (4.30%)**. The paired localized-recall difference was **−1.17 percentage points**, 95% family-bootstrap interval **[−3.52,+0.39]**.

The original project goal remains recognizing any example of the same natural category from one reference. This experiment is a narrower diagnostic: it deliberately rejects a near-identical broad shape whose fine pattern differs. It uses mathematical 2D drawings, not natural photographs or 3D rendered objects. It does not establish semantic-category transfer, human-like reasoning or superiority over published models.

## Complete metrics

Each split has 256 positive and 256 negative query rows. Thresholds are independently fixed using DEV negatives at no more than 5% false positives. TEST false-positive rates are measured, not fixed to 5%. The primary metric requires acceptance and IoU ≥0.5 with the matching object.

| Split | Method | Localized recall | Presence recall | False positives | Proposal recall | Paired ranking | AUROC | Single-box AP50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DEV | Family-only CNN | 5.08% | 5.08% | 4.69% | 100.00% | 53.12% | 50.15% | 39.95% |
| DEV | Micro-aware CNN (primary) | 5.08% | 5.08% | 4.69% | 100.00% | 55.86% | 50.42% | 38.68% |
| DEV | CNN proposals + classical NCC (secondary) | 7.03% | 7.03% | 4.69% | 100.00% | 74.41% | 51.85% | 35.80% |
| TEST | Family-only CNN | 5.47% | 5.47% | 4.30% | 100.00% | 53.91% | 50.13% | 31.87% |
| TEST | Micro-aware CNN (primary) | 4.30% | 4.69% | 4.30% | 100.00% | 51.17% | 50.15% | 30.90% |
| TEST | CNN proposals + classical NCC (secondary) | 7.03% | 7.42% | 4.30% | 100.00% | 71.48% | 51.79% | 29.12% |

Proposal recall is 100%, but the CNN scores remain near chance in pooled discrimination. The raw-pixel hybrid ranks the positive above its controlled negative in 71.48% of TEST pairs, while only reaching 51.79% pooled AUROC and 7.03% calibrated localized recall. Within-pair ranking does not imply reliable acceptance of a previously unseen scene. Single-box AP50 uses one selected box per scene with tie-aware ranking; it is not full COCO detection AP.

## Data and training budget

- Grammar universe: 4,096 compositions from four factors with eight choices each. The individual operators are familiar across splits; held-out whole compositions are new. These are not 4,096 natural semantic classes.
- TRAIN: 2,048 compositions; DEV: 128; TEST: 128; 1,792 reserved. Training and held-out family IDs, object seeds and decoded RGB image content do not overlap.
- All generated data: 4,608 supports plus 4,608 distinct query images = 9,216 distinct RGB images. Opposite enrollments reuse each query image with reversed labels, producing 9,216 query rows.
- TRAIN per arm: 4,096 distinct supports plus 4,096 distinct query images; one epoch, 1,024 updates, four enrollments/eight query rows per update. Four exact quarter-turn permutations of each support produce 16,384 reference CNN views; query exposures are 8,192.
- Both arms: 897,207 parameters, seed 111, identical random initialization, batch ordering and proposal-training roster. AdamW, learning rate .001, weight decay .0001. Final checkpoint is fixed by schedule, with no held-out early stopping.
- Natural training images: **0**. Externally supplied pretrained weights: **0**. Inference inputs: one support RGB and its enrollment box, plus the complete query RGB. Query labels, boxes, masks and pose are not inference inputs.

The CNN preserves a native-resolution branch before stride 2, but ultimately compresses each 24×24 region feature grid into 128 values. Four reference rotations supply approximate orientation coverage; arbitrary query angles are not exactly invariant. Both arms learn an all-object foreground map and balanced region matching. The micro-aware arm additionally includes candidate micro-negative BCE and a weighted paired margin; the ablation tests that bundle rather than separating its terms.

Two equal-area internal holes change orientations without changing the broad outline. Support and candidate macro-instances have independent small parameter jitter. The minimum nominal query detail width is 4.32 pixels. Paired raster ink differences average 0.1473% and reach 1.8878% (23 pixels), despite equal continuous hole area/count. All 4,608 query candidates are connected; two TRAIN supports retain an isolated one-pixel artifact. Scenes contain separated objects against simple backgrounds, so the proposal task is favorable.

## Secondary classical hybrid

The frozen [diagnostic plan](analytic_detail_bypass_plan.json) was declared before either complete CNN prediction log existed; the first CNN checkpoint already existed. It reuses the exact micro-aware CNN proposal boxes and scores normalized grayscale crops at 96px over 24 angles × 3 scales × 5 translations. It fits no additional weights and uses one enrollment source, but depends on the trained CNN proposals. Its larger rotation/alignment search is a confound: the comparison cannot uniquely locate CNN information loss. Full-patch matching can favor outline/background alignment over individual minute features. The primary CNN comparison is unchanged.

## Post hoc broad-type localization

Ignoring microstate and the acceptance threshold, the top selected box overlapped the broad shape-family candidate in **393/512 (76.76%)** TEST rows for the family-only CNN, **401/512 (78.32%)** for the micro-aware CNN, and **288/512 (56.25%)** for the classical hybrid. The candidate's macro-family is present in every scene. This is **forced-choice type-present localization only**, not absence detection, calibrated category accuracy, natural-photo accuracy or a replacement primary outcome. The descriptive check was chosen after the main results; reused query/enrollment rows remain dependent. See the [bound diagnostic](type_present_localization_diagnostic.json).

## Uncertainty and independent checks

[Paired analysis](paired_analysis.json) uses 10,000 shared bootstrap draws of all 128 TEST families, seed 1811. Both enrollments and all four paired query rows stay together. The five contrast intervals are conditional on this generator, one model initialization and the fixed DEV cutoffs; secondary intervals are not multiplicity-corrected. Reused query pixels are not independent observations.

[Independent CNN audit](inference_metric_audit.json) reproduces 56 metric values, 2,048 localization rows and 32 exactly matching predeclared inference replays. [Independent paired review](detector_analysis_independent_review.json) reproduces all 15 contrast/interval values. The [hybrid metric review](analytic_detail_bypass_independent_review.json) checks 28 metrics and 1,024 rows; its [replay audit](analytic_detail_bypass_replay_audit.json) repeats four predeclared cases exactly. Frozen sources and checkpoint hashes remain unchanged.

## Reproduce or verify

Follow the [root README](../../README.md#reproduce-the-2d-detector) to generate the images, train both CNNs, audit inference and open the all-scene gallery. No pixels or trained parameters are distributed. The [dataset plan](dataset_plan.json), fixed renderer code and original hashes make the cohort reconstructable. [Held-out manifest](heldout_manifest.json) contains labels, image hashes and reference/query boxes without pixels or private paths.

To verify saved results without training:

```bash
python publication/verify_v11_results.py benchmarks/v0.11
```

The verifier reconstructs DEV cutoffs, all 84 metric values, all 3,072 selected/proposal IoUs and the 15 paired contrast values. The three compact JSONL logs retain every DEV/TEST prediction, including failures. [Export provenance](export_provenance.json) retains original artifact hashes and specifies path sanitization. Previous v0.8.1, v0.9 and v0.10 experiments remain unchanged.
