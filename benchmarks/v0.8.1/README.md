# HyperNet v0.8.1: audited fresh-photo comparison

The predeclared full method found **7/240 positive TEST pairs**, versus **20/240** for the previous scratch CNN. False alarms were **1/240 negatives**, versus **3/240**. Recall must be read together with false-positive rate.

The paired balanced-accuracy interval favors the previous CNN on this cohort; the predeclared full method did not improve it.

This is a direct comparison on the same category-presence pairs. It is not a state-of-the-art claim, a detector-AP benchmark or an equal-pretraining-budget comparison with OWL-ViT.

## Complete TEST results

Each method uses its own global threshold selected only on DEV to allow at most 12 false positives among 240 DEV negatives. Thresholds are then frozen. TEST false-positive rates are measured, not forced to 5%. All eight HyperNet variants are retained; the best-looking ablation is not substituted for the predeclared primary.

| Method | Recall | False-positive rate | Balanced accuracy | AUROC | Average precision |
|---|---:|---:|---:|---:|---:|
| Previous scratch CNN | 8.33% | 1.25% | 53.54% | 0.5481 | 0.5663 |
| Assignment | 7.08% | 0.83% | 53.12% | 0.5544 | 0.5716 |
| Assignment + foreground | 4.58% | 0.42% | 52.08% | 0.5372 | 0.5575 |
| Assignment + tolerance | 7.08% | 1.67% | 52.71% | 0.5559 | 0.5724 |
| Assignment + counterfactual | 7.50% | 3.33% | 52.08% | 0.5577 | 0.5573 |
| Full v0.8.1 (predeclared primary) | 2.92% | 0.42% | 51.25% | 0.5412 | 0.5444 |
| Fixed features / original matching | 6.25% | 4.17% | 51.04% | 0.5610 | 0.5504 |
| Fixed features / full matching | 6.25% | 2.92% | 51.67% | 0.5612 | 0.5699 |
| OWL-ViT Base/32 (pretrained comparator) | 44.17% | 2.92% | 70.62% | 0.8586 | 0.8814 |

## Paired change from the previous CNN

Full-method differences are percentage points. Intervals use the paired category-cluster percentile bootstrap: 10,000 resamples, seed 1807, and fixed DEV thresholds. They are approximate because there are only eight TEST category clusters and query photos can recur across categories or support episodes.

| Metric | Difference | 95% interval |
|---|---:|---:|
| Recall | -5.42 | [-10.00, -1.25] |
| False-positive rate | -0.83 | [-2.08, +0.00] |
| Balanced accuracy | -2.29 | [-4.17, -0.42] |

All ablation and OWL contrasts, plus localization diagnostics, appear in [paired_analysis.json](paired_analysis.json). Localization counts use the top selected box and are not detection AP.

## Protocol and data budgets

- 16 categories: eight DEV and eight TEST. Three support episodes per category, 48 independent reference models, 960 pairs total.
- DEV contains 480 pairs (240 present, 240 absent); TEST contains 480 pairs with the same balance.
- Each learned HyperNet model trains from one source photograph and one annotated support box for 120 steps. Six learned scoring variants share that reference-specific CNN. No external image training, pretrained weights, learned teacher, query labels or query-time updates are used.
- Fixed controls have no learned parameters. Image-guided OWL-ViT Base/32 uses the official pretrained checkpoint, one support image and no text prompt or target-time updates. Its external pretraining is substantial and not counted as one-image learning.
- Support sources are disjoint from queries. Photos and category annotations can be imperfect; distinct file identities do not verify distinct physical objects or capture sessions.
- Repeated query images make pair rows dependent. Development photographs used for method design are a separate budget from reference-model training.

## Numerical repair and preserved evidence

The first attempt stopped after 47 reference checkpoints and 15 category outputs when interpolation produced a float32 value just above one. That incomplete evidence was preserved. Version 0.8.1 changes only a four-dtype-epsilon boundary tolerance and nonmutating clipping; valid-input outputs and RNG states were independently checked for bitwise agreement. All 48 CNNs were retrained and all 960 pairs rescored with the corrected code. The same cohort had already been partly scored, so this rerun is disclosed rather than described as untouched after the repair. No model, scoring, support-selection or threshold-selection parameter was chosen from those scores.

## Verify the published arithmetic

From the repository root, after installing the development dependencies:

```bash
python publication/verify_snapshot.py
python publication/verify_benchmark.py benchmarks/v0.8.1
```

[scores.jsonl](scores.jsonl) contains all 8,640 model/pair scores, COCO image IDs, labels, split/category IDs and selected boxes. No photographs, weights, local paths or full private provenance are included. Boxes use processed-query coordinates. The verifier independently reconstructs each DEV cutoff and TEST confusion, AUROC and average-precision metrics using scikit-learn; it also checks identical pair identities across all methods. It does not rerun image inference or prove provenance.

[aggregate_metrics.json](aggregate_metrics.json) retains DEV/TEST results, per-category metrics, category-cluster intervals, data budgets and source evidence hashes. [arithmetic_verification.json](arithmetic_verification.json) records the public-data check. Audited source-artifact hashes are retained without machine-specific paths.

[public_cohort_manifest.json](public_cohort_manifest.json) supplies all pair identities, COCO source IDs, original and stored image hashes, image dimensions, exact support crops, evaluation boxes and public source/license metadata. This supports independent reconstruction without redistributing the photographs. Read original RGB with EXIF transpose, crop supports to the recorded integer coordinates, and apply Pillow LANCZOS thumbnail resizing with maximum side 384. The published package contains the original preprocessing implementation; file/pixel hashes expose differences caused by source files or library versions. Source image licenses remain those recorded by COCO.

No claim of general semantic invariance, biological equivalence or superiority to established vision systems follows from this small benchmark.
