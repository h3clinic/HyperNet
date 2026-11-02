# HyperNet v0.9: audited primitive-shape control

**The primary remains near chance.** The frozen procedural-detail CNN detected **10/180 positive TEST pairs** and falsely accepted **10/180 negatives**: recall **5.56%**, FPR **5.56%**, balanced accuracy **50.00%**. The scratch reference CNN detected **4/180** with **9/180** false alarms: balanced accuracy **48.61%**.

The paired balanced-accuracy change is **+1.39 percentage points**, 95% category-bootstrap interval **[-0.56, +3.33]**. This does not establish an improvement. The primary's pooled AUROC (0.5427) and category-macro AUROC (0.5417) also trail scratch (0.5521 and 0.5504).

This is a control using primitive generated shapes, not the final richer named-shape curriculum. It is not a human-vision, state-of-the-art or equal-pretraining-budget claim. All seven HyperNet controls and the separate OWL-ViT comparator are retained.

## Complete TEST comparison

Each method has its own global cutoff, chosen only on DEV to allow at most 9 false positives among 180 DEV negatives. TEST false-positive rates are measured, not forced to 5%. Pooled ranking metrics appear below.

| Method | TP /180 | Recall | FP /180 | FPR | Balanced accuracy | AUROC | Average precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scratch, one reference | 4 | 2.22% | 9 | 5.00% | 48.61% | 0.5521 | 0.5284 |
| Procedural restoration, frozen | 9 | 5.00% | 5 | 2.78% | 51.11% | 0.5325 | 0.5226 |
| Procedural restoration, adapted | 8 | 4.44% | 9 | 5.00% | 49.72% | 0.5301 | 0.5163 |
| Procedural detail, frozen (predeclared primary) | 10 | 5.56% | 10 | 5.56% | 50.00% | 0.5427 | 0.5340 |
| Procedural detail, adapted | 10 | 5.56% | 13 | 7.22% | 49.17% | 0.5293 | 0.5131 |
| Random CNN, frozen | 18 | 10.00% | 13 | 7.22% | 51.39% | 0.5392 | 0.5531 |
| Fixed shape descriptor | 11 | 6.11% | 14 | 7.78% | 49.17% | 0.5329 | 0.5223 |
| OWL-ViT B/32 (external pretraining) | 56 | 31.11% | 5 | 2.78% | 64.17% | 0.8093 | 0.8305 |

## Paired primary change

Percentage-point differences use 10,000 paired category-bootstrap draws, seed 1809, with the same sampled category indices for every method and fixed DEV cutoffs. There are only six TEST category clusters, queries recur, and only one shared-pretraining seed was evaluated. The intervals are conditional on these models/cutoffs and exclude training/calibration uncertainty. This post-evaluation paired analysis does not replace the original marginal intervals retained in the aggregate file.

| Metric | Difference | 95% interval |
|---|---:|---:|
| Recall | +3.33 | [-1.11, +10.56] |
| FPR | +0.56 | [-5.56, +5.56] |
| Balanced accuracy | +1.39 | [-0.56, +3.33] |
| Category-macro AUROC | -0.87 | [-5.80, +4.07] |

All control contrasts are in [paired_analysis.json](paired_analysis.json); they are descriptive and not multiplicity-adjusted. No best-looking control replaces the predeclared primary.

Localization remains weak: the primary's selected box reaches IoU≥0.5 on **2/180** positive pairs, and **1/10** accepted positives. **4/10** accepted positives have zero box overlap. This top-box diagnostic is not detector AP.

## Training and evaluation budget

- Two own synthetic backbones, initialized randomly: restoration and restoration plus 0.25 foreground mask/detail prediction plus 0.2 matched-nuisance triplet loss. Same CNN initialization and procedural sequence; no borrowed weights, natural-image pretraining, learned teacher or downloaded assets.
- Per backbone: 2,000 steps × 16 triplets; 32,000 base-program draws +32,000 mutant-program draws; 96,000 raw RGB exposures and 192,000 corrupted training-view exposures. Unique image counts were not measured. The CNN has 40,976 parameters; the auxiliary training-only head has 33,280.
- Primary frozen-detail model: one real reference and one support box for enrollment; zero real-image weight updates. Adapted controls: 120 updates on 297 patches from that one reference, with independent copies of the same synthetic checkpoint. Scratch control starts randomly. No query weight updates.
- 12 categories, six DEV and six TEST; three support episodes per category; 36 real supports and 218 new source photographs; 720 pairs total. DEV and TEST each contain 360 pairs, 180 present and 180 absent.
- 108 separate reference CNNs across scratch and two adapted controls. Fixed/random controls have no weight training. One shared synthetic CNN per frozen arm is reused across episodes.
- OWL-ViT B/32 uses its official external pretrained checkpoint, image-guided inference and no text prompts. Its natural pretraining budget and possible dataset overlap differ from HyperNet and are not established as absent.
- Support/query sources are separated. Previously studied target types can recur; freshness means new photo sources with ID/hash exclusions, not categories unseen by every model. Query boxes are evaluation-only.

## Independent fine-detail diagnostic

On 96 generated native32 comparisons, the combined detail arm wins 78 (81.25%) versus 77 (80.21%) for restoration: one net comparison. Downsampling 96px inputs to 32px reduces both. This diagnostic is not natural-photo recognition accuracy or a demonstrated detail breakthrough. See [detail_diagnostic.json](detail_diagnostic.json). Because the combined objective changes two terms, it does not isolate either term alone.

## Audits and reproducibility

The final independent audit checks 5,040 HyperNet scores, 720 OWL scores, 108 reference checkpoints and two synthetic backbones. Twenty-eight read-only inference replays match exactly. A preliminary procedural-training attempt failed on thick-ring geometry; that generator defect was repaired before the completed synthetic run and before fresh-photo evaluation. No fresh score selected the repair, hyperparameters or primary model.

```bash
python publication/verify_snapshot.py
python publication/verify_v09_results.py benchmarks/v0.9
```

[scores.jsonl](scores.jsonl) contains all 5,760 scores, pair/category/split identities, COCO IDs and selected boxes. [public_cohort_manifest.json](public_cohort_manifest.json) supplies exact support crops, source/stored dimensions and hashes, evaluation boxes and original license/source URLs. Photographs and weights are not redistributed. Coordinates refer to processed query images; preprocessing is in the published package.

[aggregate_metrics.json](aggregate_metrics.json) includes DEV/TEST, per-category and marginal interval results, budgets and evidence hashes. [arithmetic_verification.json](arithmetic_verification.json) independently verifies confusion/ranking metrics with scikit-learn, DEV cutoffs, pair identities, paired intervals and localization. It does not rerun inference or prove image provenance.

COCO labels can be incomplete; distinct files do not verify physical-instance/capture independence. Repeated photos, six TEST categories and a small hand-designed shape vocabulary limit interpretation. This result is evidence for further work, not a successful category-recognition system.
