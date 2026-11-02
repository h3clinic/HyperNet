# v0.12: fresh 2D local-detail hybrid

The predeclared primary, `native_patch`, localizes and accepts **253/256 positive TEST scenes (98.83%)**, with **13/256 false positives (5.08%)**. The previous CNN on the same fresh images reaches **13/256 (5.08%)**, with **11/256 false positives (4.30%)**. The primary gain is **93.75 percentage points**, 95% paired family interval **[89.45, 97.66]**. The matched compressed-evidence control reaches **22.27%** recall with **8.59%** false positives; the native-minus-compressed gain is **76.56 points [69.53, 83.59]**.

This is a controlled synthetic **CNN-plus-analytic matcher**. It does not establish natural-photo category recognition, human-like reasoning, new mathematics, or superiority to published natural-image detectors. The historical v11 4.30% result came from another cohort; both methods are rerun here on identical fresh scenes.

| Method | Correct localized positives | Localized recall | False positives | AUROC |
|---|---:|---:|---:|---:|
| old_cnn | 13/256 | 5.08% | 11/256 (4.30%) | 50.19% |
| outline_only | 24/256 | 9.38% | 24/256 (9.38%) | 50.87% |
| centroid_only | 16/256 | 6.25% | 14/256 (5.47%) | 53.52% |
| local_moments | 250/256 | 97.66% | 19/256 (7.42%) | 98.63% |
| lowres_patch | 57/256 | 22.27% | 22/256 (8.59%) | 67.77% |
| native_patch | 253/256 | 98.83% | 13/256 (5.08%) | 99.51% |

All DEV and TEST metrics, including proposal recall, paired ranking and single-box AP50, are in [aggregate_metrics.json](aggregate_metrics.json). All six full score logs preserve every region, score, evidence field and relative image ID. [Illustrated report](report.html), [fixed example with cutoffs](assets/fresh_patch_comparison_labeled.png), [all 16 errors](assets/fresh_all_errors.png).

## Protocol and data budget

Each split has 128 previously unused composition families, 256 positive and 256 negative query rows. Opposite enrollments reuse each pair of scene images with reversed labels, so a family has four dependent rows. The full fresh cohort contains 512 unique support images and 512 unique query images. The 256 compositions are new combinations of familiar operators, not new natural semantic classes. Fresh images were generated only after the method and protocol freeze; overlap checks against 9,264 previous main/preview RGB files passed.

The reused 897,207-parameter CNN trained on 2,048 old composition families, 8,192 unique procedural RGB images and 1,024 optimizer updates. v12 adds zero optimizer updates. Method development used 64 old TRAIN families and one old DEV pass. There are no natural training images or borrowed weights. Enrollment supplies one 96px RGB reference and its box; inference receives the full 256px RGB query without query labels, boxes, masks or geometry parameters.

The primary preserves fine RGB evidence outside the CNN descriptor: filled-outline alignment, at most twelve reference-selected interior gradient patches, local normalization, five one-pixel shifts and a fixed outline/detail score. The matched `lowres_patch` control keeps proposals, native alignment, patch positions/counts and weights fixed while reducing/re-enlarging only fine RGB evidence. `centroid_only` and `local_moments` are separately reported, more domain-specific interior-feature controls. Native patches still undergo affine interpolation.

Each method's fresh DEV negatives set its cutoff at no more than 5% empirical FPR; TEST does not adjust it. Intervals use 10,000 paired draws of 128 TEST families, seed 1812, keeping both enrollments and paired queries together. They are conditional on this generator, existing CNN, selected method and fixed DEV cutoffs, without threshold refitting or simultaneous-comparison correction. Single-box AP50 is not full COCO AP.

## Reproduce inference

From the repository root, install `python -m pip install -e ".[neural,data,dev]"`, then:

```bash
python examples/download_v12_assets.py
python examples/replay_v12.py --output results/replay_v12_smoke
python examples/replay_v12.py --full --output results/replay_v12_full
python publication/verify_v12_results.py
python publication/verify_snapshot.py
```

The downloader verifies the two [release assets](https://github.com/h3clinic/HyperNet/releases/tag/v0.12.0) against [release_assets.json](release_assets.json). The exact checkpoint is our own synthetic-trained proposal CNN, with SHA-256 `15c64b81747f3ebec224ef19621793c2d24ed9dd6f93de25c6ad5f7c5c04d49a`. The RGB ZIP contains only generated pixels and a portable manifest. It contains no pretrained third-party assets or natural photographs. Paths in the manifest are relative to the extracted directory. The pure arithmetic verifier requires neither pixels nor weights.

The portable replay invokes the unchanged inference implementation and compares against published predictions. Its default sample contains four deterministically selected enrollments (two DEV, two TEST), producing 48 method predictions. Floating-point comparisons allow at most 1e-6 absolute error and zero relative tolerance across environments; boxes, IDs, labels and acceptance decisions must match exactly. Use `--atol 0` for exact scalar comparison. `--full` replays all 512 enrollments and recomputes the published aggregate metrics using the original fixed cutoffs. It marks new outputs as replication; it does not invent original generation times or regenerate the freeze. The original local acquisition/audit drivers are retained for source inspection, but depend on the earlier study layout. They are not the portable public replay entrypoint. The full renderer and deterministic seeds are also published in the package and [dataset plan](dataset_plan.json).

## Audits, provenance and remaining limits

The [independent audit](evidence/fresh_detail_run_audit.json) checked all **168 metric fields, 6,144 localization rows, 24 interval values, 96 exact full-prediction replays and 16 direct original-CNN replays**, shared proposals, unchanged weights, and the matched low-resolution control. The complete public cohort includes image hashes, boxes and generation seeds in [cohort_manifest.json](cohort_manifest.json). [Public export provenance](public_export_provenance.json) records separate original and sanitized public hashes. Only absolute provenance paths and JSON serialization change; model/scoring source files and score logs are exact copies. Original hashes in preserved audit fields refer to original local JSON bytes, not the sanitized public copies.

A documentation filename collision temporarily overwrote an offline review after freeze. Its exact original bytes were restored to the frozen SHA-256; the replacement was retained under another name. [Recovery disclosure](evidence/pipeline_review_collision_recovery.json). The final audit verified all frozen behavior and development-evidence bindings. No model, threshold or result changed.

All three missed positives selected the correct candidate but were rejected. Eleven false positives accepted the opposite micro-pattern, and two selected a different family with a confusable connector profile. All sixteen errors retained the correct candidate proposal; this post hoc analysis did not tune the model. Clean, high-contrast, separated shapes favor foreground extraction and alignment. Occlusion, touching objects, complex backgrounds, perspective, semantic variation and natural-photo transfer remain unvalidated. Existing local correlation, silhouette registration and factorized evidence are established ingredients; the study contributes a tested combination and matched information-removal control, not a claim of new mathematical primitives.
