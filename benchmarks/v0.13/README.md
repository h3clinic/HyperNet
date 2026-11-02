# v0.13: real-photo transfer failed in this cohort

**Zero of 96 positive TEST photos were correctly localized and accepted by any of the four methods in either operating mode.** The native primary using its unchanged synthetic cutoff rejects all positive and negative TEST scenes. Real-photo DEV calibration accepts 7/96 positive scenes and 2/96 negative scenes for the native matcher, but none of those positive boxes correctly localizes an annotated same-category object. Its 52.60% presence balanced accuracy is not successful localization.

The previous 98.83% v12 result concerned controlled synthetic micro-patterns. v13 asks whether another natural photograph contains any instance of the reference category. These are different tasks and benchmarks. [Metric-only illustrated report](report.html), [all metrics](aggregate_metrics.json), [source/protocol](evidence/real_photo_protocol_freeze.json).

| Mode | Method | Correct accepted localization | Positive scenes accepted | False positives | Presence balanced accuracy | Proposal coverage IoU≥.5 | Top-one IoU recall |
|---|---|---:|---:|---:|---:|---:|---:|
| transferred | old_cnn | 0/96 | 0/96 | 0/96 | 50.00% | 14.58% | 1.04% |
| transferred | native_patch | 0/96 | 0/96 | 0/96 | 50.00% | 14.58% | 0.00% |
| transferred | old_cnn_grid | 0/96 | 0/96 | 0/96 | 50.00% | 66.67% | 6.25% |
| transferred | native_patch_grid | 0/96 | 0/96 | 0/96 | 50.00% | 66.67% | 0.00% |
| real_dev_calibrated | old_cnn | 0/96 | 14/96 | 11/96 | 51.56% | 14.58% | 1.04% |
| real_dev_calibrated | native_patch | 0/96 | 7/96 | 2/96 | 52.60% | 14.58% | 0.00% |
| real_dev_calibrated | old_cnn_grid | 0/96 | 12/96 | 17/96 | 47.40% | 66.67% | 6.25% |
| real_dev_calibrated | native_patch_grid | 0/96 | 3/96 | 2/96 | 50.52% | 66.67% | 0.00% |

## Why the observed detector fails

Learned proposals cover 14/96 positive TEST scenes. Native matching selects the correct top box in 0/96 even if its cutoff is ignored. Fixed-grid windows cover 64/96, but native grid matching still selects no correct top box. The threshold therefore cannot rescue missing or wrongly ranked boxes. Both proposal coverage and category-level matching transfer are inadequate here. The data does not isolate CNN compression as the sole cause.

There is one automatically chosen reference per category. Annotation filters require at least 32px box edges and 2% frame area, not a clear, frontal or unoccluded subject. The first car reference is small and partly occluded, and 96px normalization can discard more information. Results are conditional on the selected references and do not show that every possible clearer reference would fail. No reference was replaced or method tuned after TEST scoring.

## Data and decision budgets

The frozen cohort contains 300 globally distinct original COCO val2017 photos: 12 references, 96 DEV queries (4 positive + 4 negative per category) and 192 TEST queries (8 positive + 8 negative per category). Category labels and reference identity are shared across DEV/TEST. This evaluates new photos, not unseen categories. Previous project categories were reused, while 2,472 prior COCO source IDs and 7,979 prior source/stored-input files were excluded or checked for exact overlap. File/pixel disjointness does not prove different physical objects, capture sessions or absence of near-duplicates. Annotation-negative scenes can contain missed objects.

The primary `transferred` mode uses unchanged v12 synthetic cutoffs and no real calibration labels. The secondary `real_dev_calibrated` mode uses 96 additional real DEV photos, with a pooled threshold accepting at most 2/48 DEV negatives per method. It changes no weights. Both modes use the same saved scores. TEST never selects a method, reference, coefficient or cutoff.

The inherited 897,207-parameter CNN trained on 8,192 distinct procedural RGB images. v13 adds zero optimizer updates, zero natural-photo weight training and no borrowed pretrained weights. Enrollment supplies one reference RGB plus one reference box. A 1.25× crop becomes an aspect-preserving 96px letterbox; query detail is bounded at 1024px long edge, CNN input at 512px, and all output boxes map to original coordinates. Labels/boxes from queries are read only for evaluation after inference. The 51-window grid is a separately declared diagnostic, not a silently substituted proposal method.

Localized recall requires acceptance plus overlap with any annotated same-category object at IoU≥.5. Presence metrics ignore box quality. Proposal and top-one IoU recall ignore cutoffs. Single-box AP50 is not official COCO multi-detection/multi-IoU AP. Paired bootstrap uses 12 category clusters, 10,000 draws, seed 1313, conditional on selected references and fixed cutoffs. With zero observed localized successes, its empirical [0,0] intervals are degenerate: they do not quantify an unseen-success rate or establish population equivalence.

## Try your own reference

Install from the repository root with `python -m pip install -e ".[neural,data,dev]"`.

```bash
python examples/download_v13_cohort.py --checkpoint-only
python scripts/serve_photo_lab.py --enable-uploads --checkpoint checkpoints/detector_micro_aware.pt
```

Open `http://127.0.0.1:8794/photo_lab.html`. Choose one reference, draw its subject box (or use the whole image), then choose a full query. Uploads stay in request memory and are not published. Enrollment extracts features and local patterns; it is not SGD training. The model remains experimentally unsuccessful on this benchmark.

```bash
python examples/photo_match.py --reference reference.jpg --query query.jpg --output results/photo_comparison.json
# Optional explicit reference box in original-image coordinates:
python examples/photo_match.py --reference reference.jpg --box 30 40 220 280 --query query.jpg --output results/boxed_photo_comparison.json
```

The default is the frozen synthetic cutoff. `--operating-mode real_dev_calibrated` uses the separately reported 96-photo calibration budget. The output preserves all four methods and scores, which are not probabilities. Neither manual pathway saves copies of input photos or updates weights.

## Reproduce the published failure

```bash
python examples/download_v13_cohort.py
python examples/replay_v13.py --output results/v13_replay
# Optional all 288-query replay:
python examples/replay_v13.py --full --output results/v13_full_replay
# Arithmetic checks need no images or model weights:
python publication/verify_v13_results.py
python publication/verify_snapshot.py
```

The default portable replay uses the first sorted TEST positive and negative queries and all four methods. It uses unchanged inference sources and published cutoffs in both modes, not a new experiment. `--atol 0` requests exact scalar comparison; default float tolerance is at most 1e-6 absolute, with exact boxes and decisions. Original experiment timestamps/freezes are not recreated. Acquisition and original audit drivers retain their historical layout assumptions; portable examples are the supported public entrypoints.

Natural photos are downloaded from their original official S3 object URLs with valid TLS. The helper checks raw SHA-256, decoded pixel hashes, dimensions and EXIF coordinates, and records individual Flickr/COCO sources and licenses locally. Existing different files are not overwritten. Natural photographs and local photo montages are **not** distributed in the repo or releases; original licenses apply (198 CC BY 2.0 and 102 CC BY-SA 2.0 photos). The source metadata is in [cohort_manifest.json](cohort_manifest.json) and [source_licenses.json](source_licenses.json). The reused synthetic-only CNN is the exact project-owned [v0.12 release asset](https://github.com/h3clinic/HyperNet/releases/tag/v0.12.0), SHA-256 `15c64b81747f3ebec224ef19621793c2d24ed9dd6f93de25c6ad5f7c5c04d49a`.

## Audit and provenance

The [independent audit](evidence/real_photo_independent_audit.json) verifies 1,152 rows, 2,304 scene/proposal IoUs, 2,920 metric numbers, 66 bootstrap numbers and 96 exact full-prediction replays. All 300 raw/decoded images and 395 tracked artifacts remain unchanged, with no model gradients or state changes. A separate coordinate audit checks 40,462 boxes, and the portable replay reproduces 8 predictions / 4,792 numeric values exactly. [Public export provenance](public_export_provenance.json) maps private local paths to logical identifiers and records original/public hashes separately. No natural-image base64 payloads or pixels are exported.

The original virtual-host download endpoint failed TLS hostname verification. The same official public S3 object path with valid HTTPS was used, preserving the frozen selection and audited bytes. [Transport recovery admission](evidence/real_photo_transport_recovery_admission.json). This transport repair did not modify the algorithm or choose a new dataset.
