# HyperNet v17: measured synthetic-only one-reference detection pilot

**The pilot did not beat OWL-ViT or demonstrate a benefit from the proposed spatial matcher.**

On 576 fresh photos, evaluated against 12 fixed single-reference categories, HyperNet localized 0 of 472 positive target/photo pairs. OWL-ViT localized 193. HyperNet produced 327 false alarms across 6440 negative pairs, versus 295 for OWL.

| Method | Macro localized recall | False alarms / 10,000 negative pairs (macro) | Presence AUROC |
|---|---:|---:|---:|
| HyperNet spatial matcher | 0.00% | 503.1 | 0.5267 |
| Matched bag-of-parts control | 0.29% | 487.2 | 0.4941 |
| Untrained initialization | 0.00% | 539.8 | 0.5033 |
| Raw template matching | 0.00% | 366.0 | 0.5195 |
| Official image-guided OWL-ViT B/32 | 48.86% | 459.7 | 0.9079 |

The paired HyperNet minus OWL macro recall difference is -48.86 percentage points (95% photo-bootstrap interval -52.79 to -44.85). Macro rates give each category equal weight; the counts above are pooled. A zero-hit percentile-bootstrap interval can collapse to zero and does not prove exact population equivalence.

The implementation uses six mathematical 2D construction families, independent reference/query instances, structural hard negatives, and a CNN with dense local matching. Both trained arms used exactly the same 32,768 episodes: 65,536 unique RGB inputs and 10,512 distinct target structural programs. Those are procedural programs, not 10,512 natural semantic categories. The two arms reuse the same images; their combined exposures must not be counted as twice as many distinct images. The 57,683-parameter primary model was initialized randomly.

The full generator preflight and consumption audit caught and corrected identical ridge labels, disappearing thin strokes, visually ambiguous scene labels, foreground-frame misalignment, and numerical cancellation in the control. Every planned episode passed before training, and all 64 completed ledger shards were verified. These checks establish execution and data integrity, not recognition success.

On separate synthetic development diagnosis, the primary model had 91.3% proposal coverage but only 34.8% top-one localization on 46 positives. Wrong-family references affected its predictions; near-matches remained difficult. Ground-truth support extents used only as a diagnostic did not rescue ranking. The detail reconstruction loss acts at the early stem, so its low value does not prove the final pooled matching descriptors preserve fine details. This identifies a concrete direction for a separately recorded follow-up; it does not establish a causal explanation by itself.

No algorithmic novelty is established. Formula-generated pretraining, local correlations and geometric/template matching have substantial prior work; see research/novelty_scope.json. The no-geometry control changes the aggregation operator as well as removing spatial alignment.

All TEST photos were acquired only after checkpoint, behavior and DEV-threshold hashes were frozen. No TEST exclusions or replacement sampling occurred. The frozen metadata contains 472 positive and 6,440 negative pairs. An early narrative arithmetic error said 572 positives; research/metadata_count_correction.json records that correction. The roster, labels, calibration and evaluated arrays were always correct.

The measured comparison uses all 12 targets on 576 eligible fresh query photos.
Fresh means unused by prior HyperNet iterations; COCO may occur in OWL training.
Each method receives the same one annotated reference crop per target. HyperNet's
weights use 32,768 synthetic episodes (65,536 RGB input views) per trained
arm and no borrowed pretrained weights. There are also 288 labeled natural DEV
photos for engineering/calibration. This is not a no-natural-development-data claim.

The primary metric is macro top-one localized recall at IoU>=0.5, using one global
threshold fitted to nominal5% DEV negative-pair FPR. TEST false-positive rates and
paired photo-cluster intervals are reported. This custom metric is not COCO AP;
the result concerns the pinned image-guided OWL-ViT B/32, not the whole OWL family.

## Replay

See REPLAY_USAGE.txt. Run `python scripts/replay_published.py --self-test`, then
use published_benchmark.json with explicit data-cache, checkpoint and new output
paths. Install an appropriate PyTorch build plus requirements_replay.txt. Unit
tests additionally require requirements_tests.txt. Run `python -m pytest tests`.
The public helper downloads official COCO objects on request and verifies both
original-file and decoded-pixel hashes. No photos or Google weights are bundled.
The helper retains the published DEV cutoffs and verifies expected output arrays.
For the optional OWL baseline, run `python scripts/download_owl_baseline.py
--destination /path/to/owlvit-base-patch32` to fetch the exact official revision
and copy the supplied acquisition metadata after verifying every downloaded hash.

The training source and final admissions live under runs/primary_2048 and
runs/bag_2048. Their immutable source_snapshot folders, actual-consumption ledger
shards and histories accompany the own primary final, bag final and shared random
checkpoint. The bag run's duplicate random initialization and all exploratory,
smoke, failed or diagnostic checkpoints are omitted. Original completion records
retain hashes for checkpoints they describe, including omitted duplicates.

## Provenance and scope

artifact_sources.json records original-source and public-artifact SHA256 values.
JSON copies remove workstation filesystem paths while preserving measurements.
Historical hash bindings can refer to omitted external sources; the portable
published_benchmark.json is the operative replay manifest. EXPORT_ALLOWLIST.json
lists the exact bundle files and SHA256 values except its own self-hash, which is
recorded in a separate export receipt outside the bundle.

The original benchmark-orchestration source is archived under
audit/original_evaluation for inspection. It depends on the original private
acquisition workspace; use the public replay helper, not those orchestration
scripts. The private-acquisition test_detection_metrics.py is omitted; portable
metric tests and helper gate/hash/crop tests are included instead. All other
included unit tests operate on generated arrays/temporary own checkpoints.

Synthetic assets are mathematical 2D scenes, not natural photos. Real photo
derivatives and failure galleries, external datasets, OWL weights, old private
workspace files and unrelated repository changes are excluded. Model limitations,
data licensing metadata, comparisons and novelty qualifications are retained.
