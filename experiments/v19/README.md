# HyperNet v19: synthetic-only one-reference detection experiment

V19 does not beat official image-guided OWL-ViT. On 576 fresh photographs scored against 12 fixed one-reference targets, v19 correctly localizes 1 of 483 positive target/photo cases; OWL localizes 213. Primary category-averaged localized recall is 0.1302% for v19 and 52.8009% for OWL. V19 raises 324 false alarms in 6,429 negative cases (504.0 per 10,000 negative target/photo comparisons), versus OWL's 295 (458.9 per 10,000). These false-alarm rates are pooled; the full paired comparison also reports category averages and uncertainty. Each method's cutoff was fitted on the same old development cohort at nominal 5% negative-pair false alarms, then frozen. This is top-one localization at IoU >= 0.5, not COCO AP or ordinary classification accuracy.

The new model separates local/context features, uses a scale-normalized image pyramid, expands generic candidate coverage, and learns point correspondence plus region quality on mathematical 2D construction graphs. It was initialized randomly and trained on 32,768 procedural episodes, comprising 65,536 raw reference/query RGB views. This is the actual training volume, not millions of images. Four procedural grammar families generate many distinct programs; those programs are not natural semantic categories. There were no natural-image weight updates and no borrowed pretrained weights in HyperNet. Natural development images informed engineering, selection and threshold calibration. Each natural target has one fixed reference image. OWL retains its official pretrained weights; freshness is relative to HyperNet, not proof that COCO was unseen during OWL pretraining.

Held-out synthetic controls show local learning without reliable type discrimination. On identical inputs, trained point-candidate accuracy is 23.32% versus 3.07% for random initialization, and sampled-region quality AUROC improves from 0.560 to 0.819. Point correctness accepts a generated neighborhood around a visible counterpart or the unmatched option for a missing point; it is not exact-pixel localization. However, the correct shape beats its one-edit alternative in only 31 of 64 positive cases, versus 33 of 64 for random weights; corresponding AUROC is 0.470 versus 0.496. Full generic search gives 0 of 64 accepted localized hits for either weight state. Point and true-region diagnostics use generated annotations; full-search inference receives RGB alone. These synthetic diagnostics are distinct from the natural-photo benchmark. They show that the problem already exists within the synthetic distribution and cannot be attributed only to transfer from synthetic to real images.

Development candidate coverage improved from 14/42 to 42/42 targets, yet learned scoring ranked the wrong full-photo region first in all 42 pilot cases. In the complete 255-positive-case development set, only two unthresholded predictions localize correctly and none pass the prescribed cutoff. The loss averages near-alternative negatives with other objects/background and trains on 64 sampled regions per episode; it does not separately require a correct-versus-one-edit margin or optimize the maximum over thousands of competing proposals. This objective/search mismatch is a plausible explanation, not an isolated causal finding. Explicit counterfactual ranking and competing-region training are proposed follow-up experiments, not changes implemented or validated in v19.

The package preserves measured failures, random/final own checkpoints, source snapshots, actual-consumption ledgers and reproducible score arrays. The initial aborted 96-episode attempt is disclosed as additional discarded compute; its weights were not used to initialize the final run. No natural photos or Google weights are redistributed. Neither algorithmic novelty nor general natural-category recognition is established by this experiment.

The completed run trained from random initialization on 32,768 procedural episodes
(65,536 RGB input views). No natural images updated model weights and no borrowed
pretrained weights initialized HyperNet. Natural DEV photos did inform engineering,
inference-scale choice and acceptance thresholds; this is not a claim of zero natural development data.

The measured comparison uses the same one boxed reference crop for each of12 categories,
288 old DEV photos, and 576 eligible fresh TEST photos. The primary
metric is macro top1 localized recall at IoU>=0.5 at a global threshold fitted to5%DEV
negative-pair FPR. Actual TEST false alarms and paired photo-cluster uncertainty are reported.
This custom metric is not COCO AP. OWL refers to the pinned image-guided OWL-ViT B/32.

The earlier failed attempt is preserved in audit/failed_attempts.json. Its consumed
episodes are additional discarded compute, not part of the final checkpoint initialization.
Source snapshots, random/final checkpoints and actual optimizer-consumption ledgers are included.
Architectural novelty and universal semantic transfer remain hypotheses, not established claims.

## Replay

From this directory, run `python scripts/replay_published.py --self-test`. Install an
appropriate PyTorch build plus requirements_replay.txt. The published_benchmark.json
manifest binds every measured result, source hash, original-file and decoded-pixel hash,
and exact DEV cutoff. Replay requires a new cache/output directory; it does not recalibrate.
Example: `python scripts/replay_published.py --manifest published_benchmark.json --method v19 --split test --data-root /path/to/coco-cache --checkpoint runs/primary_4096_r2/final.pt --device cuda --output /path/to/new-replay`.

Official COCO images are downloaded only when replay is explicitly requested; no photos
or Google weights are bundled. Baseline OWL replay requires separately supplied official
weights with the published file hashes. v17 replay uses the existing sibling experiment's
own checkpoint. OWL replay validates the supplied official files, then stages hard links
(or byte-identical copies when links are unavailable) plus the bundled acquisition
metadata in a temporary directory. It does not modify the supplied checkpoint folder.
Run `python -m pytest tests` for local model/data/metric tests and
`python scripts/verify_training_v19.py --run runs/primary_4096_r2 --replay-count 9` to check
the saved training ledger and regenerate a bounded sample of its mathematical inputs.

PUBLIC_EXPORT_MANIFEST.json records original and exported hashes when private filesystem
provenance is normalized. Scientific source snapshots, tensors and measured numeric arrays
remain unchanged. EXPORT_ALLOWLIST.json is the complete bounded file inventory.

Scripts under audit/original_source and audit/original_helpers preserve historical
workspace-oriented analyses for inspection. They may require the original directory
layout; use the portable replay and training-verification commands above for this bundle.
