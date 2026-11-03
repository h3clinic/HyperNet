# HyperNet v18: controlled patch-ranking proof

This prototype tests whether explicit conditional-match supervision can improve
our synthetic-only one-reference detector. **It did not demonstrate improved
real-photo detection or a win over OWL-ViT.** The negative results are retained.
It is an engineering experiment, not an established algorithmic novelty claim.

## Method

1. Our frozen v17 detector proposes up to 100 regions from a query photograph.
2. A separate shared CNN encodes one reference image and each proposed region.
   Its encoder starts from **our own v17 synthetic-only weights**, not a borrowed
   pretrained model. Region patches are 96 px, preserve aspect ratio, and use
   median-border RGB letterboxing.
3. The ranker retains all 256 ordered similarities between 16 reference cells and
   16 region cells. A small MLP produces a conditional-match logit. It uses no
   selected top-half parts, global-moment compression, or query ground truth.
4. Existing regions are reordered by that logit. Their coordinates and proposal
   count remain unchanged. Query patch descriptors are batched and cached across
   references by their original-image integer crop coordinates.

Each training quartet contains a reference, an independently rendered same-type
positive patch, a one-attribute negative patch, and an absent different-family
reference. Positive/near-negative scenes share nuisance, background, crop,
padding, and resizing. The wrong-reference comparison keeps the positive query
patch fixed, so generic query objectness cannot solve that comparison.

One class-balanced binary logistic objective anchors absolute scores:

`0.5 BCE(correct positive,1) +0.25 BCE(same-support near-negative,0)
 +0.25 BCE(wrong-support positive-patch,0)`

Program IDs, masks, and boxes define procedural labels/crops but never enter the
network. At inference, inputs are one reference RGB image and one query RGB
image. No real-image weight updates occur.

## Fixed proof and provenance

- 512 updates, batch 8: 4,096 quartets and 16,384 additional RGB input views.
- 2,387 distinct target programs and 16,384 distinct input rasters in this run.
- The own v17 initialization previously consumed 65,536 synthetic RGB views.
  Combined **processed exposures** are 81,920; this is not a claim of 81,920
  distinct images across both stages.
- 90,708 ranker parameters; 88,649 optimized. The separately frozen proposal
  model has 57,683 parameters.
- Training took 121.44 seconds on the recorded RTX 5070 Laptop environment, with
  about 0.85 GiB peak allocated CUDA memory for this process. Other GPU tasks ran
  concurrently, so timing is not an isolated speed benchmark.
- All 4,736 predeclared train/development/test/compositional quartets passed CPU
  preflight. Each retained at least eight distinguishing pixels after resizing.
- 24 CPU tests passed. Every saved consumption-ledger entry, source snapshot,
  and checkpoint hash was verified. No natural photographs trained the weights.

The frozen v17 generator and encoder implementations are vendored byte-for-byte.
`PUBLIC_PROVENANCE.json` records their source hashes and the required own-v17
checkpoint hash. Checkpoints contain no absolute filesystem paths; no metadata
rewrite was needed. `EXPORT_ALLOWLIST.json` lists exported source and
artifact paths/checksums for publishing. It does not authorize uploading photos.

## Results

The synthetic pair test has 256 held-out quartets. The important distinction is:

| Comparison | AUROC | Correct paired ranking |
|---|---:|---:|
| Correct type versus one-edit negative |0.5735 |63.28% |
| Correct type versus wrong-family reference |0.9724 |97.66% |

The model learned coarse-family rejection much better than fine type separation.
Mixed-negative AUROC hides this weakness and should not be the headline result.

On 256 **same full synthetic test scenes**, both methods used the same reference
images and the same 100 frozen proposals. Each method used its own 5% development
scene-maxima cutoff; patch cutoffs were not reused for scenes.

| Method | Presence AUROC | Unthresholded localized top1 | Calibrated localized recall | Actual false-alarm rate |
|---|---:|---:|---:|---:|
| Frozen v17 |0.5661 |33.33% |0.775% |3.150% |
| v18 reranking |0.5495 |31.01% |3.876% |8.661% |

Higher thresholded recall came with more false alarms, while ranking AUROC and
unthresholded localization declined. This is not a clean improvement.

An exploratory **old-development-only** real-photo pilot used the first 48
image-ID-sorted photos and all 12 existing references: 576 comparisons, 42
positive and 534 negative. Thresholds were fitted inside this pilot and are
descriptive, not fresh validation or a cutoff chosen for a final test.

| Method | Micro presence AUROC | Localized hits | False alarms | Unthresholded localized hits |
|---|---:|---:|---:|---:|
| Saved v17 development subset |0.4463 |1/42 |26/534 |2/42 |
| v18 proof |0.4556 |0/42 |26/534 |2/42 |

Per-category metrics, ignored/crowd rules, raw predictions, arrays, and hashes
are retained in the result artifacts. The v18 training/evaluation runners did not open v17 fresh TEST images or
per-example records; no v18 fresh-real-test result is reported. The wider
project had aggregate v17 fresh-test findings before v18 was complete. No
additional training was performed after this fixed proof.

## Run the code

Install a PyTorch build suitable for your platform, then the dependencies in
`requirements.txt`. Exact measured versions are recorded in provenance; other
versions/backends have not been checked for bit-identical rendering/training.

From the repository root, first run `cd experiments/v18`. The own v17
checkpoint is already available at `../v17/runs/primary_2048/final.pt`.
From the v18 directory:

```bash
python -m pytest -q tests
python scripts/train_pair_ranker.py --own-checkpoint ../v17/runs/primary_2048/final.pt --run runs/reproduction --steps 512 --batch-size 8 --device cuda
python scripts/evaluate_ranker.py --checkpoint runs/reproduction/final.pt --own-checkpoint ../v17/runs/primary_2048/final.pt --output experiments/reproduction
```

The script refuses to overwrite a nonempty run.
The initializer must match the recorded own-v17 checkpoint and frozen encoder
source. No pretrained model is downloaded automatically.

Inference is available through `hypernet_v18.reranker.FrozenPatchReranker`:
construct it with the v18 and own-v17 checkpoints, call `enroll(reference_rgb)`,
then `encode_query(query_rgb)` and `predict(bank,encoded_query)`. For multiple
references, `predict_many(banks,encoded_query)` shares region-feature work.
Input arrays are RGB uint8; output boxes use original-image absolute xyxy
coordinates. Scores are uncalibrated logits, not probabilities.

The natural-photo pilot runner is archival: it requires the separately supplied
DEV-only COCO manifest and hash-checked licensed photos in its documented local
layout. It performs no downloads. Photos and the original workspace manifest
are excluded from the publication allowlist.

## Limits

Positive crop labels mean the type is present; they do not guarantee that the
crop is a tight IoU-qualified detection box. The score head has no box-quality
supervision. A one-edit pair visibly differs at fixed nuisance, but this does
not prove that entire type distributions remain separable under every allowed
deformation. Phase/orientation and nuisance transformations may overlap.

The inherited v17 detail-reconstruction loss acted at the early stem. Its inputs
already included the contrast/derivative targets, so a low reconstruction loss
did not prove that final descriptors or pooled matching cells retained tiny
details. V18 applies the discriminative loss to the end score, but the present
proof still shows weak fine-type discrimination and weak real-photo transfer.
