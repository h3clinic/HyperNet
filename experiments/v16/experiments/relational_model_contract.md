# v16 mathematical structural experiment

The frozen model uses 93,483 parameters initialized randomly. No natural image,
borrowed weight, teacher, or natural-image calibration is used by this trainer.
The primary and no-relation-auxiliary runs each consume 4096 optimizer batches:
64 type visits and two independent instances per visit, or 524,288 rendered
training views per run. Actual unique signatures and raster hashes are counted
by the consumption ledger; visits are not claimed as distinct types.

The generator is a deliberately narrow controlled grammar: 24 possible edges
on an axis-relative 4 by 4 planar lattice. Independent positive instances retain
the edge arrangement while varying spacing, bounded shears, local displacement,
edge curvature, width, material and background. The 32,768 base signatures are
revisited; nuisance-matched hard-negative partners can add other signatures.
These are mathematical graph arrangements, not semantic object categories.
Foreground brightness remains a possible synthetic-domain shortcut. Raster
connectivity QA checks a fixed sample rather than proving every possible raster.

## Inference interface

`RelationalEncoder()(rgb)` accepts float32 RGB BCHW in [0,1], dimensions at least
32 pixels. It reflect-pads right and bottom to a multiple of 8 internally and
crops spatial outputs back. It does not rotate or flip inputs. The principal
output is unit-normalized `embedding[B,64]`, using a foreground-weighted whole
image region plus four quadrants. `dense[B,32,ceil(H/2),ceil(W/2)]` is a normalized
stride-2 local/context representation; `raw_dense` precedes normalization.
Stride-2 centers correspond to native pixel indices (2i+0.5, 2j+0.5).

The fine branch remains stride 1, with pixel-unshuffle transitions preserving
phase before learned compression. Channel normalization operates per spatial
location. The longest convolutional receptive field is 130 input pixels for
the direct RGB path, or 138 including local-contrast preprocessing. The pooled
embedding then aggregates over the entire input. Pixel-unshuffle itself is
invertible; the learned encoder and final pooled descriptor are not lossless.
Whole-crop and resized context recipes belong to the separate real-data matcher.

## Loss and ablation

All runs use cross-instance supervised contrast over structural signatures,
foreground BCE (0.25), per-image fine-detail reconstruction (0.25) and a variance
floor (0.2). The primary additionally uses adjacency BCE (0.35) and node-degree
cross-entropy (0.15), sampled from the actual shared dense feature maps. The
ablation removes exactly those two relation terms. It keeps the same model,
initial state, training stream, optimizer and 4096-step budget. Positive views
have independent materials and deformations, so no cross-instance same-pixel
correspondence labels are invented.

AdamW uses initial learning rate 0.001, cosine decay to 0.00003, weight decay
0.0001 and gradient clipping at 5. Both runs use seed 1615, stream seed 1627,
four CPU threads, deterministic float32 CUDA, no AMP and no TF32. Final-budget
checkpoints are the declared scientific models; intermediate checkpoints are
weight snapshots, not candidates for selection using TEST results. They do not
contain AdamW optimizer state; exact training reproduction starts from the
declared random seed and consumes the full stream again.

Each successful optimizer batch records structural codes, both instance seeds,
and SHA-256 hashes of actual RGB and raster masks. Source snapshots and hashes
are admitted before the first batch. Smoke runs are separate and excluded from
the full-run budget. `synthetic_diagnosis_plan.json` specifies heldout graph
retrieval and nuisance-matched negative checks; these are not real-image results.
