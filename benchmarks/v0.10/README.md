# v0.10: mathematical surface pilot

The CNN learned to recognize new objects from **eight known, supervised mathematical families**. Native128 achieved **93.23%** nearest-support family recognition; the matched lowpass32 control achieved **93.45%**. The higher-resolution input did not improve this primary metric.

This is a synthetic mechanism experiment. It is **not** a fresh-natural-photo result or an unseen-category few-shot benchmark. Its accuracy must not be compared numerically with the separate [v09 natural-photo benchmark](../v0.9/README.md).

## All controls

| Initialization and input | Representation | Known-family recognition | Same-object retrieval | Balanced surface matching |
|---|---|---:|---:|---:|
| Random, native128 | Combined | 50.74% | 15.62% | 61.72% |
| Random, lowpass32 | Combined | 47.62% | 16.15% | 57.03% |
| Trained, native128 | Combined | 93.23% | 20.31% | 57.81% |
| Trained, lowpass32 | Combined | 93.45% | 17.19% | 60.94% |
| Trained, native128 | Shape | 93.38% | 19.79% | 57.03% |
| Trained, lowpass32 | Shape | 93.38% | 18.23% | 60.94% |
| Trained, native128 | Detail | 72.92% | 17.71% | 66.41% |
| Trained, lowpass32 | Detail | 66.67% | 14.06% | 58.59% |

[Full results](aggregate_metrics.json) also retain every random head, supervised classifier scores, spectrum cosine, training history, runtime and alternate-input evaluations. The primary native-minus-lowpass difference is **−0.22 percentage points**. The detail head's +7.81-point secondary surface-matching difference does not demonstrate a general benefit: the combined head moves the other way. The balanced surface task can be solved by broad roughness sensitivity; it does not require matching individual tiny grooves.

The native-trained detail head scores 52.34% on lowpass input versus 66.41% on native input. That supports sensitivity to information removed by the transformation, conditional on this model, but also involves an input distribution shift. The lowpass-trained combined head scores 69.53% on native input. All these predeclared diagnostics are retained; none was selected as a new primary result.

## Budget and protocol

- Eight hand-designed families: superellipsoid, supertoroid, lobed vase, trefoil tube, corrugated torus, saddle shell, radial rosette and multipart spindle.
- **256 training object programs** and **64 held-out programs**, with no object-seed/base-program overlap. Each has two independently sampled views and displacement strengths 0, 0.03 and 0.07.
- **1,536 training frames + 384 held-out frames = 1,920** native 128×128 renders. No natural images, external meshes, generated-image services or pretrained weights.
- Both arms use the same **181,525-parameter** architecture, random initial tensors, batch plan, AdamW settings and **400 updates × 16 images = 6,400 exposures**. Each arm actually samples 1,515 distinct frames from the training pool. The lowpass arm adds area reduction to32×32 and bilinear enlargement to128×128; network and update count remain equal, not total preprocessing computation.
- Two normalized64-D heads form a128-D combined embedding with fixed squared weights0.75 shape/0.25 detail. Family cross-entropy, instance contrastive learning,64-bin spectrum prediction, mask, normal and depth losses are all supplied by our generator or analytic measurements.
- No natural reference is enrolled in this experiment. One **synthetic support per family** is enrolled for each evaluation episode after multi-image synthetic training.

The [training protocol](training_protocol.json), [exact batch plan](batch_plan.json), [dataset plan](dataset_plan.json) and [frame manifest](frame_manifest.json) are public. Absolute local paths are sanitized; original file hashes are retained in [export provenance](export_provenance.json).

Family recognition averages eight support rotations. Queries are different held-out objects and another view, at three strengths:1,344 decisions with reused queries/supports, uniform chance12.5%. Same-object retrieval has64 enrolled identities and192 cross-view queries, chance1.56%. Balanced clean/rough matching has128 triplets with both reference roles, chance50%. Ties receive fractional credit.

One initialization seed and eight known families do not support broad superiority claims. The decisions are dependent; no pair-level binomial confidence intervals are presented. Final400-step checkpoints, loss weights and head combination were fixed before scoring. No held-out model selection or early stopping occurred.

## Reproduce

Install the root package with the `neural,data,dev` extras. From the repository root, on a clean checkout without generated outputs:

```bash
python scripts/prepare_surface_data.py --stage all
python scripts/train_surface_encoder.py --steps 400 --threads 4
python scripts/audit_surface_encoder.py
python scripts/serve_surface_gallery.py
```

Open the printed localhost gallery URL to inspect all1,920 RGB frames, masks, normals and depth. Generation and training refuse to overwrite frozen experiment evidence. Generated data, checkpoints and local audit paths remain on your machine. The published audit records the original CPU/Python/PyTorch environment; other numerical environments may not reproduce bitwise equality.

To verify published ranking arithmetic immediately, without rendering or training:

```bash
python publication/verify_snapshot.py
python publication/verify_v10_results.py benchmarks/v0.10
```

The four `heldout_*.npz` files contain **derived descriptors, classifier logits and predicted spectrum vectors**, with rows in held-out manifest order. They contain no image pixels or model parameters. The public verifier checks all three representation heads and classifier accuracy for the four controls. Reconstructing spectrum targets, alternate-input inference and model outputs requires the generated data and trained checkpoints.

The [independent audit](inference_metric_audit.json) recomputed metrics and replayed2,304 forward frames, including all saved random/trained features and alternate-input evaluations. Saved features matched bit for bit; no training or gradient changes occurred during auditing. The [data audit](data_audit.json) checked5,760 generated files, paired nuisance controls and object-program separation. Audit records bind sources and outputs; they do not prove arbitrary hidden training history.

## Limits and originality

These are known synthetic categories with simple orthographic scenes, material colors and lighting. There are no natural backgrounds, semantic labels beyond the eight families, or tests of transfer to fresh photographs. Some surfaces are open or intersecting; the generator does not promise watertight physical solids. Pixel-center rendering without antialiasing can alias fine edges. Displacement is bounded in a parameter-domain Gaussian/groove field, not geodesic or physically learned category variation. The64px geometry head and global spectrum auxiliary target cannot guarantee preservation of every128px detail.

The implementation and matched experiment combine established mathematics and learning methods. Superquadrics, Gaussian basis functions, power spectra and contrastive learning are not new inventions. See the [primary-source review](../../research/surface_math.html) and [exact implemented formulas](../../research/mathematical_surface_notes.json). No claim of human-like reasoning, state of the art, or natural-image improvement follows from this pilot.
