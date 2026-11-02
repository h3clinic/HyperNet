# HyperNet

**Category recognition from one real reference, using either a reference-only CNN or a CNN we pretrain ourselves on procedural images. No borrowed pretrained weights are used in these HyperNet paths.**

Version **0.9.0** adds synthetic-only pretraining before reference enrollment. The CNN starts at random weights and learns from generated shapes, Gaussian variations and surface patterns. The primary control keeps that CNN frozen when it receives a real reference; optional controls adapt an independent copy using only that reference. No natural photographs train the shared backbone.

**v09 is an audited primitive-shape control, not the final richer shape curriculum. The primary reached only 50.00% balanced accuracy: 10/180 positives detected and 10/180 false alarms.** Its change from scratch was +1.39 percentage points, 95% paired category-bootstrap interval [−0.56,+3.33]; this does not establish improvement. Read [all v09 results and reproducibility data](benchmarks/v0.9/README.md). It does not establish human-like reasoning, a breakthrough in fine-detail recognition, or superiority over published detectors. Formula-driven pretraining and triplet learning are established ideas. More complex, named mathematical shapes are a separate development direction.

The previous audited result remains available: the full v0.8.1 method found **7/240 positive TEST pairs**, versus **20/240** for the previous CNN, with **1/240 versus 3/240 false alarms**. Its paired balanced-accuracy difference was **−2.29 percentage points**, 95% category-bootstrap interval **[−4.17, −0.42]**. See [all v0.8.1 ablations and data budgets](benchmarks/v0.8.1/README.md).

## Install

Use Python 3.11 or later. CPU execution is supported:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[neural,data,dev]"
python -m pytest -q
python publication/verify_snapshot.py
```

Installing PyTorch/torchvision does not download or load pretrained weights. For hardware-specific installation, follow the [official PyTorch instructions](https://pytorch.org/get-started/locally/). The examples below construct our CNN and never call a pretrained model loader.

## Train the procedural CNN

```bash
python examples/pretrain_shapes.py --mode detail_metric --output checkpoints/procedural_detail_metric.pt
python examples/pretrain_shapes.py --mode restoration --output checkpoints/procedural_restoration.pt
```

Each default run uses **2,000 updates × 16 triplets**, seed 73 and a 40,976-parameter CNN. That is **32,000 base-program draws + 32,000 mutant-program draws**, **96,000 generated RGB exposures**, and **192,000 corrupted training-view exposures** per arm. Unique image counts are not measured. Both objectives receive the same generated sequence and initial CNN weights. Each run saves its metadata beside the checkpoint and refuses to overwrite existing outputs.

The restoration control predicts a fixed 64-D HOG/gray/edge descriptor. The combined objective adds a **0.25-weight foreground mask/detail head** and a **0.2-weight matched-nuisance triplet loss**. Its auxiliary head is discarded for inference. A comparison of these two arms cannot identify the separate effects of the extra head and triplet term.

For a quick execution check, use `--steps 2 --triplets 2` and a different output path. That produces a small smoke-test model, not the evaluated 2,000-update model. [Published control metadata](publication/procedural_control.json) records the full experimental budget without distributing weights.

## Inspect every procedural training image

```bash
python examples/view_curriculum.py
```

Open the printed localhost URL. The browser exposes **all 2,000 batches**, including raw RGB, Gaussian targets, both actual corrupted CNN inputs, and known procedural masks. It regenerates the published seed 73, 16-triplet curriculum with exact corruption RNG replay; no sample filtering is used. This viewer is fixed to that published recipe and does not represent custom command-line training settings. Rendered PNGs quantize the float32 training inputs to 8-bit for display. The 32×32 images are enlarged with nearest-neighbor display, so they should not be mistaken for detailed natural photographs.

## Enroll a reference and score another photograph

Supply a tightly cropped reference showing the intended object type and a separate query image:

```bash
python examples/match_procedural.py --pretrained checkpoints/procedural_detail_metric.pt --reference path/to/reference.jpg --query path/to/query.jpg --output match.json
```

By default, the procedural CNN remains frozen. The reference supplies descriptors for matching, **not gradient updates**. To adapt a fresh copy with that one reference:

```bash
python examples/match_procedural.py --pretrained checkpoints/procedural_detail_metric.pt --reference path/to/reference.jpg --query path/to/query.jpg --adapt --output adapted_match.json
```

Adaptation uses the unchanged v07 recipe: 297 reference patches, 120 AdamW updates at learning rate 0.002, and Gaussian shape/pattern variations. Every reference starts from the same synthetic checkpoint; adapted weights do not pass between references. Query photographs never update the weights. The source checkpoint and tensor hashes are recorded.

The output contains a raw score, selected candidate box and provenance. **The score is not a probability.** No universal cutoff is supplied. Select thresholds on separate development photographs, freeze them, and then evaluate held-out photographs. Support-box selection and development calibration are additional supervision/research budgets and must be disclosed.

## Controls and older reference-only path

The v09 registry includes `scratch_reference`, `procedural_restoration_frozen`, `procedural_restoration_adapted`, `procedural_detail_frozen`, `procedural_detail_adapted`, `random_frozen`, and `fixed_shape`. They use the same 32px tiles, 64-D descriptors, candidate windows and spatial scorer. Frozen models reuse one synthetic backbone; scratch/adapted controls use separate reference models.

The earlier scratch-only example remains usable:

```bash
python examples/match_one_image.py --reference path/to/reference.jpg --query path/to/query.jpg --model v07_cnn --output scratch_match.json
```

Its one support image supplies all training pixels and it uses no pretraining. The default `v08_full` option retains the negative-result v08 scoring experiment for reproducibility; it is not recommended as a proven improvement.

Optional historical pretrained comparison adapters remain in the package. OWL-ViT is a separate comparator using external pretrained weights, outside the HyperNet constraint. No such weights are distributed here.

## Training scope and limitations

The new protocol is **synthetic-only pretraining plus one real reference**, not “one image total” and not “no pretraining.” Synthetic shapes introduce a hand-designed visual prior and additional optimization. Natural category variation, articulation, perspective, material, occlusion and background complexity remain unresolved. More Gaussian samples do not automatically supply missing category semantics.

The historical `external_pretraining: false` field means no externally supplied weights. v09 explicitly also reports `pretrained: true`, the procedural origin and synthetic exposure counts. Frozen reference enrollment uses zero real-image weight updates; adapted controls use one real training source. Calibration images and research development images are reported separately.

## Research and originality

- [Procedural-pretraining synthesis](research/procedural_research.html) and [six primary-source entries](research/procedural_primary_sources.json), including FractalDB, VisualAtom, OFDB and FDSML.
- [v08 mathematical synthesis](research/research_synthesis.html), [geometry sources](research/geometry_primary_sources.json), and [variation/invariance sources](research/variation_primary_sources.json).

Open HTML reports locally after cloning. The potential contribution is an audited integration and evaluation under explicit budgets. Gaussian augmentation, formula-driven pretraining, metric learning, covariance weighting and partial assignment are established components. Published transfer-learning scores are not evidence for our exact one-reference photograph task.

## Reproducibility and publication status

[Source hashes](publication/source_snapshot.json) bind **all 67 package files** byte for byte to the frozen v09 experiment. All 60 previous v0.8.1 package files remain unchanged. Packaging, portable examples and publication documents are separate additions. The historical internal version string is preserved; distribution metadata is 0.9.0.

The [prior source manifest](publication/source_snapshot_v0_8_1.json), Git history and [v0.8.1 benchmark](benchmarks/v0.8.1/README.md) preserve the previous experiment. Its interpolation roundoff repair changed only input-boundary handling; it did not improve the model algorithm. All earlier audited score records remain available.

The v09 fresh-photo protocol has 720 reference/query pairs: 360 DEV and 360 TEST, each with 180 positives and 180 negatives, across six categories per split. Seven HyperNet controls and a separately pretrained OWL-ViT comparator are reported with the same metrics in the [audited v09 benchmark](benchmarks/v0.9/README.md). There are only six TEST category clusters, limiting uncertainty estimates. No fresh-score tuning is permitted.

Portable tests check implementation and provenance properties, not recognition accuracy. No photographs, trained checkpoints, caches, credentials or machine-specific experiment paths are distributed. The [full v09 control results](benchmarks/v0.9/README.md) include 5,760 score rows, the reconstructable cohort, paired intervals, localization and independent arithmetic verification.

## License status

The source is publicly viewable. No open-source license has been selected or granted by this repository. Third-party dependencies and referenced publications retain their respective licenses and rights.
