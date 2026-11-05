# HyperNet

**One-reference recognition experiments with explicit training budgets. Version 0.14 still fails practical natural-photo detection: 1/96 positive TEST photos correctly localized and accepted, with 9/96 false alarms.** No externally pretrained weights or natural-photo weight gradients are used in the selected model.

The new proposal search covers 83/96 positive scenes, but the selected matcher chooses a correct top box in only 5/96 before thresholding. A control using 200 fixed windows also finds 1/96 accepted correct boxes, with 2/96 false alarms. This does not establish an end-to-end improvement. [All 14 methods, diagnostics, data budgets and audits](benchmarks/v0.14/README.md) retain the negative finding.

The selected model is a 755,577-parameter CNN trained from random weights on our own 2D drawings. Its **32 manually authored object programs explicitly include all 12 evaluated category archetypes**. It uses whitened global descriptors; local-part matching is a separate control. One boxed reference is the enrollment budget. **Another 96 real DEV photos selected the checkpoint, scoring rule and cutoffs from 63 candidates.** The 192 TEST photos were acquired only after freezing those choices. This is neither an unseen-category result nor one photo total development data.

After installation below, compare your own reference and query with the frozen model:

```bash
python examples/download_v14_cohort.py --models-only
python examples/parts_match.py --reference reference.jpg --query query.jpg --output publication_smoke/v14_manual.json
# Optional --box X1 Y1 X2 Y2 crops the reference subject.
python scripts/serve_parts_lab.py --enable-uploads --port 8795
# Open http://127.0.0.1:8795/parts_lab.html
```

Enrollment extracts features with zero weight updates. Uploads remain in localhost request memory; the CLI writes scores and geometry, not image pixels. The decision uses the published real-DEV cutoff. This interface is experimental and does not imply reliable recognition.

To replay published predictions with original source photos:

```bash
python examples/download_v14_cohort.py
python examples/replay_v14.py --limit-queries 2 --atol 0 --output publication_smoke/v14_sample
# Omit --limit-queries for all 24 predeclared replay queries.
# Add --full instead for all 288 queries and aggregate verification.
```

The downloader checks original file and decoded-pixel hashes and preserves individual licenses. No natural photographs or user uploads are published in GitHub or release assets. Strict replay is checked in the [recorded runtime](publication/runtime_v14.json); other PyTorch/OpenCV/platform versions can change floats or segmentation. Exact own-trained and random-control checkpoints are release assets; full score/proposal logs and code are public.

## Earlier experiments

Version **0.13.0** tests one automatically selected boxed reference per category on 288 new COCO query photos. The primary keeps its synthetic cutoff and rejects all TEST positives and negatives. A secondary cutoff calibrated using **96 additional real DEV photos** accepts 7/96 positive scenes and 2/96 negative scenes for the native matcher, but **none of its selected positive boxes is correct at IoU ≥0.5**. Presence balanced accuracy of 52.60% does not establish successful localization. [All four methods, both operating points and audits](benchmarks/v0.13/README.md) are published, including the failure.

The learned proposals cover 14/96 positive TEST scenes; the native matcher selects zero correct top boxes even without a cutoff. A separate fixed-grid diagnostic covers 64/96 but still selects zero correct native-match boxes. Proposal coverage and ranking both fail. References were annotation-selected, sometimes small or occluded, then normalized to 96px. This result is conditional on those twelve references, not proof that every clearer reference would fail.

The local app and CLI let you inspect the frozen model with your own images. **Enrollment extracts features; it does not perform SGD or update weights.** After installation below:

```bash
python examples/download_v13_cohort.py --checkpoint-only
python scripts/serve_photo_lab.py --enable-uploads --checkpoint checkpoints/detector_micro_aware.pt
# Open http://127.0.0.1:8794/photo_lab.html, choose two files, and draw a reference crop.
# Or use the whole reference from the command line:
python examples/photo_match.py --reference reference.jpg --query query.jpg --output results/photo_comparison.json
```

Uploads remain in localhost request memory. The CLI writes scores and geometry only. The default decision uses the unchanged synthetic cutoff; the optional `--operating-mode real_dev_calibrated` explicitly adds the 96-photo calibration budget. These tools are experimental and the benchmark does not establish reliable natural-photo recognition.

To reproduce published predictions, acquire the original licensed photos directly from their official source, then run a small replay or the image-free arithmetic verifier:

```bash
python examples/download_v13_cohort.py
python examples/replay_v13.py --output results/v13_replay
python publication/verify_v13_results.py
```

The downloader verifies original bytes, decoded pixels and dimensions, and saves source/license attributions locally. **No v13 natural photographs, natural-photo montages or user uploads are included in GitHub or release assets.** The 300 original photos are globally distinct: 12 references, 96 DEV queries and 192 TEST queries. Both splits share the same twelve category labels and enrolled references. Historical results below remain unchanged.

Version **0.12.0** improves the controlled 2D micro-pattern probe on an entirely fresh cohort: **253/256 positives localized and accepted (98.83%)**, compared with **13/256 (5.08%)** for the previous CNN on those same images. False positives are **13/256 (5.08%)** versus **11/256 (4.30%)**. The paired localized-recall gain is **93.75 percentage points**, 95% family-bootstrap interval **[89.45,97.66]**. Reducing only the fine RGB evidence by four times, while preserving native alignment and patch locations, lowers recall to **22.27%**, with **8.59%** false positives. See [all six methods, errors and audits](benchmarks/v0.12/README.md) and the [illustrated report](benchmarks/v0.12/report.html).

**This is a CNN-plus-analytic matcher on separated synthetic 2D shapes. It is not a new CNN-only accuracy result or evidence of natural-category recognition.** One reference describes enrollment. The reused CNN previously trained on 8,192 distinct procedural images; v12 makes no new optimizer updates and uses no natural training images. The primary method was selected on old development data and frozen before generating 128 fresh DEV and 128 fresh TEST composition families. Only DEV negatives calibrate acceptance thresholds. All six methods share the same CNN proposals. The original goal—recognizing any natural example of a type from one reference—remains unvalidated.

After installation below, reproduce a small inference sample with the exact project-trained checkpoint and synthetic images:

```bash
python examples/download_v12_assets.py
python examples/replay_v12.py --output results/replay_v12_smoke
python publication/verify_v12_results.py
# Optional full inference replay:
python examples/replay_v12.py --full --output results/replay_v12_full
```

The two [v0.12 release assets](https://github.com/h3clinic/HyperNet/releases/tag/v0.12.0) are a 3.60 MB synthetic-only checkpoint and a 5.42 MB procedural RGB archive. The downloader checks SHA-256 hashes before use. Code remains byte-identical to the frozen experiment; portable wrappers handle public paths and mark outputs as replication. No natural photographs, credentials or local user paths are included. Historical experiments below remain available, including unsuccessful approaches.

Version **0.11.0** is a **controlled 2D micro-pattern detection probe**, using a 96px reference and a 256px scene with six objects. The CNN trains on 2,048 procedural composition families and is evaluated on 128 DEV plus 128 TEST families with unseen combinations of familiar drawing rules. Two counterbalanced enrollments reuse identical query scenes with opposite labels. The micro-aware objective is compared against the same architecture, initialization and images without candidate micro-negative match supervision. **The added micro-pattern objective did not establish an improvement:** TEST localized recall was **4.30% versus 5.47%** for the family-only control, with **4.30% false positives in both arms**. The paired difference was −1.17 percentage points, 95% family-bootstrap interval [−3.52,+0.39]. All target boxes were available among the proposals, while pooled AUROC remained near 50%. See [all v11 results and audits](benchmarks/v0.11/README.md).

The v11 model takes one support image and its support box, plus the full query image. It receives no query box, mask or family label at inference. Training uses 8,192 distinct procedural RGB images, so “one reference” describes enrollment rather than total training data. Natural training images and externally pretrained weights are both zero. These controlled synthetic compositions are not thousands of natural semantic classes. The original goal remains recognizing any example of the same natural category. This probe deliberately rejects a near-identical broad shape with the wrong fine pattern, and does not replace or validate that semantic-category goal.

Version **0.10.0** adds a separate mathematical-surface curriculum and a native128 CNN with shape and detail heads. It learns from our own analytic geometry and renderer targets, with no natural photographs or externally pretrained weights. On different objects from eight **known, explicitly supervised synthetic families**, its one-support-per-family accuracy was **93.23%**, versus **93.45%** for the matched lowpass32 control. Higher resolution did not improve this primary metric. These synthetic scores are not natural-photo recognition results. Read [all v10 controls, protocol and reproducibility data](benchmarks/v0.10/README.md).

The v09 pathway adds synthetic-only pretraining before real-reference enrollment. The CNN starts at random weights and learns from generated primitive shapes, Gaussian variations and surface patterns. The primary control keeps that CNN frozen when it receives a real reference; optional controls adapt an independent copy using only that reference. No natural photographs train the shared backbone.

**v09 is an audited primitive-shape control. The primary reached only 50.00% balanced accuracy: 10/180 positives detected and 10/180 false alarms.** Its change from scratch was +1.39 percentage points, 95% paired category-bootstrap interval [−0.56,+3.33]; this does not establish improvement. Read [all v09 results and reproducibility data](benchmarks/v0.9/README.md). Neither experiment establishes human-like reasoning, a breakthrough in minute-detail recognition, or superiority over published detectors. Formula-driven pretraining and triplet learning are established ideas.

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

## Reproduce the 2D detector

Start from a clean checkout with no generated v11 data, freezes or checkpoints. The drivers preserve prior evidence and refuse to overwrite frozen runs.

```bash
python experiments/audit_2d_grammar.py
python scripts/prepare_2d_data.py --stage plan --expected-renderer-sha 1f9c35619a84ba65e761fcce4f38df9b5ee7a7220a192555b355236d7a4b77a1
python scripts/prepare_2d_data.py --stage render
python scripts/prepare_2d_data.py --stage audit
python scripts/audit_2d_detector.py --stage plan
python scripts/train_2d_detector.py --epochs 1 --threads 4
python scripts/audit_2d_detector.py --stage audit
python scripts/report_2d_detector.py
python scripts/serve_2d_gallery.py
```

Open `http://127.0.0.1:8792/two_dimensional_gallery.html` for the paginated gallery, or `/index.html` for the generated report. The gallery includes all 9,216 query rows and their references, masks and renderer annotations; completed DEV/TEST predictions have a separate magenta overlay. It loads only 24 rows per page. Pixel images and checkpoints are generated locally and are not distributed in the repository.

Both arms use the same 897,207-parameter CNN, 1,024 optimizer updates and 8,192 query-image exposures. Four exact quarter-turn views of each enrolled support produce 16,384 reference CNN views from 4,096 support enrollments. A native-resolution branch precedes downsampling, but the final region descriptor is still compressed to 128 dimensions. The controlled primary metric is TEST localized recall at IoU ≥0.5 with each arm's threshold fixed using DEV negatives at ≤5% false positives. A measured TEST false-positive rate can differ from 5%.

A separately declared classical diagnostic reuses the exact CNN proposal boxes and scores raw grayscale crops over a fixed alignment bank. It reached 7.03% localized recall, 4.30% false positives and 71.48% within-pair ranking, but only 51.79% pooled AUROC. This is a hybrid diagnostic, not a successful new CNN; its different rotation/alignment search prevents attributing the difference solely to lost CNN detail. Verify the pixel-free saved results with `python publication/verify_v11_results.py benchmarks/v0.11`.

The [research synthesis](research/two_dimensional_research.html) separates adopted Siamese matching and ranking ideas from the experimental curriculum. A new combination of established components is not evidence of a novel mathematical operation, human-like reasoning or natural-photo superiority.

## Generate and train on mathematical surfaces

```bash
python scripts/prepare_surface_data.py --stage all
python scripts/train_surface_encoder.py --steps 400 --threads 4
python scripts/audit_surface_encoder.py
python scripts/serve_surface_gallery.py
```

The gallery exposes all 1,920 generated RGB frames, masks, normals and depth. The dataset has 256 training object programs and 64 held-out programs from the same eight labeled families. Both training arms use the same 181,525-parameter network, initialization and batch plan, with 400 × 16 = 6,400 image exposures per arm. The native arm receives original 128px renders; the control receives32px-reduced images enlarged to128px. This is a controlled synthetic information-preservation study, not the completed one-reference natural-category model.

The native detail head's balanced surface matching was66.41% versus58.59% for lowpass, but the combined embedding scored57.81% versus60.94%, and primary family recognition did not improve. General roughness sensitivity can solve this surface task without identifying individual tiny grooves. The [full v10 benchmark](benchmarks/v0.10/README.md) retains every head, random controls and alternate-input evaluations.

These drivers write local data and checkpoints and refuse to overwrite frozen evidence. No generated pixels or trained weights are distributed. Four derived-feature archives permit arithmetic checking without training:

```bash
python publication/verify_v10_results.py benchmarks/v0.10
```

## Train the v09 procedural CNN

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
- [Surface mathematics review](research/surface_math.html), [primary sources](research/surface_primary_sources.json) and [exact implemented renderer formulas](research/mathematical_surface_notes.json). Current ingredients are separated from researched future alternatives.
- [v08 mathematical synthesis](research/research_synthesis.html), [geometry sources](research/geometry_primary_sources.json), and [variation/invariance sources](research/variation_primary_sources.json).

Open HTML reports locally after cloning. The potential contribution is an audited integration and evaluation under explicit budgets. Gaussian augmentation, formula-driven pretraining, metric learning, covariance weighting and partial assignment are established components. Published transfer-learning scores are not evidence for our exact one-reference photograph task.

## Reproducibility and publication status

[Source hashes](publication/source_snapshot.json) bind **all 77 package files** to the frozen v13 implementation. All 75 v12 package files are unchanged; two real-photo adapter/metric modules are added. Acquisition and evaluation drivers are exact copies; portable examples and publication documents are separate additions. The [v12 source snapshot](publication/source_snapshot_v0_12.json) and earlier snapshots preserve historical bindings. Distribution metadata identifies 0.13.0 while frozen internal metadata remains unchanged.

The [prior source manifest](publication/source_snapshot_v0_8_1.json), Git history and [v0.8.1 benchmark](benchmarks/v0.8.1/README.md) preserve the previous experiment. Its interpolation roundoff repair changed only input-boundary handling; it did not improve the model algorithm. All earlier audited score records remain available.

The v09 fresh-photo protocol has 720 reference/query pairs: 360 DEV and 360 TEST, each with 180 positives and 180 negatives, across six categories per split. Seven HyperNet controls and a separately pretrained OWL-ViT comparator are reported with the same metrics in the [audited v09 benchmark](benchmarks/v0.9/README.md). There are only six TEST category clusters, limiting uncertainty estimates. No fresh-score tuning is permitted.

Portable tests check implementation and provenance properties, not recognition accuracy. No natural photographs, caches, credentials or private machine paths are distributed. v12 release assets explicitly include our own synthetic-trained CNN and generated RGB cohort for exact inference replay; earlier releases retain their original pixel-free records. The [full v09 control results](benchmarks/v0.9/README.md) include 5,760 score rows, the reconstructable cohort, paired intervals, localization and independent arithmetic verification.

## License status

The source is publicly viewable. No open-source license has been selected or granted by this repository. Third-party dependencies and referenced publications retain their respective licenses and rights.
