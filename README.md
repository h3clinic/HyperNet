# HyperNet

**Experimental category recognition from one reference photograph, with no pretrained weights.**

HyperNet asks whether a small CNN trained from patches and synthetic variations of one image can recognize another example of the same object type in a different photograph. The v0.8 experiment combines an existing one-reference CNN with partial part assignment, support-only Gaussian variation statistics, foreground heuristics and structural counterfactual scores.

This is a research prototype. It does not claim human-like reasoning, state-of-the-art recognition, a new optimal-transport theorem or superiority over pretrained detectors. The fresh-photo evaluation is undergoing audit; audited results will be added separately.

## What is included

- The frozen v0.8 Python package, including earlier experiment components needed by its imports.
- Portable tests for exact partial assignment, support statistics, region scoring and fresh-cohort separation using synthetic fixtures.
- A command-line example that trains from your reference image and scores your query image.
- Sixteen primary-source research entries and a synthesis explaining the adopted mathematics and its limitations.

Photographs, pretrained checkpoints, learned per-reference checkpoints and machine-specific experimental evidence are not distributed in this source snapshot.

## Install

Python 3.11 or later is required. CPU execution is supported. Create and activate a virtual environment, then install the package:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[neural,data,dev]"
python -m pytest -q
```

The `neural` extra installs PyTorch and torchvision. Merely installing these libraries does not load pretrained weights. The v0.8 example uses only a freshly initialized CNN and fixed mathematical image descriptors. For hardware-specific PyTorch installation, follow the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

## Score a pair of your own photographs

Choose a tightly cropped reference showing the intended object type. The query should be a separate photograph. Selecting the support crop is part of the reference specification; automatic support cropping is not claimed.

```bash
python examples/match_one_image.py --reference path/to/reference.jpg --query path/to/query.jpg --output match.json
```

The default run trains one 40,976-parameter CNN for 120 steps from the reference alone, stores a reference-bound checkpoint under `checkpoints/`, then evaluates the full v0.8 scoring method. Repeating the command reuses a matching checkpoint. You can select a fixed-feature control without CNN training:

```bash
python examples/match_one_image.py --reference path/to/reference.jpg --query path/to/query.jpg --model fixed_v08 --output fixed_match.json
```

The output includes a raw score, selected candidate box and provenance. **The score is not a probability.** Counterfactual scores may be negative. No universal cutoff is provided: estimate a decision threshold on separate development photographs, freeze it, and only then evaluate held-out photographs. A high score on one pair does not establish accuracy.

Other available examples are `v07_cnn`, `v08_assignment`, `v08_foreground`, `v08_tolerance`, `v08_counterfactual` and `fixed_v06`. They isolate scoring changes while sharing a descriptor family.

## The data constraint

For each learned reference model:

- Exactly one support image supplies training pixels.
- All Gaussian fields, resampling, detail variations and crops derive from that support; they are synthetic variants, not additional natural category observations.
- No pretrained visual or language weights, learned teacher, query labels or query-time weight updates are used.
- The support model is newly initialized for its reference; training does not transfer weights between references.

Development images used to design a method or select thresholds are a separate research budget. “One image per reference model” does not mean the whole research program examined only one photograph. Natural category recognition is substantially harder than recognizing transformations of the original image.

The package preserves optional historical pretrained comparison adapters. Those are outside the v0.8 scratch-only experiment. OWL-ViT, when reported as a separate comparator, uses external pretraining and must have its training budget stated separately. No OWL-ViT model or pretrained weights are included here. Use `examples/match_one_image.py` for the explicit scratch-only path.

## What v0.8 changes

The reference and each fixed candidate window produce fourteen descriptors: whole region, 2×2 grid and 3×3 grid. One-to-one partial assignment prevents several support parts from claiming the same query tile. Zero-gain dummy choices permit missing parts. A post-solve weighted coverage gate rejects weak support explanations; it is not a coverage-constrained optimization.

Sixteen synthetic variants estimate a strongly regularized diagonal tolerance metric. Half ordinary cosine and half precision-weighted cosine form each pair similarity. This is not a Mahalanobis likelihood or an estimate of semantic class uncertainty. GrabCut and grayscale contrast provide bounded support-part weights, with an explicit all-foreground fallback. Four fixed part shuffles supply a per-region counterfactual margin; that margin is not a statistical p-value.

The fixed coordinate grid does not represent object anatomy. Background matches, imperfect foreground masks, pose changes, unseen within-category variation and proposal coverage remain material limitations.

## Research and originality

The potential contribution is a tested integration under an unusually restrictive data budget. Gaussian augmentation, covariance weighting, graph cuts, partial assignment and optimal transport are established ingredients. This repository does not establish broad algorithmic novelty.

- [Mathematical synthesis](research/research_synthesis.html) — open locally in a browser after cloning.
- [Geometry and partial-transport sources](research/geometry_primary_sources.json).
- [Variation, invariance and human-vision sources](research/variation_primary_sources.json).
- [Distortable Canvas](https://www.nature.com/articles/s44387-025-00017-7), [optimal partial graph matching](https://arxiv.org/abs/2410.16718), and [GrabCut](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf) are relevant prior work, not implementations reproduced in full here.

No published paper's benchmark score is presented as a direct comparison to this experiment. Such a comparison requires the same photographs, task, threshold protocol and metric, with pretraining budgets disclosed.

## Reproducibility and publication status

[Source hashes](publication/source_snapshot.json) identify every exported package file and its equality to the experiment snapshot. Package files are preserved byte for byte; only publication documentation, packaging and the portable example are added. The historical internal `hypernet.__version__` string remains unchanged to preserve that snapshot; package distribution metadata identifies this release as 0.8.0.

The portable test suite validates implementation properties. It does not validate recognition accuracy. Local cohort-specific experiment drivers are not portable artifacts and are excluded from this initial release. Sanitized audited experimental results and reproducibility instructions will be added when ready.

## License status

The source is publicly viewable. No open-source license has been selected or granted by this repository. Third-party dependencies and referenced publications retain their respective licenses and rights.
