# Gravity Spy one-reference glitch retrieval (v0.2)

**Reading HyperNet out at its early layers raises TEST mean AP from 0.219 (v0.1) to 0.559, but the winning readout uses random weights, not HyperNet's synthetic training.** Against the strongest v0.1 baseline, plain raw pixels (0.485), the gain is +0.074 with a class-bootstrap 95% interval of [−0.007, +0.157], so it's not statistically clear. On the five rare types the winner scores 0.462 versus 0.363 for raw pixels.

## What changed from v0.1

v0.2 updates no weights. The [v0.1 diagnosis](../gravity_spy_v0.1/README.md#diagnosis-where-the-information-is-lost-dev-only) found that the trained middle and deep blocks lose glitch information. So v0.2 reads features from earlier in the network.

**Candidates (18):** two weight sets × four layers (stem, fine, middle, coarse) × two poolings (6×6 grid, 1+2+3 pyramid), plus one mix of the stem, fine and middle grids per weight set.

**Weights:**
- trained: the frozen v0.14 model, run03 epoch 12
- random: its own seed-1403 random start

Both checkpoints are SHA-256 checked against the v0.14 release.

**New data budget:** 600 unlabeled train-split spectrograms (not references, seed 7) set each feature's mean and scale. Their labels aren't used.

Everything else matches v0.1: the same 100 references, splits and metrics, and 1% DEV false-alarm cutoffs.

**Selection:** two rules were declared before TEST. `protocol.json` (SHA-256 `79d2ac66…`) was written with DEV scores only. The test stage refuses to run if any DEV score has changed since the freeze.

| Rule | Selected | DEV mean AP |
|---|---|---|
| Best CNN readout | random weights, early-layer mix | 0.566 |
| Best trained readout | trained stem, 6×6 grid | 0.514 |

## TEST results (1,179 spectrograms, 20 target types)

| Method | Mean AP | Rare-type AP | Recall at 1% DEV false alarms | Rare recall | 20-way one-shot accuracy |
|---|---|---|---|---|---|
| **Best CNN readout** (random-weight early mix) | **0.559** | **0.462** | **47.8%** | **50.2%** | **56.8%** |
| Best trained readout (trained stem) | 0.508 | 0.352 | 41.0% | 39.7% | 51.8% |
| v0.1 raw pixels | 0.485 | 0.363 | 40.8% | 38.5% | 56.0% |
| v0.1 random-weight final embedding | 0.388 | 0.297 | 30.6% | 30.5% | 48.5% |
| Raw pixels with the same 600-image scaling | 0.379 | 0.206 | 28.4% | 22.0% | 38.4% |
| v0.1 HyperNet v0.14 | 0.219 | 0.059 | 11.5% | 7.8% | 24.6% |
| Chance | 0.049 | | | | 5.0% |

Paired class-bootstrap differences in mean AP, with 95% intervals:

| Comparison | Best CNN readout | Best trained readout |
|---|---|---|
| vs v0.1 HyperNet v0.14 | +0.340 [+0.251, +0.436] | +0.289 [+0.222, +0.366] |
| vs v0.1 random-weight embedding | +0.171 [+0.094, +0.252] | +0.120 [+0.031, +0.208] |
| vs v0.1 raw pixels | +0.074 [−0.007, +0.157] | +0.023 [−0.073, +0.114] |
| vs scaled raw pixels | +0.180 [+0.104, +0.260] | +0.130 [+0.056, +0.202] |

The DEV-selected readout was also the best of all 18 candidates on TEST. Non-selected candidates are in `results.json` for completeness only.

By type, the random-weight mix improves most on 1080Lines, 1400Ripples, Scattered_Light and Wandering_Line. It's worse than raw pixels on Chirp (0.51 vs 0.81), Power_Line (0.69 vs 0.92), Tomte and Blip.

## Reading this

- The early layers plus HyperNet's fixed input channels carry useful glitch structure. Reading them out instead of the final embedding fixes most of the v0.1 failure.
- **HyperNet's synthetic training isn't what helps.** The best readout uses random weights. The selected trained readout is the stem, which scored about the same as the random stem on DEV (0.514 vs 0.511). This result supports the architecture and input design, not the training curriculum.
- Neither readout clearly beats plain raw pixels yet. A claim of beating simple baselines needs a clear interval, or more types and references.
- Every method still gives roughly 90 false alarms per 10,000 negatives at the 1% cutoff. That's too many to find a one-in-10,000 event in a large archive.

## Reproduce

```bash
python scripts/gravity_spy_v02.py --stage select --data <Zenodo files> --benchmark benchmarks/gravity_spy_v0.1 \
    --runs <hypernet_v14/checkpoints> --output benchmarks/gravity_spy_v0.3
python scripts/gravity_spy_v02.py --stage test   --data <Zenodo files> --benchmark benchmarks/gravity_spy_v0.1 \
    --runs <hypernet_v14/checkpoints> --output benchmarks/gravity_spy_v0.3
```

A second TEST-stage run from the frozen protocol reproduced `results.json` exactly. The run took 147 s on CPU. `tests/test_gravity_spy.py` covers the shared metrics (11 tests).
