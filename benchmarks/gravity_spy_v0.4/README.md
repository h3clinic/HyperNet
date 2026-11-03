# Gravity Spy one-reference glitch retrieval (v0.4)

> **Update (v0.5):** this contrast did not hold on 2,369 unseen O3b events (trained 0.477 vs random 0.495, [−0.066, +0.029]); the objective effect did. See [v0.5](../gravity_spy_v0.5/README.md).

**For the first time, HyperNet's synthetic training helps on real data.** A CNN trained only on a synthetic time-frequency curriculum, where each type is a distribution and positives are independent members, reaches **0.688 TEST mean AP**. The same architecture at random weights reaches 0.586–0.620 across five seeds. The trained model beats four of the five random seeds with class-bootstrap intervals that exclude zero, and the fifth (the best seed, 0.620) by +0.068 [−0.000, +0.132], just short.

The objective is what matters. On the identical synthetic data, the old v0.14 loss (two augmented views of one picture) reaches only 0.531, below random. Family objective minus old objective: **+0.157 [+0.070, +0.253]**.

## What changed from v0.3

Three things, all decided on DEV before TEST was read. The [research report](research/research_report.md) explains why.

1. **New synthetic data** (`hypernet/spectro_curriculum.py`): 4,096 families of glitch-like patterns (blobs, chirps, bands, bursts, blip trains, arches, wandering lines, ripples, mirrored pairs) on 1/f noise, 140×170 like the Gravity Spy arrays. A family is a recipe with parameter spreads; every member is sampled independently.
2. **New objective** (`scripts/train_spectro.py --objective family`): cross-entropy over family ids + supervised contrastive on the global embedding + 0.25 × mask loss. Augmentations are time shift, time mask, frequency-band gain and gain only; no flips, rotations or colour changes. Control arm `--objective instance`: the unchanged run02 loss on the same data.
3. **Measurement fixes**: five random seeds instead of one; per-arm normalization (per-feature for CNN features, per-image z for raw pixels, both for scattering); 139 exactly constant spectrograms excluded from the haystack and two blank references replaced by a seeded rule; four-duration score fusion (mean of per-reference z-scores over the 0.5/1/2/4 s views) as a second scoring for every arm; an untrained wavelet scattering arm and a pixels-plus-random hybrid as controls.

Both training arms: 2,688 steps, 32 families × 2 members per step, 172,032 rendered images, same seed-1403 start as v0.14 (52 and 86 minutes on CPU, run side by side). Training records are in `training/`. No spectrogram, photograph or borrowed weight is used in training. Checkpoints are in `outputs/hypernet_gs_v04/checkpoints/` and aren't distributed.

## Selection (DEV, 1,182 items)

Rule: within each arm, the readout × scoring × epoch (or seed) with the highest DEV mean AP. `protocol.json` (SHA-256 `a35b955b…`) was written before TEST; the test stage checks every DEV score reproduces.

| Arm | Selected | DEV mean AP |
|---|---|---|
| Trained, family objective | epoch 8, coarse 6×6 grid, fused4 | 0.692 |
| Trained, old objective (control) | epoch 12, four-layer mix, fused4 | 0.542 |
| Random (best of 5 seeds) | seed 2026, four-layer mix, fused4 | 0.611 (seeds: 0.589–0.611) |
| Scattering | per-feature, fused4 | 0.599 |
| Hybrid pixels + random1403 | fused4 | 0.606 |
| Raw pixels | 1.0 s | 0.505 |

## Results (TEST, 1,159 items, 20 types, chance 0.049)

| Method | Mean AP | Rare-type AP | Recall at 1% DEV false alarms | 20-way one-shot |
|---|---|---|---|---|
| **Trained, family objective** | **0.688** | **0.619** | **62.7%** | **63.0%** |
| Random, best seed (2026) | 0.620 | 0.533 | 52.6% | 57.9% |
| Random, five seeds | 0.586–0.620 | | | |
| Hybrid pixels + random | 0.603 | 0.492 | 51.8% | 56.1% |
| Scattering (untrained) | 0.589 | 0.488 | 51.7% | 56.9% |
| Trained, old objective | 0.531 | 0.279 | 44.7% | 53.3% |
| Raw pixels | 0.502 | 0.398 | 42.5% | 56.8% |

Paired class-bootstrap differences in TEST mean AP, trained (family) minus:

| Comparison | Difference | 95% interval |
|---|---|---|
| Random seed 2026 (pre-declared primary) | +0.068 | [−0.000, +0.132] |
| Random seed 1403 (the v0.1–v0.3 control) | +0.072 | [−0.003, +0.138] |
| Random seed 7 | +0.101 | [+0.029, +0.170] |
| Random seed 11 | +0.071 | [+0.007, +0.129] |
| Random seed 23 | +0.075 | [+0.013, +0.131] |
| Old objective, same data | +0.157 | [+0.070, +0.253] |
| Hybrid pixels + random | +0.085 | [+0.003, +0.159] |
| Scattering | +0.099 | [+0.032, +0.173] |
| Raw pixels | +0.186 | [+0.082, +0.291] |

The selected epoch-8 model was also the best of all 50 scored readouts on TEST (epoch 12 is 0.686). Four-duration fusion alone lifts the random control from 0.576 (v0.3) to 0.620 (+0.044 [+0.006, +0.083]), so part of the headline gain over earlier versions is the scoring, not the training; the trained-vs-random contrasts above use fusion on both sides.

By type, training helps most on extended and structured patterns: Helix 0.86 vs 0.50 random, Chirp 0.88 vs 0.70, Power_Line 0.96 vs 0.72, 1400Ripples 0.69 vs 0.47, Paired_Doves 0.60 vs 0.41, Repeating_Blips 0.43 vs 0.21. It loses on Tomte (0.69 vs 0.94), Air_Compressor (0.57 vs 0.84) and Koi_Fish (0.71 vs 0.81): compact low-frequency shapes where the curriculum's vocabulary is thinnest.

## Reading this

- The research predicted the failure mode and the fix. Two views of one picture let the network solve training with a shortcut; independent members of a family force it to learn what defines a type. The same data with the old loss confirms it is the objective, not the data alone.
- The primary contrast against the single best random seed falls 0.0004 short of excluding zero. Against the control as it was defined in v0.1–v0.3 (seed 1403), and against three other seeds, the gain is clear. We report both rather than pick.
- This is the fourth use of the v1.1 TEST split under frozen protocols. The next confirmation should be on fresh O3 data (Zenodo 5649212), which v1.1 predates entirely.
- False alarms are unchanged at about 100 per 10,000 negatives at the 1% cutoff. Recall at that cutoff rose from 53% to 63%, but a one-in-10,000 search still needs the cohort-normalization and hubness methods in the research plan.
- Synthetic held-out retrieval (0.94 on 16-way) does not predict real performance by itself: the old-objective arm scored 100% on its own training pairs and 0.17 on held-out families.

## Reproduce

```bash
python scripts/train_spectro.py --objective family   --output <runs>/spectro_family
python scripts/train_spectro.py --objective instance --output <runs>/spectro_instance
python scripts/gravity_spy_v04.py --stage select --data <Zenodo files> --benchmark benchmarks/gravity_spy_v0.1 \
    --runs <hypernet_v14/checkpoints> --trained <runs> --output <new folder>
python scripts/gravity_spy_v04.py --stage test ...same arguments...
```

Needs `kymatio` for the scattering arm. DEV selection took 15 minutes and TEST 37 minutes on CPU. `dev_studies/` holds the DEV-only analyses that set the duration, fusion, normalization and blank rules; `research/` holds the literature study.
