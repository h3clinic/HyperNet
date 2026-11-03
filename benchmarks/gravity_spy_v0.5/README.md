# Gravity Spy one-reference glitch retrieval (v0.5): O3b confirmation

**Verdict: not confirmed.** On 2,369 O3b events that no HyperNet version had seen, the v0.4 synthetic-trained CNN does not beat the same architecture at random weights: 0.477 versus 0.495 mean AP over 17 glitch types, a difference of −0.018 with a paired class-bootstrap 95% interval of [−0.066, +0.029]. The v0.4 estimate on v1.1 TEST was +0.068 [−0.000, +0.132]. Under the rule frozen before scoring, the v0.4 gain did not transfer to O3b. This result is kept alongside v0.4 as a negative finding.

What did transfer: the objective effect. On the same synthetic data, the family objective beats the old two-views-of-one-picture objective by +0.142 [+0.081, +0.206] on O3b (v0.4: +0.157). The trained model also remains better on extended line and texture types (1080Lines 0.61 vs 0.51, 1400Ripples 0.43 vs 0.34, Scattered_Light 0.73 vs 0.65, Light_Modulation 0.38 vs 0.25) and worse on compact low-frequency ones (Air_Compressor 0.52 vs 0.77, Low_Frequency_Burst 0.65 vs 0.79, Power_Line 0.47 vs 0.60, Tomte 0.88 vs 0.96). The O3b cohort has many more of the second kind at equal weight, which is one reason the mean flipped.

## Verdict rule (frozen before TEST)

> Confirmed on O3b if best_trained minus best_random in TEST mean AP over evaluated_classes is > 0 and its paired class-bootstrap 95% interval (10,000 draws, seed 0) excludes zero. Same sign with an interval that includes zero is reported as "same direction, not confirmed". A difference <= 0 is reported as "not confirmed". The item bootstrap is a robustness check; the other contrasts are descriptive.

## What is fresh and what is not

Fresh: the haystack (O3b events from [Zenodo 5649212](https://zenodo.org/records/5649212), CC BY 4.0; the v1.1 training set predates O3 and shares no event id with these tables) and the 1% false-alarm cutoffs (calibrated on a disjoint O3b CALIB set).

Not fresh: the 100 references (v1.1 train), the 600 unlabeled standardization images (v1.1 train), and the label model. O3b labels are the Gravity Spy CNN's own predictions at confidence ≥ 0.95, and that CNN was trained on v1.1. The confidence filter keeps the labeler's most prototypical events (Paired_Doves 74 of 3,331 pass, 1400Ripples 70 of 393, Power_Line 253 of 771, Scattered_Light 96,385 of 117,903), which raises absolute AP for every arm. Confident label errors are not assumed random. The primary contrast compares two arms under the same labels.

Nothing was selected in v0.5. The six arms are the v0.4 DEV selections, scored through the v0.4 code path (`gravity_spy_v05.py` imports `gravity_spy_v04.Bank`). Checkpoint hashes were checked against v0.4's protocol, and the seed-1403 random encoder against the published random control. Before the real TEST ran, a dry run executed the whole pipeline with CALIB images standing in for TEST, so the arms' CALIB-level mean APs were seen before TEST; no protocol field was changed afterwards.

## Cohort

From 292,261 O3b rows (H1 and L1 tables, MD5-checked): confidence ≥ 0.95, one event per 4 s same-detector time group, stratified by label with caps of 150 (TEST) and 50 (CALIB) per class, TEST and CALIB from disjoint groups, seed 20261103. Events with ml_label outside the 20 target classes are "other" distractors: in TEST 104 Fast_Scattering, 40 Blip_Low_Frequency and 6 No_Glitch, counted as negatives for every reference. A pre-declared secondary analysis removes Fast_Scattering items for the Scattered_Light references and Blip_Low_Frequency items for the Blip references.

All 2,369 TEST and 785 CALIB events were fetched (12,616 PNGs, no failures), converted and scored; none was blank. Each event's four durations were converted by the exact v1.1 recipe (crop x 105:671, y 66:532; skimage gray weights; bilinear resize to 170×140; verified to reproduce the HDF5 arrays with zero error on 25 v1.1 events).

Evaluated classes: the 17 with at least 5 CALIB and 20 TEST positives. Wandering_Line (3 TEST, 0 CALIB), Chirp (5, 1) and Helix (8, 2) are reported descriptively only. No evaluated class has fewer than 30 TEST positives, so the O3b "rare" set is empty.

## Results (O3b TEST, 2,369 events, 17 classes)

| Arm (v0.4 selection) | Mean AP | Recall at 1% CALIB false alarms | False alarms per 10k | 17-way one-shot |
|---|---|---|---|---|
| Hybrid pixels + random1403, fused4 | **0.500** | **35.7%** | 97 | **46.0%** |
| Random seed 2026, four-layer mix, fused4 | 0.495 | 34.6% | 110 | 42.3% |
| Trained, family objective (`spectro_family_e8`, coarse 6×6, fused4) | 0.477 | 31.5% | 114 | 44.1% |
| Scattering, per-feature, fused4 | 0.428 | 31.0% | 110 | 33.6% |
| Raw pixels, 1.0 s | 0.426 | 30.0% | 108 | 45.9% |
| Trained, old objective (`spectro_instance_e12`), fused4 | 0.335 | 18.2% | 115 | 36.2% |

Contrasts, trained (family) minus:

| Comparison | Difference | 95% interval |
|---|---|---|
| Random seed 2026 (primary; class bootstrap, 10,000 draws, seed 0) | −0.018 | [−0.066, +0.029] |
| Random seed 2026 (item bootstrap, 2,000 draws, seed 1515) | −0.018 | [−0.029, −0.007] |
| Old objective, same data | +0.142 | [+0.081, +0.206] |
| Hybrid | −0.023 | [−0.107, +0.060] |
| Scattering | +0.048 | [−0.038, +0.144] |
| Raw pixels | +0.051 | [−0.041, +0.145] |

Absolute numbers are not comparable with v0.4's (0.688 vs 0.620): O3b TEST is cap-stratified (13 classes at 150 of 2,369, base rate about 0.06 versus 0.002–0.235 on v1.1 TEST), negatives are mostly other target classes at equal counts, and cutoffs come from O3b CALIB negatives. Every arm scores lower here than on v1.1 for those reasons.

Thin classes, descriptive AP with no cutoff (trained / random / raw pixels): Chirp (5 events) 0.59 / 0.30 / 0.56; Helix (8) 0.21 / 0.01 / 0.09; Wandering_Line (3) 0.04 / 0.09 / 0.00.

Secondary masked analysis: removing the 104 Fast_Scattering distractors for the Scattered_Light references moves random from 0.651 to 0.724 and trained from 0.730 to 0.732; removing the 40 Blip_Low_Frequency distractors for the Blip references changes neither arm by more than 0.003.

## Reading this

- v0.4's gain over random was the smallest of its contrasts (+0.068 with an interval touching zero) and rested on one TEST split used four times. The honest reading now is that the synthetic-trained features and random features are about equal on average, with the trained ones better on extended patterns and worse on compact low-frequency ones. The v0.4 README's "for the first time, training helps" claim is narrowed by this result to "the objective matters, and the trained features win on some glitch types".
- What held up across both cohorts: the family objective beats the old objective by a wide margin; the stem-to-coarse readout plus four-duration fusion is a strong label-free method; the pixels-plus-random hybrid is the best arm on O3b and was within noise of the best on v1.1.
- Why the mean flipped: O3b at equal weight has many more compact low-frequency types (Tomte, Air_Compressor, Low_Frequency_Burst, Power_Line, Koi_Fish, Blip) where the synthetic vocabulary is thinnest. A curriculum with better coverage of those shapes is the obvious next training change, and the v0.4 research report's coverage diagnostic is how to choose it without touching real labels.
- False alarms stay at about 100 per 10,000 at the 1% cutoff for every arm.

## Reproduce

```bash
python scripts/prepare_gravity_spy_o3.py --stage roster --tables <O3b tables> --output <cohort>
python scripts/prepare_gravity_spy_o3.py --stage fetch  --output <cohort>
python scripts/prepare_gravity_spy_o3.py --stage arrays --output <cohort>
python scripts/gravity_spy_v05.py --stage freeze --data <v1.1 files> --o3 <cohort> --runs <hypernet_v14/checkpoints> --trained <hypernet_gs_v04/checkpoints> --output <new folder>
python scripts/gravity_spy_v05.py --stage test   ...same arguments...
```

`protocol.json` (SHA-256 `76c7221b…`) pins the roster, both roster CSVs, both array files, both array manifests, the two scripts and the torch version by hash, and holds a hash of all CALIB scores; the test stage refuses to run if any differs. The TEST stage took 21 minutes on CPU. Images and arrays are not redistributed; the roster CSVs in `cohort/` name every event.
