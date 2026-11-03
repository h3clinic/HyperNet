# v0.4 DEV-only design studies

Run before any v0.4 protocol was written. Both scripts read only the frozen v0.1 references, the 1,200 DEV spectrograms and the 600 unlabeled train-split scaling images. No TEST row is read. They use the published random-weight v0.14 checkpoint (`random_v3_1403_statistics.pt`), the 6×6-grid readouts from v0.2/v0.3, and the v0.1 raw-pixel baseline.

## `dev_perclass_duration.py`: which duration, which classes

Mean DEV AP by spectrogram duration (px = raw pixels, cnn4 = random four-layer mix):

| Duration | px | cnn4 |
|---|---|---|
| 0.5 s | 0.499 | 0.597 |
| 1.0 s (used in v0.1–v0.3) | 0.493 | 0.582 |
| 2.0 s | 0.447 | 0.540 |
| 4.0 s | 0.382 | 0.457 |

Each class has its own best duration. Raw pixels beat the CNN on compact bright transients (Chirp 1.00 vs 0.87, Power_Line 0.94 vs 0.75, Tomte 0.92 vs 0.90 at 0.5 s); the CNN wins on extended line and texture patterns (1400Ripples 0.56 vs 0.17, Wandering_Line 0.41 vs 0.09, Scattered_Light 0.81 vs 0.48). Dropping the horizontal-flip reference view changes raw-pixel AP by at most −0.01 (flip helps slightly).

## `dev_fusion.py`: fusing durations and methods

Scores are z-normalized per reference using that reference's scores against the 600 unlabeled train images, then averaged.

| Method | DEV mean AP | Rare AP |
|---|---|---|
| cnn4, 1.0 s only | 0.582 | 0.531 |
| cnn4, 0.5 s only | 0.597 | 0.534 |
| **cnn4, all four durations fused** | **0.611** | **0.570** |
| px + cnn4, 0.5 + 1.0 s fused | 0.606 | 0.536 |
| px, all four durations fused | 0.489 | 0.395 |

Four-duration fusion is the best readout found on DEV. Adding raw pixels to the CNN does not help. Z-normalization alone leaves per-reference AP unchanged; it only matters when scores from different references or views share one cutoff.

These are exploratory DEV numbers used to write the v0.4 protocol. They are not results.

## `dev_readout_study.py`: per-class failures, normalization and blanks

Written by a research agent from the frozen v0.1–v0.3 artifacts plus DEV-only re-scoring; results in `dev_readout_results.json`.

- **Where raw pixels win:** compact, low-frequency, centred glitches (Blip, Tomte, Power_Line, Air_Compressor, Chirp, Low_Frequency_Burst). The loss under the CNN's 6×6 pooling tracks spatial resolution: raw pixels pooled to 6×6 fall to 0.353 DEV, and Tomte and Air_Compressor need the full 48×48.
- **The standardization anomaly:** per-pixel standard deviation over the 600 unlabeled images varies 15× between frequency rows (6.8 at the top, 44 at the bottom). Per-feature standardization amplifies the quiet high-frequency rows and drowns low-frequency classes: raw pixels 0.333 per-feature vs 0.493 per-image z on DEV. The same per-feature rule is still best for the random CNN features (0.539 vs 0.519). v0.4 therefore normalizes per arm.
- **Blank spectrograms:** 139 constant-floor images (113 Extremely_Loud). Two of the 100 v0.1 references were blank (Extremely_Loud ref 0, Paired_Doves ref 2). Under any CNN readout all blanks match each other at cosine 1.0. Excluding them moves the frozen v0.1 TEST mean AP by only +0.009 (raw) and +0.012 (random). v0.4 excludes blanks from the haystack and replaces blank references by a seeded rule.
- **Hybrids:** z-fusion of per-image-z raw pixels with the random early mix scores 0.597 DEV; PCA on 600 unlabeled images lifts raw pixels alone to 0.544; pooling the random middle block over time only gives 0.546. v0.4 carries the hybrid as a named control.
