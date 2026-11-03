# Gravity Spy one-reference glitch retrieval (v0.1)

**The frozen v0.14 CNN finds LIGO glitches better than chance but worse than every baseline, including the same network with random weights.** Mean TEST average precision over 20 glitch types: HyperNet 0.219, random weights 0.388, untrained gradient histogram 0.378, raw pixels 0.485, chance 0.049.

## Task

One labeled spectrogram from the train split is the reference. HyperNet ranks all 1,179 TEST and 1,200 DEV spectrograms by similarity. There's no localization: each item is one whole 1.0 s spectrogram. Five references per type (seed 20261102) are averaged. `None_of_the_Above` and `No_Glitch` stay in the haystack as distractors but aren't targets. Labels only pick references and score rankings. No weights are updated, and no spectrogram ever trained the CNN.

The splits are the publisher's train/validation/test labels, without regrouping events that overlap in time. On the same detector, 3 of 1,179 TEST items fall within 4 s of a reference, and 45 fall within 4 s of a DEV item. Near-duplicates could slightly inflate every method's score, which matters most for pixel matching. The v15 benchmark in `experiments/v15/` groups overlapping events before splitting.

Data: [Gravity Spy Training Set v1.1](https://zenodo.org/records/1486046), CC BY 4.0, `trainingsetv1d1.h5` plus `trainingset_v1d1_metadata.csv`, both MD5-checked. Images aren't redistributed here.

## Results (TEST)

| Method | Mean AP | Rare-type AP | Recall at 1% DEV false alarms | 20-way one-shot accuracy |
|---|---|---|---|---|
| HyperNet v0.14 (whitened global, the frozen selection) | 0.219 | 0.059 | 11.5% | 24.6% |
| Same CNN, published random weights | 0.388 | 0.297 | 30.6% | 48.5% |
| Untrained gradient histogram | 0.378 | 0.212 | 28.8% | 47.2% |
| Raw 48×48 pixels | **0.485** | **0.363** | **40.8%** | **56.0%** |
| Chance | 0.049 | | | 5.0% |

Rare types have fewer than 100 labeled examples in total: 1400Ripples, Air_Compressor, Chirp, Paired_Doves and Wandering_Line. The cutoffs give about 90 false alarms per 10,000 TEST negatives for every method.

Class-bootstrap 95% intervals for the HyperNet minus baseline difference in mean AP all exclude zero:

- random weights: −0.169 [−0.293, −0.058]
- gradient histogram: −0.159 [−0.237, −0.082]
- raw pixels: −0.266 [−0.403, −0.140]

HyperNet's own local-parts channels score higher than its selected global channel (0.306 centered parts), but they're still below every baseline.

HyperNet has the best AP only on Helix (0.327). It's weakest on Chirp, Tomte and Air_Compressor, where raw pixels reach 0.81–0.88.

## Reading this

- Gravity Spy spectrograms share fixed time and frequency axes, so raw pixels are a strong baseline here.
- The synthetic training made the CNN worse than its own random initialization on this domain. The synthetic-fit whitening isn't the main cause: the trained global channel without whitening scores about the same (0.200). The DEV-only diagnosis below locates the loss.
- Even the best method's ~90 false alarms per 10,000 would bury a one-in-10,000 event. A discovery search on large archives needs a cutoff with far fewer false alarms.

## Reproduce

```bash
python -m pip install h5py pandas
python scripts/gravity_spy_benchmark.py --data <folder with the two Zenodo files> \
    --checkpoints <folder with parts_trained_v14.pt and parts_random_v14.pt> --output benchmarks/gravity_spy_v0.2
python -m pytest -q tests/test_gravity_spy.py
```

The script checks the Zenodo MD5s and the published v0.14 checkpoint SHA-256s. It writes `protocol.json` before computing any score, and it won't overwrite an existing output folder. `per_reference.jsonl` holds every reference's metrics and its top-20 TEST ids. `scores.npz` holds every score. The run took 92 s on CPU (PyTorch 2.14.1).

## Diagnosis: where the information is lost (DEV only)

`scripts/gravity_spy_diagnose.py` reuses the frozen references but scores only the 1,200 DEV spectrograms; TEST was not touched. Features are standardized with 600 unlabeled train-split spectrograms. Results are in [`diagnosis/diagnosis.json`](diagnosis/diagnosis.json). Chance mean AP is 0.049; standardized raw pixels score 0.386.

**1. The loss is inside the trained convolution blocks, and it grows with depth.** Mean DEV AP, features pooled to a 6×6 grid:

| Stage | Trained (run03, epoch 12) | Its random start | Difference |
|---|---|---|---|
| Fixed input channels (not learned) | 0.462 | 0.462 | 0 |
| Stem | 0.514 | 0.511 | +0.004 |
| Fine block | 0.405 | 0.526 | −0.121 |
| Middle block | 0.340 | 0.539 | −0.199 |
| Coarse block | 0.257 | 0.515 | −0.258 |
| Final 128-D embedding | 0.197 | 0.432 | −0.235 |

**2. All three training curricula cause it, and most of the damage happens in the first epoch.** Final-embedding AP at epochs 0 / 1 / 12: run01 (shape families) 0.432 / 0.171 / 0.086; run02 (with anti-collapse loss) 0.432 / 0.218 / 0.163; run03 (object programs, the frozen model) 0.432 / 0.216 / 0.197. In run01 the fine block recovers to its random-start level by epoch 12 (0.535), but the deeper blocks don't.

**3. Losing position isn't the cause.** Moving a spectrogram 16 px up (higher frequency) leaves trained coarse features at 0.18 cosine similarity to the original, versus 0.48 for the random start. The trained features are *more* position-sensitive at large shifts, and more stable only at 2 px (0.92 vs 0.78). The earlier guess that training made the CNN ignore where energy sits is therefore not supported.

**Reading the diagnosis:** the most likely explanation, not yet tested directly, is that the synthetic curricula re-tune the middle and deep filters toward the drawings they train on, and those filters respond poorly to spectrogram texture: smooth blobs, streaks and noisy backgrounds. That would be a domain mismatch in the learned filters, not a lost-position problem and not a whitening problem. The best single probe was the *random* middle block on a 6×6 grid (0.539 DEV AP). It's a diagnostic found by looking at DEV, not a selected model; any method built on it needs its own frozen protocol and a TEST run.
