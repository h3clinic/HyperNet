# GravitySpy: six frozen one-reference pattern-search arms

**The final CNN did not beat the cheap raw-image control.** At the declared CALIB operating point, the final model's mean per-target recall is 3.64%, versus 19.70% for centered 48Ã—48 grayscale correlation. The final model also has a higher mean per-target false-alarm rate. This result does not support a published-model win or useful high-recall rare-pattern search.

| Arm | CALIB-selected method | Recall | Mean FP/10k | Mean AUROC |
|---|---|---:|---:|---:|
| primary_final | learned:translation_006 | 3.64% | 18.89 | 0.7896 |
| random | learned:translation_012 | 2.70% | 9.74 | 0.8147 |
| scale_065536 | learned:translation_012 | 4.35% | 11.19 | 0.7965 |
| scale_262144 | learned:translation_006 | 4.50% | 11.63 | 0.8006 |
| analytic | analytic:translation_012 | 0.30% | 12.92 | 0.7106 |
| raw_pixels48 | raw_pixels48 | 19.70% | 11.37 | 0.7975 |

The table uses the predeclared 10-false-alarms-per-10,000-CALIB-negatives operating point. Each target keeps its own fixed threshold on TEST; achieved TEST false-alarm rates need not equal the calibration budget. `Recall` is a macro average of 19 per-target recalls, not classification accuracy or a percentage of all images correctly classified. `Mean FP/10k` averages normalized per-target negative-event rates. No target has 10,000 observed negatives in this test, and shared query images are not 19 independent trials.

When all 19 targets are screened together, count an event once if **any wrong target** exceeds its threshold, even if the correct target also alarms. The final model produces 134 such events out of 4,044, or 331.36 per 10,000 screened events; raw48 produces 81/4,044, or 200.30 per 10,000. Those union rates are different quantities from the table's mean per-target rates. Full per-target, group, union and secondary zero-CALIB-error results are retained in `metrics.json`.

## Data and selection budget

- One reference for each of **19 available target morphologies**, plus **1,579 labeled CALIB events** used for recipe selection and target thresholds. No real-image gradient updates occur. Twenty raw references were acquired; one exactly constant reference made that target unavailable, leaving 19. Twenty-nine exactly constant CALIB plots were excluded, without replacement.
- **4,044 fresh TEST events in 3,954 time groups**. The eligible cohort contains 3,736 events of enrolled targets and 308 other/background events. Prior exposed events and their whole time groups were excluded before v15 TEST acquisition/scoring. Of 4,119 remaining candidates, 75 exactly constant RGB plots were excluded by the fixed blind source-quality rule, without replacements.
- Development is not entirely fresh: **3 references and 567 CALIB events** belong to groups used in the older benchmark history. This exposure is disclosed and kept separate from the fresh TEST claim.
- One official 2-second spectrogram per event; the fixed plot-only crop is `[105,66,671,532]` from the publisher's 800Ã—600 image. Axes, titles and color bars are removed; native crop pixels enter overlapping 64px CNN tiles without resizing. Matching then samples descriptors at **stride4**, so not every pixel descriptor is retained. The scorer searches translations, not flips or rotations of the time/frequency axes.
- The final million-family checkpoint is the predeclared primary. Random, 65k and 262k checkpoints are controls from the same initialization/training trajectory. Each checkpoint chooses one global recipe on CALIB only; the analytic control selects its own recipe by the same rule. Raw48 has no recipe search. TEST does not select thresholds, recipes or checkpoints.

The primary failure is preserved. The numerical audits check all recipes/targets/operating points against saved score logs, source/checkpoint/calibration bindings, event IDs, labels, groups and sklearn ranking metrics. A separate replay uses exactly the two predeclared valid TEST events, all four checkpoints, every frozen method and all target outputs. It is an implementation replay, not another independent performance sample.

## Uncertainty

The predeclared bootstrap resamples whole time groups jointly across every arm, target and operating point, with 5,000 valid draws and seed1515. Thresholds remain fixed. It reports method intervals and paired final-minus-control contrasts. Draws lacking a positive or negative for any fixed target are rejected and counted; this conditions the intervals on target presence. The rarest target has only four positive events. Distinct time groups do not guarantee physical independence, and the intervals are conditional on these references, checkpoints, recipes, CALIB data and targets.

Final primary paired-group bootstrap 95% intervals: macro recall **2.77%–4.61%**; mean per-target FP/10k **15.73–22.16**. These are conditional benchmark intervals, not deployment guarantees.

The paired final-minus-random recall interval is **-0.23 to 2.15 percentage points**. It includes zero, so this benchmark does not demonstrate a recall gain over random initialization. Final-minus-raw48 recall is **-16.06 percentage points**, with a 95% interval of **-17.95 to -14.11 points**.

Bootstrap plan/results and their exact original hashes are included. The primary/random TEST audits used an archived exact `audit_gravity_scores.py` source; the remaining audits use an equivalent vectorized group-count implementation, with its 1,000-case comparison record included. Historical CALIB audits may cite an even earlier source hash because TEST-binding checks were added afterward. Those historical fields remain unchanged; the current source snapshot is not falsely presented as every earlier byte-identical audit implementation.

## Pixel-free public artifacts

`scores.npz` contains six frozen reported arms Ã— 4,044 TEST events Ã— 19 targets, the matching 1,579-event CALIB score tensors, IDs, labels, time groups, pixel SHA-256 strings, reference IDs, and the two fixed threshold sets. Arrays contain no images, feature vectors, masks or executable objects. Open with `allow_pickle=False`.

`metrics.json` carries each selected arm's audited metrics and exact checkpoint/result/log bindings. `provenance/` contains sanitized protocol, results, numerical/data/replay audits, quality/exposure records, bootstrap evidence and the objective-scope review. Large per-recipe raw score logs and all real-image media are excluded from Git. The compact arrays preserve every score needed for the six reported arms at both operating points.

Download the separate `hypernet_v15_gravity_score_logs.zip` asset from the [v0.15.0 release](https://github.com/h3clinic/HyperNet/releases/tag/v0.15.0). It contains the **eight original full CALIB/TEST score logs**, including every diagnostic recipe, plus an internal relative-path/hash/length manifest. Every ZIP entry is verified against its original. `full_score_logs_release_manifest.json` records the archive SHA-256 and size; status/upload fields describe historical preparation state, not current release availability. The archive is outside Git and contains no images, masks, feature vectors, or absolute-path metadata.

`../public_metadata_manifest.json` maps each original relative artifact and original SHA-256 to the public copy and its SHA-256. Local prefixes become `${V15_ROOT}`, `${PUBLIC_REPO}` or `${WORKSPACE_ROOT}`. Embedded original hash references are preserved. Cleaned historical protocols are not executable admissions on another machine; original frozen artifacts were not changed.

From the experiment directory:

```powershell
python scripts/recompute_public_gravity_metrics.py
python scripts/recompute_public_gravity_metrics.py --bootstrap
```

The first command recomputes all six arms' thresholds, target counts/ranking metrics, negative-group counts, macros and wrong-target event/group unions without reading any images. The optional second command also reproduces the paired-bootstrap intervals from compact arrays; it can take longer. Original benchmark scripts and bootstrap tests are included byte-for-byte, together with source snapshot manifests.

## Source and interpretation

The source is [GravitySpy v1.1 on Zenodo](https://zenodo.org/records/1486046), with the curated labels and official Panoptes image host. Attribution/license metadata is retained; no images are redistributed. To perform a new image-level rerun, acquire the official metadata and one 2-second image per declared event, reproduce the fixed crop and quality rule, and rebuild local source/roster/checkpoint/calibration/protocol admissions. The original prior-exposure audit additionally depends on the previous benchmark history. The compact roster, preserved acquisition hashes and exposure records identify the measured cohort, but path relocation and regenerated gates are necessary; this is not a one-command portable replay of the original acquisition environment. A newly selected cohort is a new experiment and needs its own untouched TEST set.

The [objective-scope review](provenance/experiments/v15_objective_scope_review.json) distinguishes the tasks: training pairs are two views of one exact canonical mathematical program, whereas morphology retrieval needs invariance between independently occurring same-type examples. Gaussian offsets are fixed within each training pair. The network improved measured synthetic correspondence and responded to selected one-pixel changes; that alone did not specify useful cross-instance invariance or anomaly decisions. Pixel unshuffle is invertible before learned compression, but the full CNN/matcher is not lossless and does not preserve every minute detail. The review's proposed changes are **untested future hypotheses**, not explanations proven by this experiment or changes applied after seeing TEST.
