# OnceNet

One-reference retrieval of rare visual and acoustic patterns. The repository retains its historical **HyperNet** name and Python package paths.

## Current models

- **Gravity Spy:** a 42,480-parameter CNN trained on project-generated mathematical 2D spectrogram patterns, with no borrowed pretrained weights. A frozen encoder combines full- and low-frequency views from four durations of one event. Each target is enrolled from one positive reference event; negative fitting and calibration data are additional, separate budgets.
- **Bering Sea:** a separate 1,566,756-parameter acoustic model, pretrained within the project on 77,000 unlabeled real acoustic windows. It searches for a reference pattern in recordings. Its retrieved candidates remain unconfirmed; this experiment is not synthetic-only pretraining.

The [merged whale implementation](whale/README.md) contains the separate acoustic search code, controls and candidate reports. Its archive searches are distinct from the Gravity Spy benchmark below.

[Current code, weights and reproduction tools](oncenet/current/) include the frozen implementations and their source manifests. Historical photo and synthetic-shape experiments remain available below.

## Latest controlled comparison

A frozen 2x2 test compared mathematical training versus saved random initializations, and cosine similarity versus negative-aware exemplar LDA (ELDA). All conditions used the same 2,400 fresh Gravity Spy events, three seeds and five separate one-reference episodes per class. ELDA used 512 fitting negatives per target; both matchers used the same separate 4,049-4,082 calibration negatives.

| Encoder / matcher | Standardized pAUC | Balanced accuracy | MCC | Recall | FA/10,000 |
|---|---:|---:|---:|---:|---:|
| Random / cosine | 0.5792 | 64.05% | 0.422 | 28.52% | 40.87 |
| Random / ELDA | 0.6365 | 67.03% | 0.495 | 34.32% | 26.16 |
| Trained / cosine | 0.6232 | 66.45% | 0.481 | 33.27% | 35.80 |
| Trained / ELDA | 0.6650 | 69.44% | 0.559 | 39.08% | 19.71 |

These are macro averages over targets, seeds and reference episodes. Standardized pAUC covers FPR <=0.001; operating metrics use thresholds calibrated at alpha=0.001. Training's average pAUC effect was +0.03626 (97.5% paired interval 0.02386-0.04981); ELDA's was +0.04954 (0.03616-0.06678). [Frozen results and independent audits](oncenet/current/workspace/outputs/oncenet_factorial_20261105/).

The gain is not uniform: training improved Scattered Light but reduced Low Frequency Burst recall under ELDA. Achieved false alarms exceeded the nominal calibration rate. Searching all four targets together produced a mean 73 alarms per 10,000 outside-target events, a different measure from the table's per-target average. DINO was compared on an earlier 195-event confirmation cohort, not this 2,400-event test.

## Install and reproduce

Use Python 3.11 or later. CPU execution is supported; use the recorded runtime for numerical comparisons.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[neural,data,dev]"
```

Check the mathematical training objective with one CPU forward pass and no optimizer updates:

```bash
python oncenet/current/repro_tools/train_gravity.py --workspace oncenet/current/workspace --smoke
```

Generate mathematical examples or explicitly start a new training reproduction:

```bash
python oncenet/current/repro_tools/export_generator.py --workspace oncenet/current/workspace --export --count 5 --output runs/shape_examples
python oncenet/current/repro_tools/train_gravity.py --workspace oncenet/current/workspace --train --seed 45029 --device cpu --output runs/seed_45029
```

Use new output directories. The wrappers verify source hashes and preserve frozen experiments. Historical runners retain original paths and are not all directly portable; use the [current package instructions](oncenet/current/). New training runs do not inherit the reported results.

## Data and scope

Source manifests record dataset locations, reference identities, splits and hashes. Obtain external images and audio under their respective terms. The 48 public Gravity Spy fixture images are excluded; restore their recorded URLs and hashes before deployment verification or query inference, following the [package instructions](oncenet/current/README.md). The Gravity Spy adapter accepts the verified 800x600 spectrogram renderer; it is not a general photograph detector. One-reference enrollment does not mean one image of total training or development data. Calibration thresholds are not guarantees of an archive's false-alarm rate, and similarity matches alone do not establish discoveries.

## Earlier experiments

- [Natural-photo detection, v0.13](benchmarks/v0.13/README.md).
- [Synthetic fine-detail matching, v0.12](benchmarks/v0.12/README.md) and [v0.11](benchmarks/v0.11/README.md).
- [Mathematical surfaces, v0.10](benchmarks/v0.10/README.md) and [procedural pretraining, v0.9](benchmarks/v0.9/README.md).
- [Archived workspace overview](README.legacy.md) and [procedural-training research](research/procedural_research.html).

## License

No repository-wide open-source license has been granted. Third-party dependencies, datasets and publications retain their respective licenses and rights.
