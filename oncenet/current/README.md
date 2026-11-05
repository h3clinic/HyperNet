# Current OnceNet implementation

This snapshot separates the frozen Gravity Spy experiment from the Bering acoustic search and later branch development. See the [project overview](../../README.md) for results and limitations.

| Path | Contents |
|---|---|
| `gravity_inference/` | Selected 42,480-parameter encoder, fitted matchers, calibration and strict-FP32 deployment runtime |
| `repro_tools/` | Portable training, generator export and query wrappers |
| `workspace/outputs/hypernet_v46/spectrogram/` | Mathematical generator, original training recipe and three trained seeds |
| `workspace/outputs/oncenet_factorial_20261105/` | Frozen 2x2 protocol, scores, results and independent audits |
| `bering_inference/` | Separate acoustic deployment; its encoder used 77,000 unlabeled real acoustic windows |
| `branch_snapshot/whale-oneshot_e647212/` | Later branch source, separate from the frozen audited runtime |
| `source_manifest.json` | Published file hashes and original source bindings |

## Mathematical training

Install the dependencies from the repository root, then run:

```bash
python oncenet/current/repro_tools/train_gravity.py --workspace oncenet/current/workspace --smoke
python oncenet/current/repro_tools/export_generator.py --workspace oncenet/current/workspace --export --count 5 --output runs/shape_examples
```

The smoke check uses synthetic inputs and zero optimizer updates. To reproduce training in a new directory:

```bash
python oncenet/current/repro_tools/train_gravity.py --workspace oncenet/current/workspace --train --seed 45029 --device cpu --output runs/seed_45029
```

Seeds 45011 and 45047 are also available. The launcher retains the original 800-step recipe and verifies its source bindings. Generated illustrations and new training runs are not new benchmark results.

## Gravity Spy deployment

The code and model payload are included; 48 original public fixture PNGs are not. Restore them before verification:

1. Read `gravity_inference/events.json`. Each event's `windows` entries give `source_url`, relative `path` and `sha256` for four durations.
2. Download each original PNG to that path under `gravity_inference/`, preserving its bytes, and verify its SHA-256. Do not resize or recolor the source.
3. Run the payload verification below. Its default mode replays saved descriptors without neural inference.

```bash
python oncenet/current/repro_tools/predict_gravity.py --bundle oncenet/current/gravity_inference
```

To score a new event using the four already-enrolled target references:

```bash
python oncenet/current/repro_tools/predict_gravity.py --bundle oncenet/current/gravity_inference --query --d05 event_05.png --d1 event_1.png --d2 event_2.png --d4 event_4.png
```

The four inputs must be distinct authentic duration renders of the same event, in the verified 800x600 Gravity Spy format. No encoder updates, target refits or threshold changes occur. The deployment preserves primary alpha=0.001 and historical secondary alpha=0.0005; the factorial study separately evaluated secondary alpha=0.01.

## Reproduction boundaries

Historical scripts retain their original absolute paths and frozen hashes. Use the portable wrappers rather than changing those bindings. The saved test cohorts have already been evaluated; they are not untouched validation data for further development. Exact arithmetic can vary across runtimes, so retain the recorded environment when comparing scores.

Bering candidates remain similarity matches requiring source identification. The later whale branch snapshot does not establish new deployment results. Source images, audio and third-party dependencies retain their original terms; no repository-wide open-source license has been granted.
