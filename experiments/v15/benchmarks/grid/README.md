# MVTec AD Grid: frozen one-reference pilot

**The trained model failed this real-image test.** Better synthetic correspondence did not produce useful Grid anomaly detection under the declared one-reference nearest-neighbor scorer.

| Method | Detected defects | False alarms on normals | Image AUROC | Image AP |
|---|---:|---:|---:|---:|
| Final mathematical CNN | 0/57 | 4/21 | 0.18797 | 0.56929 |
| Same initial random CNN | 2/57 | 2/21 | 0.43024 | 0.67440 |
| Fixed analytic details | 1/57 | 5/21 | 0.30994 | 0.61435 |

AP is prevalence-dependent: 57/78 TEST images are defective. It should not be interpreted as recognition accuracy. Final pixel AUROC/AP are 0.68100/0.01167 across all 81,788,928 original pixels; these pixel metrics do not rescue the failed image decisions. The independent audit reproduced image metrics and selected maps, not a second independent full-pixel ranking implementation.

One normal reference (`grid/train/good/002.png`) supplies the 4,096-token bank. **64 other normal images calibrate thresholds**, so the real data budget is 65 normal images before TEST, with zero natural-image gradient updates. Every official TEST image is included: 57 defective and 21 normal. Native 1,024×1,024 pixels enter 64px overlapping CNN tiles; descriptors are sampled at stride2. Image score is the mean of the largest `ceil(0.005*N)` patch residuals. The operating threshold is the maximum normal DEV image score plus the next representable float toward infinity.

Final false-positive rate is 4/21 = 19.05%, or 1,904.76 per 10,000. Its 95% exact-binomial reference interval is 544.64–4,190.66 per 10,000. This is a normalized estimate from **21 observed negatives**. It is not an empirical 10,000-negative archive search. Repeating this scorer or changing its threshold after seeing TEST would be a new, development-informed experiment; this release preserves the failure.

## Included evidence

- `scores.npz`: method names, 78 TEST IDs/labels/defect types/scores, 64 DEV IDs/scores, and three fixed thresholds. Arrays contain no pixels, feature vectors, masks, heatmaps, or executable objects; load with `allow_pickle=False`.
- `provenance/`: sanitized copies of original data manifests, protocol/admission, DEV/TEST result JSON and score logs, data/method audits, and independent review/replay evidence. Score logs reference map hashes, but maps themselves are deliberately absent.
- `../source_snapshot/`: byte-identical copies of the frozen benchmark source files. The matching runnable files are also present under the experiment's `hypernet/` and `scripts/` folders.
- `../public_metadata_manifest.json`: original artifact SHA-256, public-copy SHA-256, original relative artifact identity, and transformation status. Local absolute paths become the symbolic `${V15_ROOT}` prefix. No original local username/path is published.

An unchanged original SHA inside a sanitized protocol still refers to the **original artifact**, not to the transformed public file. Do not feed these cleaned historical protocols directly to the gate verifier: their paths/hash bindings are not a portable executable admission. Original frozen files were not edited.

The arithmetic audit checked 234 saved image scores from their full patch maps, all 192 DEV score rows, and image TP/FP/AUROC/AP. It then replayed the first sorted normal and first sorted defect image through all three methods, obtaining bit-exact maps, coordinates and image scores with unchanged weights/reference banks. The archived originals and six replay-map hashes are recorded in the independent audit.

From the experiment directory, image-score statistics can be recomputed without downloading any real images:

```powershell
python scripts/recompute_public_grid_metrics.py
```

## Acquire and perform a fresh rerun

The source is the [official MVTec AD download page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads), Grid category. Dataset terms are [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). No data license is replaced by the project's code distribution. The original archive had 160,763,852 bytes and local SHA-256 `d94a9a651cbfaee22c94d06cf8fa8204a590c872362de0489eb34f1645328de8`. No publisher-supplied checksum was available. The acquisition code contains the public mirror link used at the time; if the publisher changes it, record a new acquisition implementation/version rather than silently changing the frozen provenance.

Use a **new experiment copy/output tree** for a rerun. Keep the `hypernet/`, `scripts/`, checkpoint files and dependency list together, and ensure its `data/`, `results/` and runtime protocol output paths do not contain old run artifacts. Use the official archive/source and preserve native pixels. Before inference, copy the archived method plan to `experiments/grid_method_plan.json` in that new tree, recording it as the reused method specification. Then run the following sequential stages from that tree:

```powershell
python scripts/prepare_mvtec_grid.py --stage acquire
python scripts/prepare_mvtec_grid.py --stage inventory
python scripts/prepare_mvtec_grid.py --stage prepare
python scripts/prepare_mvtec_grid.py --stage audit
python scripts/evaluate_grid_v15.py --stage dev --checkpoint checkpoints/details_run01/families_1048576.pt --random-checkpoint checkpoints/details_run01/random_initial.pt --output results/grid_rerun_dev --device cuda --batch-size 32
python scripts/evaluate_grid_v15.py --stage freeze --dev-results results/grid_rerun_dev/grid_results.json
```

The freeze command prints the SHA-256 of the **newly generated local protocol**. Supply that actual digest, not the original or sanitized protocol digest, to the admission command:

```powershell
python scripts/prepare_mvtec_grid.py --stage admit --protocol experiments/grid_protocol_freeze.json --expected-protocol-sha NEW_LOCAL_PROTOCOL_SHA256
python scripts/evaluate_grid_v15.py --stage test --protocol experiments/grid_protocol_freeze.json --output results/grid_rerun_test
```

The placeholder must be replaced with the printed hash. New timestamps and absolute paths necessarily change admission/protocol hashes. For exact pixel replay, compare acquired raw/native-pixel hashes and reference/DEV/TEST identities to the archived manifests, and use the recorded library versions/device settings. Full saved-map and full-pixel metrics need the separately acquired dataset; the compact NPZ supports only image-level score arithmetic. CPU runs or changed GPU/library versions can differ numerically and are new execution records, not byte-identical replicas.

GravitySpy source is included in this package, including the v15 `gravity_spy.py` metadata adapter. Its separately audited TEST results are in the [GravitySpy supplement](../gravity_spy/README.md). That pipeline additionally requires official metadata, a fixed crop, quality admission, and whole-time-group prior-exposure exclusions. The original prior-exposure audit also depends on the earlier benchmark history. Do not represent the sanitized historical gates as a one-command portable GravitySpy rerun.
