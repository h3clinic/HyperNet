# Gravity Spy one-reference glitch retrieval (v0.3)

**Retraining with the early layers locked doesn't make the training useful.** In every matched comparison, a trained coarse block does no better than the same block at random weights (TEST differences −0.004 to −0.029 mean AP, all intervals crossing zero).

The pre-declared selected readout does beat both earlier results with clear intervals:
- the v0.2 winner: +0.013 [+0.003, +0.022]
- plain raw pixels: +0.086 [+0.008, +0.166]

That gain comes from adding a fourth layer to the readout, not from training it. The same four-layer mix with all-random weights scores 0.576 on TEST, against 0.572 for the selected model.

## What was retrained

`scripts/train_locked_early.py` runs the unmodified v0.14 trainers with chosen blocks set to `requires_grad=False` before the optimizer is built. Each run starts from the seed-1403 random state of the original runs, which the script checks by its recorded hash. After training it confirms that the locked blocks are byte-identical to that start and that the trained blocks changed.

| Run | Curriculum | Locked at random start | Trained |
|---|---|---|---|
| `math_lock3` | run02 math-shape families (`train_parts_v2.py`) | stem, fine, middle | coarse, local, embedding, heads |
| `math_lock2` | run02 math-shape families | stem, fine | middle and the rest |
| `objects_lock3` | run03 object programs (`train_semantic_parts.py`) | stem, fine, middle | coarse, local, embedding, heads |
| `objects_lock2` | run03 object programs | stem, fine | middle and the rest |

Each run is 12 epochs, 2,688 steps and 86,016 synthetic image exposures, the same budget as the v0.14 runs. No spectrogram trains any weight.

`train_parts_v2.py` hashes `experiments/run01_learning_diagnosis.json` into its admission record. That file isn't in the public checkout, so the wrapper hashes the `outputs/hypernet_v14` copy instead. That copy's hash matches the one run02 recorded, and each `locked.json` notes the redirect. It doesn't affect training.

Checkpoints are in `outputs/hypernet_gs_v03/checkpoints/` and aren't distributed.

## Protocol

v0.3 uses the same references, splits, metrics and 600-image unlabeled scaling as v0.2.

**Candidates:** 20 readouts that use at least one trained block (coarse grid and pyramid, global embedding, all-four-layer mix, plus middle readouts for the lock2 runs). The two all-random mixes are included as controls.

**Selection rule:** highest DEV mean AP among readouts that use a trained block. `protocol.json` (SHA-256 `a3745695…`) was written before TEST was scored. The test stage checks that every DEV score reproduces, including the v0.2 winner's.

This is the third use of TEST, after v0.1 and v0.2. Each use followed a frozen protocol, but repeated use still weakens it as a fully held-out test. A future method should be confirmed on fresh data, such as Gravity Spy's later observing runs.

## Results

| Method | DEV mean AP | TEST mean AP | Rare AP | Recall at 1% DEV false alarms | 20-way one-shot |
|---|---|---|---|---|---|
| **Selected: `objects_lock3`, four-layer mix** | 0.575 | **0.572** | **0.481** | **48.7%** | **58.2%** |
| Control: all-random four-layer mix | 0.582 | 0.576 | | | |
| v0.2 winner: all-random early mix | 0.566 | 0.559 | 0.462 | 47.8% | 56.8% |
| v0.1 raw pixels | | 0.485 | 0.363 | 40.8% | 56.0% |

Trained block vs the same block at random weights, four-layer mix, TEST mean AP:

| Run | Difference | 95% interval |
|---|---|---|
| `math_lock3` | −0.006 | [−0.023, +0.012] |
| `math_lock2` | −0.029 | [−0.062, +0.003] |
| `objects_lock3` | −0.004 | [−0.015, +0.007] |
| `objects_lock2` | −0.008 | [−0.035, +0.020] |

Single trained readouts are worse than the mixes. The trained global embedding is the worst of all (TEST 0.25–0.43), so the embedding head stays the weakest part of every trained model. The all-random four-layer mix was the best of all 22 readouts on DEV. It wasn't eligible under the selection rule because it uses no trained block.

## Reading this

- Across v0.1–v0.3, HyperNet's architecture and fixed input channels carry useful glitch structure. Neither synthetic curriculum (math shapes or object drawings) adds to it, with or without locking.
- The contrastive objectives train the network to match two renders of the same synthetic family. On this evidence, those features don't transfer to spectrograms.
- To make training help, the objective or the synthetic data probably has to change. Locking which layers learn isn't enough.
- About 90 false alarms per 10,000 at the 1% cutoff remains the limit for rare-event searches.

## Reproduce

```bash
python scripts/train_locked_early.py --curriculum object_programs --lock stem fine middle \
    --cache <hypernet_v14/data>/synthetic_parts_v14_semantic.npz --output <new folder>/objects_lock3
# likewise math_shapes with synthetic_parts_v14_moderate.npz, and --lock stem fine
python scripts/gravity_spy_v03.py --stage select --data <Zenodo files> --benchmark benchmarks/gravity_spy_v0.1 \
    --v02 benchmarks/gravity_spy_v0.2 --runs <hypernet_v14/checkpoints> --locked <new folder> --output <new output>
python scripts/gravity_spy_v03.py --stage test ...same arguments...
```
