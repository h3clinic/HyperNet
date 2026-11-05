"""Pure scoring core for the paired OnceNet encoder-by-matcher experiment.

No training, acquisition, label access, or evaluation runs on import. Historical
models and metadata are read-only. The caller owns the frozen data protocol,
reference episodes, calibration, score commitment, and subsequent metrics.

Input contract: inputs_dict[duration][view] is an existing float32 N x 128 x 128
energy image array; duration is d05/d1/d2/d4 and view is full/low. Equivalent
flat (duration, view) keys or ``full_d05`` keys are accepted, never inferred
from absent windows. Extraction returns duration -> float32 N x 4032 arrays.
Both matchers consume exactly the same float32 concatenation divided by two.
ELDA calculations and all final matching scores use float64.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
DURATIONS = ("d05", "d1", "d2", "d4")
VIEWS = ("full", "low")
SEEDS = (45011, 45029, 45047)
TARGETS = ("Low_Frequency_Burst", "Low_Frequency_Lines", "Scattered_Light", "Blip")
PARAMETERS = 42480
SINGLE_DIMENSION = 4032
JOINT_DIMENSION = 16128
REFERENCE_IDS = {
    "Low_Frequency_Burst": "632xGf8n9x",
    "Low_Frequency_Lines": "Vengh4mj2t",
    "Scattered_Light": "dac0D6ni5R",
    "Blip": "yxumvoPABn",
}


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registry(workspace: str | Path | None = None) -> list[dict]:
    """Verify and return the exact three saved trained/random checkpoint pairs."""
    root = Path(workspace or WORKSPACE).resolve()
    source = root / "outputs/hypernet_v46/spectrogram"
    models = _read(source / "checkpoints.json")
    protocol = _read(source / "protocol.json")
    plan = protocol["plan"]
    if not models["training_complete"] or tuple(plan["seeds"]) != SEEDS:
        raise ValueError("Expected all three completed historical seeds")
    if (plan["parameters"], plan["steps"], plan["batch"]) != (PARAMETERS, 800, 4):
        raise ValueError("Historical recipe changed")
    if plan["borrowed_neural_weights"] or plan["real_encoder_training_images"] != 0:
        raise ValueError("Expected mathematical-only encoder training")
    if models["protocol_sha256"] != sha(source / "protocol.json"):
        raise ValueError("Training registry/protocol mismatch")
    trained = {int(row["seed"]): row for row in models["models"]}
    initials = {int(row["seed"]): row for row in plan["initial_controls"]}
    if set(trained) != set(SEEDS) or set(initials) != set(SEEDS):
        raise ValueError("Missing or extra seed")
    result = []
    for seed in SEEDS:
        completed = _read(source / f"seed_{seed}/completed.json")
        if completed["step"] != 800 or completed["stage"] != "completed":
            raise ValueError("Incomplete historical training")
        if (completed["seed"] != seed or completed["total"] != 800
                or completed["protocol_sha256"] != models["protocol_sha256"]
                or not completed["no_checkpoint_selection"]):
            raise ValueError("Historical completion record does not match the frozen recipe")
        if completed["random_initialization_sha256"] != initials[seed]["sha256"]:
            raise ValueError("Random control is not the saved trained initialization")
        if completed["checkpoint_sha256"] != trained[seed]["sha256"]:
            raise ValueError("Completed checkpoint/registry mismatch")
        for arm, evidence, relative in (
            ("trained", trained[seed], f"outputs/hypernet_v46/spectrogram/seed_{seed}/spectrogram.pt"),
            ("random", initials[seed], f"outputs/hypernet_v45/multiseed/seed_{seed}/random.pt"),
        ):
            path = root / relative
            if sha(path) != evidence["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {path}")
            result.append({
                "id": f"s{seed}_{arm}", "seed": seed, "arm": arm,
                "training": "mathematical_spectrogram" if arm == "trained" else "initial_random",
                "path": str(path), "sha256": evidence["sha256"], "parameters": PARAMETERS,
                "paired_initialization_sha256": initials[seed]["sha256"],
                "training_protocol_sha256": sha(source / "protocol.json"),
                "workspace": str(root),
            })
    return result


def historical_split_plan(workspace: str | Path | None = None) -> dict:
    """Read fixed reference/FIT/CAL indices, not pixels or outcome scores.

    These indices refer to v46/data/records.json. Only the locked reference is
    provided here; the runner must predeclare alternative reference episodes.
    """
    root = Path(workspace or WORKSPACE).resolve()
    path = root / "outputs/cloud_plan_20261105/gs_fp32/migration_manifest.json"
    manifest = _read(path)
    fitted = _read(root / "outputs/hypernet_v46/evaluation/fits.json")
    targets = {}
    for target in TARGETS:
        src = manifest["targets"][target]
        historical = fitted["s45029_spectrogram"]["multi__" + target]
        if src["reference_id"] != REFERENCE_IDS[target]:
            raise ValueError("Locked positive reference changed")
        if (src["reference"] != historical["reference_index"]
                or src["fit"] != historical["fit_indices"]
                or src["cal"] != historical["calibration_indices"]):
            raise ValueError("Historical deployment and v46 splits disagree")
        if len(src["fit"]) != 512 or len(set(src["fit"])) != 512:
            raise ValueError("Expected exactly 512 distinct FIT negatives")
        if set(src["fit"]) & set(src["cal"]):
            raise ValueError("FIT/CAL overlap")
        if src["reference"] in set(src["fit"]) | set(src["cal"]):
            raise ValueError("Reference leaked into negative pool")
        targets[target] = {key: src[key] for key in ("reference", "reference_id", "fit", "cal")}
    return {
        "targets": targets, "row_count": manifest["used_event_count"],
        "development_indices": manifest["development_indices"],
        "records_path": str(root / "outputs/hypernet_v46/data/records.json"),
        "migration_manifest_sha256": sha(path),
        "effective_dimension": SINGLE_DIMENSION,
    }


def _input(inputs: Mapping, duration: str, view: str):
    candidates = []
    if duration in inputs and isinstance(inputs[duration], Mapping) and view in inputs[duration]:
        candidates.append(inputs[duration][view])
    if (duration, view) in inputs:
        candidates.append(inputs[(duration, view)])
    if view + "_" + duration in inputs:
        candidates.append(inputs[view + "_" + duration])
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one explicit input for {duration}/{view}")
    arr = candidates[0]
    if arr.dtype != np.float32 or arr.ndim != 3 or arr.shape[1:] != (128, 128):
        raise ValueError(f"Expected float32 N x128x128 input for {duration}/{view}")
    return arr


def extract_features(entry: dict, inputs_dict: Mapping, device: str = "cpu", batch: int = 32,
                     progress: Callable[[dict], None] | None = None,
                     workspace: str | Path | None = None) -> dict[str, np.ndarray]:
    """Explicit inference only; no data is opened and no features are written.

    The caller supplies already-bound inputs. Disable TF32 and AMP in both CPU
    and any explicitly requested CUDA execution; the runner must parity-gate
    and freeze its numerical runtime before scoring an evaluation cohort.
    """
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("Positive integer batch size required")
    root = Path(workspace or entry.get("workspace") or WORKSPACE).resolve()
    checkpoint = Path(entry["path"])
    if sha(checkpoint) != entry["sha256"] or entry["parameters"] != PARAMETERS:
        raise ValueError("Unbound or incorrect checkpoint")
    arrays = {(d, v): _input(inputs_dict, d, v) for d in DURATIONS for v in VIEWS}
    lengths = {len(a) for a in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("All eight input arrays must have the same nonzero event count")
    count = next(iter(lengths))
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    arch = _module("factorial_deployment_encoder", root / "outputs/cloud_plan_20261105/gs_fp32/kaggle_pilot/model.py")
    net = arch.load_encoder(checkpoint, device).eval()
    if sum(p.numel() for p in net.parameters()) != PARAMETERS:
        raise ValueError("Encoder parameter count changed")
    if any(p.dtype != torch.float32 for p in net.parameters()):
        raise ValueError("Expected float32 encoder parameters")
    result = {}
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for duration in DURATIONS:
            parts = []
            for view in VIEWS:
                arr = arrays[(duration, view)]
                features = np.empty((count, SINGLE_DIMENSION // 2), dtype=np.float32)
                for start in range(0, count, batch):
                    chunk = np.array(arr[start:start + batch], dtype=np.float32, copy=True)
                    if not np.isfinite(chunk).all() or chunk.min() < 0 or chunk.max() > 1:
                        raise ValueError("Invalid preprocessed relative intensity")
                    tensor = torch.from_numpy(chunk[:, None]).repeat(1, 3, 1, 1).to(device)
                    features[start:start + len(chunk)] = arch.descriptor_base(net.net(tensor)).cpu().numpy()
                    if progress:
                        progress({"model": entry["id"], "duration": duration, "view": view,
                                  "done": min(start + batch, count), "total": count})
                parts.append(features)
            joined = np.concatenate(parts, axis=1)
            joined /= np.maximum(np.linalg.norm(joined, axis=1, keepdims=True), 1e-12)
            if not np.isfinite(joined).all() or joined.dtype != np.float32:
                raise ValueError("Invalid extracted descriptor")
            result[duration] = joined
    return result


def joint_features(durations: Mapping[str, np.ndarray]) -> np.ndarray:
    """Historical equal-weight fusion; float32 concat(d05,d1,d2,d4)/2."""
    if set(durations) != set(DURATIONS):
        raise ValueError("All four authentic durations are required")
    sizes = set()
    for duration in DURATIONS:
        arr = durations[duration]
        if arr.dtype != np.float32 or arr.ndim != 2 or arr.shape[1] != SINGLE_DIMENSION:
            raise ValueError("Expected float32 N x4032 duration descriptors")
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite duration descriptor")
        if not np.allclose(np.linalg.norm(arr, axis=1), 1, rtol=0, atol=3e-6):
            raise ValueError("Duration descriptor is not unit normalized")
        sizes.add(len(arr))
    if len(sizes) != 1:
        raise ValueError("Duration event orders/counts differ")
    return np.concatenate([durations[d] for d in DURATIONS], axis=1) / np.float32(2)


def fit_eldaneg(background: np.ndarray, reference: np.ndarray,
                effective_dimension: int = SINGLE_DIMENSION,
                workspace: str | Path | None = None) -> dict:
    """Call the original unchanged v46 shrinkage ELDA implementation.

    Uses exactly 512 negative FIT rows in the real experiment. Smaller matrices
    are allowed only so the pure numerical self-test can compare a direct inverse.
    No CAL, development or evaluation rows may be passed by the runner.
    """
    root = Path(workspace or WORKSPACE).resolve()
    original = _module("factorial_original_elda", root / "outputs/hypernet_v46/common.py")
    state = original.fit(background, reference, effective_dimension)
    return dict(state, method="elda")


def prepare_elda(background: np.ndarray, effective_dimension: int = SINGLE_DIMENSION) -> dict:
    """Reference-independent factorization, exactly matching the historical fit.

    Keep this transient object in memory and reuse it for each of the five
    predeclared reference episodes. Only completed matcher states are artifacts.
    The expression and operation order mirror v46/common.py::fit.
    """
    bg = np.asarray(background, np.float64)
    if (bg.ndim != 2 or len(bg) <= 1 or bg.shape[1] == 0
            or effective_dimension <= 0 or not np.isfinite(bg).all()):
        raise ValueError("Expected finite FIT matrix and positive effective dimension")
    mu = bg.mean(0)
    x = bg - mu
    n = len(x)
    beta = .5 / (n - 1)
    ridge = max(.5 * np.square(x).sum() / ((n - 1) * effective_dimension), 1e-12)
    gram = x @ x.T
    v, b = np.linalg.eigh(gram)
    inverse = (b / (ridge + beta * np.maximum(v, 0))) @ b.T
    return dict(mu=mu, x=x, beta=beta, ridge=ridge, inverse=inverse,
                effective_dimension=effective_dimension)


def fit_prepared_elda(prepared: Mapping, reference: np.ndarray) -> dict:
    """Complete a one-reference matcher using the unchanged Woodbury formula."""
    ref = np.asarray(reference, np.float64)
    mu, x = prepared["mu"], prepared["x"]
    if ref.shape != mu.shape or not np.isfinite(ref).all():
        raise ValueError("Expected one finite reference of the prepared dimension")
    beta, ridge, inverse = prepared["beta"], prepared["ridge"], prepared["inverse"]
    r = ref - mu
    weight = r / ridge - (beta / ridge) * (x.T @ (inverse @ (x @ r)))
    return dict(method="elda", weight=weight, offset=np.array(-mu @ weight),
                ridge=np.array(ridge), beta=np.array(beta), mu=mu,
                effective_dimension=np.array(prepared["effective_dimension"]))


def fit_cosine(reference: np.ndarray) -> dict:
    """One positive vector only; no centering, whitening or FIT negatives."""
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 1 or not np.isfinite(ref).all():
        raise ValueError("Expected one finite reference descriptor")
    norm = float(np.linalg.norm(ref))
    if norm <= 1e-12:
        raise ValueError("Cosine reference has zero norm")
    return {"method": "cosine", "weight": ref / norm, "offset": np.array(0., dtype=np.float64)}


def score(features: np.ndarray, state: Mapping, batch: int = 256) -> np.ndarray:
    """Increasing target-evidence scores, with float64 deterministic reduction."""
    f = np.asarray(features)
    weight = np.asarray(state["weight"], dtype=np.float64)
    method = str(np.asarray(state.get("method", "elda")).item())
    if method not in ("elda", "cosine") or batch < 1:
        raise ValueError("Unknown matcher or invalid batch")
    if f.ndim != 2 or weight.shape != (f.shape[1],):
        raise ValueError("Matcher/feature dimensions disagree")
    if not np.isfinite(weight).all() or not np.isfinite(state["offset"]).all():
        raise ValueError("Nonfinite matcher state")
    output = np.empty(len(f), dtype=np.float64)
    for start in range(0, len(f), batch):
        q = np.asarray(f[start:start + batch], dtype=np.float64)
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite query descriptor")
        if method == "cosine":
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            if np.any(norms <= 1e-12):
                raise ValueError("Cosine query has zero norm")
            q = q / norms
        output[start:start + len(q)] = np.einsum("ij,j->i", q, weight, optimize=False) + state["offset"]
    return output


def calibrate(calibration_scores: np.ndarray, scores: np.ndarray, alpha: float) -> tuple:
    """Deployment's conservative upper-tail rank, with ties kept negative.

    Returns p-values, Boolean alarms and the score cutoff (None if the finite
    calibration resolution cannot support the requested alpha). Sorting is
    internal; no outcome labels or evaluation-derived thresholds are used.
    """
    cal = np.asarray(calibration_scores, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if cal.ndim != 1 or not len(cal) or values.ndim != 1:
        raise ValueError("Expected nonempty 1D CAL scores and 1D query scores")
    if not np.isfinite(cal).all() or not np.isfinite(values).all() or not 0 < alpha < 1:
        raise ValueError("Invalid calibration values or alpha")
    cal = np.sort(cal)
    fraction = Fraction(str(alpha))
    k = (len(cal) + 1) * fraction.numerator // fraction.denominator
    numerators = 1 + len(cal) - np.searchsorted(cal, values, side="left")
    return numerators / (len(cal) + 1), numerators <= k, float(cal[-k]) if k else None


def self_test(workspace: str | Path | None = None) -> dict:
    """Small mathematical fixtures only; never reads real images or outcomes."""
    rng = np.random.default_rng(2026110507)
    background = rng.normal(size=(17, 9))
    reference = rng.normal(size=9)
    queries = rng.normal(size=(13, 9))
    state = fit_eldaneg(background, reference, 9, workspace)
    mu = background.mean(axis=0)
    inverse = np.linalg.inv(.5 * np.cov(background, rowvar=False) + float(state["ridge"]) * np.eye(9))
    expected = (queries - mu) @ inverse @ (reference - mu)
    actual = score(queries, state, batch=3)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(actual, score(queries, state, batch=7))
    prepared = prepare_elda(background, 9)
    for ref in [reference, *rng.normal(size=(4, 9))]:
        original_state = fit_eldaneg(background, ref, 9, workspace)
        reused_state = fit_prepared_elda(prepared, ref)
        for key in ("weight", "offset", "ridge", "beta", "mu", "effective_dimension"):
            np.testing.assert_array_equal(reused_state[key], original_state[key])
        np.testing.assert_array_equal(score(queries, reused_state), score(queries, original_state))
    repeated = lambda x: np.concatenate([x] * 4, axis=-1) / 2
    repeated_state = fit_eldaneg(repeated(background), repeated(reference), 9, workspace)
    np.testing.assert_allclose(score(repeated(queries), repeated_state), actual, rtol=1e-10, atol=1e-10)
    cosine_state = fit_cosine(reference)
    cosine_expected = queries @ reference / (np.linalg.norm(queries, axis=1) * np.linalg.norm(reference))
    np.testing.assert_allclose(score(queries, cosine_state), cosine_expected, rtol=1e-14, atol=1e-14)
    _, ties, cutoff = calibrate(np.array([1., 2., 3., 4.]), np.array([4., 4.0001]), .2)
    np.testing.assert_array_equal(ties, [False, True])
    _, unsupported, null_cutoff = calibrate(np.array([1., 2.]), np.array([100.]), .001)
    if unsupported.any() or null_cutoff is not None or cutoff != 4.:
        raise AssertionError("Conservative calibration mismatch")
    return {"passed": True, "real_data_access": False, "neural_inference": False,
            "elda_direct_inverse_max_abs_error": float(np.max(np.abs(actual - expected))),
            "prepared_elda_exact_parity_five_references": True,
            "repeated_input_parity": True, "cosine_parity": True, "calibration_ties_and_resolution": True}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Pure array numerical checks; no image access")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        parser.print_help()
