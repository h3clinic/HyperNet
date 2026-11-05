"""Post hoc paired DEV uncertainty; no scoring, fitting or holdout access.

Two comparisons requested after v47 results were known.  Thresholds and all
predictions stay fixed.  The same event-group draws serve every target, seed,
configuration and contrast.  Training seeds are averaged, not resampled.
"""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

R = Path(__file__).resolve().parent
V46 = R.parent / 'hypernet_v46'
O = R / 'evaluation'
SEEDS = [45011, 45029, 45047]
TARGETS = ['Low_Frequency_Burst', 'Low_Frequency_Lines', 'Scattered_Light', 'Blip']
METRICS = ['balanced_accuracy', 'mcc', 'recall', 'fa_per10000']
ALPHA = .001
BOOTSTRAPS = 2000
RNG_SEED = 47051
CONFIGURATIONS = {
    'original_pooled_single': dict(matcher='single1s', pool='original', grouping='pooled'),
    'original_pooled_mean': dict(matcher='late_mean4', pool='original', grouping='pooled'),
    'expanded_pooled_mean': dict(matcher='late_mean4', pool='expanded', grouping='pooled'),
    'expanded_detector_mean': dict(matcher='late_mean4', pool='expanded', grouping='detector'),
}
CONTRASTS = [
    dict(id='mean4_minus_single1s_original_pooled', left='original_pooled_mean', right='original_pooled_single'),
    dict(id='detector_minus_pooled_mean4_expanded', left='expanded_detector_mean', right='expanded_pooled_mean'),
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(weights, labels, predictions):
    """Vectorized count-based metrics; rows are common bootstrap resamples."""
    pos = weights @ labels.astype(np.float64)
    neg = weights.sum(1) - pos
    tp = weights @ (labels & predictions).astype(np.float64)
    fp = weights @ (~labels & predictions).astype(np.float64)
    tn, fn = neg - fp, pos - tp
    assert (pos > 0).all() and (neg > 0).all()
    recall = tp / pos
    ba = .5 * (recall + tn / neg)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)
    return np.stack([ba, mcc, recall, fp / neg * 10000], axis=1)


def main():
    threadpool_limits(2)
    output = R / 'uncertainty.json'
    assert not output.exists(), 'Post hoc uncertainty artifact already exists; do not rerun'
    protocol = read(R / 'protocol.json')
    results = read(O / 'results.json')
    splits = read(R / 'splits.json')
    rows = read(V46 / 'data/records.json')
    assert results['complete'] and results['development_only'] and results['reserve_closed']
    assert results['protocol_sha256'] == sha(R / 'protocol.json')
    assert results['protocol_sha256'] == 'f07bef0f3bf62ca5dac0af4c83449aa90fb0b0840fce99bfd44cc46b2ea40967'
    for path in [R / 'splits.json', V46 / 'data/records.json']:
        assert protocol['files'][str(path)] == sha(path)
    dev = np.asarray(splits['indices']['development'], dtype=np.int64)
    assert len(dev) == 1558 and len(set(dev)) == len(dev)
    assert all(rows[i]['split'] == 'development' for i in dev)
    group_names = sorted({rows[i]['group'] for i in dev})
    group_index = {group: i for i, group in enumerate(group_names)}
    memberships = np.asarray([group_index[rows[i]['group']] for i in dev], dtype=np.int64)
    group_sizes = np.bincount(memberships, minlength=len(group_names))
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, len(group_names), size=(BOOTSTRAPS, len(group_names)), dtype=np.int64)
    multiplicities = np.zeros((BOOTSTRAPS, len(group_names)), dtype=np.int32)
    for b in range(BOOTSTRAPS):
        multiplicities[b] = np.bincount(draws[b], minlength=len(group_names))
    assert np.all(multiplicities.sum(1) == len(group_names))
    weights = multiplicities[:, memberships].astype(np.float64)
    original = np.ones((1, len(dev)), dtype=np.float64)
    labels = np.asarray([rows[i]['label'] for i in dev])
    files = [Path(__file__).resolve(), R / 'protocol.json', R / 'splits.json',
             O / 'results.json', V46 / 'data/records.json']
    sources = {str(path): sha(path) for path in files}
    decisions = {}
    for seed in SEEDS:
        for grouping in ['pooled', 'detector']:
            path = O / f's{seed}_spectrogram' / (grouping + '_decisions.npz')
            assert results['files'][str(path)] == sha(path)
            sources[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                decisions[seed, grouping] = {k: np.asarray(z[k], bool)[dev] for k in z.files}
    points, replicates, seed_points = {}, {}, {}
    for name, config in CONFIGURATIONS.items():
        seed_replicates, seed_values = [], []
        for seed in SEEDS:
            per_target, point_target = [], []
            for target in TARGETS:
                key = config['pool'] + '__' + config['matcher'] + '__' + target + '__' + str(ALPHA)
                prediction = decisions[seed, config['grouping']][key]
                y = labels == target
                per_target.append(metrics(weights, y, prediction))
                point = metrics(original, y, prediction)[0]
                frozen = [v for v in results['rows']
                          if v['model'] == f's{seed}_spectrogram' and v['target'] == target
                          and v['alpha'] == ALPHA and all(v[k] == x for k, x in config.items())]
                assert len(frozen) == 1
                np.testing.assert_allclose(point, [frozen[0][m] for m in METRICS], atol=1e-12, rtol=1e-12)
                point_target.append(point)
            seed_replicates.append(np.mean(per_target, axis=0))
            seed_values.append(np.mean(point_target, axis=0))
        replicates[name] = np.mean(seed_replicates, axis=0)
        points[name] = np.mean(seed_values, axis=0)
        seed_points[name] = {
            str(seed): dict(zip(METRICS, value.tolist())) for seed, value in zip(SEEDS, seed_values)
        }

    contrasts = []
    for contrast in CONTRASTS:
        left, right = contrast['left'], contrast['right']
        difference = replicates[left] - replicates[right]
        point = points[left] - points[right]
        interval = np.quantile(difference, [.025, .975], axis=0, method='linear')
        contrasts.append(dict(**contrast, metrics={
            metric: dict(left=float(points[left][j]), right=float(points[right][j]),
                         difference=float(point[j]), percentile_95=[float(interval[0, j]), float(interval[1, j])],
                         units='false alarms per10000 negatives' if metric == 'fa_per10000' else 'fraction')
            for j, metric in enumerate(METRICS)
        }, bootstrap_differences={metric: difference[:, j].tolist() for j, metric in enumerate(METRICS)}))
    audit_path = O / 'audit.json'
    if audit_path.exists():
        audit = read(audit_path)
        assert audit['passed'] and audit['results_sha256'] == sha(O / 'results.json')
        sources[str(audit_path)] = sha(audit_path)
    for path, digest in sources.items():
        assert sha(path) == digest, 'A source changed during postprocessing'
    result = dict(
        complete=True,
        post_hoc=True,
        preregistered_in_original_experiment=False,
        development_only=True,
        no_new_model_scoring=True,
        no_threshold_refit=True,
        no_reserved_inputs_read=True,
        main_evaluation_audit='passed_and_bound' if audit_path.exists() else 'pending_at_postprocessing',
        protocol_sha256=sha(R / 'protocol.json'),
        source_hashes=sources,
        alpha=ALPHA,
        configurations=CONFIGURATIONS,
        bootstrap=dict(
            replicates=BOOTSTRAPS, seed=RNG_SEED, numpy_version=np.__version__,
            resampling='Nonparametric event-group sampling with replacement; unstratified.',
            group_field='group', group_order='Lexicographically sorted existing DEV group identifiers.',
            events=len(dev), groups=len(group_names),
            group_size_counts=dict(sorted(collections.Counter(group_sizes.tolist()).items())),
            same_draws_across_seeds_targets_and_contrasts=True,
            seeds_fixed_not_resampled=SEEDS,
            aggregation='Within each resample: metric per target, mean over4targets, then mean over3fixedseeds.',
            interval='2.5th and97.5th percentiles of paired differences; NumPy linear quantile.',
            draws_sha256=hashlib.sha256(draws.astype('<i8', copy=False).tobytes()).hexdigest(),
        ),
        point_seed_metrics=seed_points,
        contrasts=contrasts,
        limitations=[
            'The comparisons and uncertainty analysis were requested after DEV outcomes were observed; these are descriptive post hoc intervals, not preregistered confirmatory inference.',
            'DEV has repeatedly informed research choices. Bootstrap resampling does not undo adaptive development or estimate performance on the closed reserve.',
            'All1558existing DEV groups are singletons. This bootstrap preserves documented group identity but does not model longer-range detector/run/time correlations; intervals may be too narrow under such dependence.',
            'Only DEV sampling is resampled. Reference selection, FIT/TUNE/CAL sampling, fixed training seeds and domain shift are held fixed, so this is not total model or deployment uncertainty.',
            'Three seeds predict the same events and do not triple the sample size. They are averaged within each common event resample, not treated as independent datasets.',
            'Two contrasts and four metrics are reported without multiplicity adjustment; they were chosen after a16-configuration development grid. Excluding zero is not a multiple-comparison-corrected discovery claim.',
            'False-alarm results describe macro one-vs-rest rates, not the union probability of any alert or a certified archive-level error budget.',
            'Unstratified resampling estimates uncertainty for this observed mixture, not robustness to changed class/detector/observing-era composition.',
        ],
    )
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output), contrasts=[
        {k: v for k, v in contrast.items() if k != 'bootstrap_differences'} for contrast in contrasts
    ]), indent=2))


if __name__ == '__main__':
    main()
