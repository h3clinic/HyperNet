"""One-reference retrieval benchmark on Gravity Spy LIGO glitch spectrograms.

Each item is a whole spectrogram, so the task is retrieval, not localization:
one labeled reference glitch ranks every haystack image by similarity. The
labels are used only to choose references and to score the ranking.

Data: Gravity Spy Training Set v1.1 (Zenodo record 1486046, CC BY 4.0),
trainingsetv1d1.h5 and trainingset_v1d1_metadata.csv.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ZENODO_RECORD = 'https://zenodo.org/records/1486046'
MD5 = {'trainingsetv1d1.h5': 'a6665b4ebee7b4ea57164ee743c2bd11',
       'trainingset_v1d1_metadata.csv': 'c16d135f5ad3f029a3ef32114e280e79'}
DURATIONS = ('0.5', '1.0', '2.0', '4.0')
# Catch-all and empty classes have no single pattern to search for. They stay
# in the haystack as distractors but are never used as targets.
NON_TARGET = ('None_of_the_Above', 'No_Glitch')
RARE_LIMIT = 100  # a target class with fewer labeled examples in total is "rare"


def target_classes(labels):
    names, counts = np.unique(np.asarray(labels), return_counts=True)
    return [str(n) for n in names if n not in NON_TARGET], {str(n): int(c) for n, c in zip(names, counts)}


def choose_references(labels, splits, classes, per_class, seed):
    """Fixed reference rows drawn only from the train split."""
    labels = np.asarray(labels); splits = np.asarray(splits)
    rng = np.random.default_rng(seed); chosen = {}
    for name in classes:
        pool = np.flatnonzero((labels == name) & (splits == 'train'))
        if len(pool) < per_class:
            raise ValueError(f'{name} has only {len(pool)} train rows')
        chosen[name] = sorted(int(i) for i in rng.choice(pool, per_class, replace=False))
    return chosen


def average_precision(scores, positive):
    """Non-interpolated AP: mean precision at each positive's rank."""
    scores = np.asarray(scores, float); positive = np.asarray(positive, bool)
    if positive.sum() == 0:
        raise ValueError('AP needs at least one positive')
    order = np.argsort(-scores, kind='stable'); hits = positive[order]
    ranks = np.flatnonzero(hits) + 1
    return float(np.mean(np.arange(1, len(ranks) + 1) / ranks))


def auroc(scores, positive):
    """Probability that a random positive outscores a random negative (ties count half)."""
    scores = np.asarray(scores, float); positive = np.asarray(positive, bool)
    pos, neg = scores[positive], scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError('AUROC needs positives and negatives')
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind='mergesort'); ranks = np.empty(len(allv))
    sorted_v = allv[order]; i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_at_false_alarm_rate(negative_scores, rate):
    """Lowest cutoff that accepts at most floor(rate * n) of these negatives."""
    s = np.sort(np.asarray(negative_scores, float))[::-1]
    if len(s) == 0 or not 0 <= rate < 1:
        raise ValueError('Need negatives and 0 <= rate < 1')
    allowed = int(np.floor(rate * len(s)))
    return float(np.nextafter(s[allowed], np.inf))


def operating_point(scores, positive, threshold):
    scores = np.asarray(scores, float); positive = np.asarray(positive, bool)
    accepted = scores >= threshold; negatives = int((~positive).sum())
    false_alarms = int((accepted & ~positive).sum())
    return {'found': int((accepted & positive).sum()), 'positives': int(positive.sum()),
            'recall': float((accepted & positive).sum() / max(1, positive.sum())),
            'false_alarms': false_alarms, 'negatives': negatives,
            'false_alarms_per_10k': float(1e4 * false_alarms / max(1, negatives))}


def one_shot_accuracy(score_rows, item_labels, classes):
    """N-way, one-reference classification: each item takes the class of its best reference.

    score_rows maps class -> scores over the items for that class's single reference.
    """
    matrix = np.stack([np.asarray(score_rows[c], float) for c in classes])
    labels = np.asarray(item_labels); keep = np.isin(labels, classes)
    predicted = np.asarray(classes)[matrix[:, keep].argmax(0)]
    return float(np.mean(predicted == labels[keep])), int(keep.sum())


def paired_class_bootstrap(first, second, samples=10000, seed=0):
    """95% interval for mean(first - second), resampling classes with replacement."""
    diff = np.asarray(first, float) - np.asarray(second, float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (samples, len(diff)))].mean(1)
    return float(diff.mean()), [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def evaluate_method(scores, labels, dev, test, classes, rare, references_per_class, rate, item_ids=None):
    """Score one method. scores maps (class, reference index) -> scores over all haystack items.

    DEV negatives of each reference set its own cutoff; every reported number is TEST.
    """
    labels = np.asarray(labels); dev = np.asarray(dev, bool); test = np.asarray(test, bool)
    records, per_class = [], {}
    test_index = np.flatnonzero(test)
    for name in classes:
        group = []
        for r in range(references_per_class):
            s = np.asarray(scores[(name, r)], float); pos = labels == name
            cutoff = threshold_at_false_alarm_rate(s[dev & ~pos], rate)
            ranked = test_index[np.argsort(-s[test], kind='stable')[:20]]
            rec = {'class': name, 'reference': r,
                   'test_ap': average_precision(s[test], pos[test]), 'test_auroc': auroc(s[test], pos[test]),
                   'test_base_rate': float(pos[test].mean()), 'dev_cutoff': cutoff,
                   'test_at_dev_cutoff': operating_point(s[test], pos[test], cutoff),
                   'test_top20_hits': int((labels[ranked] == name).sum())}
            if item_ids is not None:
                rec['test_top20_gravityspy_ids'] = [str(item_ids[i]) for i in ranked]
            group.append(rec); records.append(rec)
        per_class[name] = {'ap': float(np.mean([x['test_ap'] for x in group])),
                           'auroc': float(np.mean([x['test_auroc'] for x in group])),
                           'recall_at_1pct_dev_far': float(np.mean([x['test_at_dev_cutoff']['recall'] for x in group])),
                           'false_alarms_per_10k': float(np.mean([x['test_at_dev_cutoff']['false_alarms_per_10k'] for x in group])),
                           'top20_precision': float(np.mean([x['test_top20_hits'] / 20 for x in group])),
                           'base_rate': group[0]['test_base_rate']}
    accuracy = [one_shot_accuracy({c: np.asarray(scores[(c, r)])[test] for c in classes}, labels[test], classes)[0]
                for r in range(references_per_class)]
    mean = lambda key, subset: float(np.mean([per_class[c][key] for c in subset]))
    summary = {'per_class': per_class,
               'mean_ap': mean('ap', classes), 'rare_mean_ap': mean('ap', rare),
               'mean_auroc': mean('auroc', classes),
               'mean_recall_at_1pct_dev_far': mean('recall_at_1pct_dev_far', classes),
               'rare_mean_recall_at_1pct_dev_far': mean('recall_at_1pct_dev_far', rare),
               'mean_false_alarms_per_10k': mean('false_alarms_per_10k', classes),
               'mean_top20_precision': mean('top20_precision', classes),
               'one_shot_accuracy_mean': float(np.mean(accuracy)), 'one_shot_accuracy_draws': accuracy,
               'one_shot_ways': len(classes)}
    return summary, records


def dev_mean_ap(scores, labels, dev, classes, references_per_class):
    """Selection metric: mean AP on DEV only, averaged over references then classes."""
    labels = np.asarray(labels); dev = np.asarray(dev, bool)
    return float(np.mean([np.mean([average_precision(np.asarray(scores[(c, r)])[dev], labels[dev] == c)
                                   for r in range(references_per_class)]) for c in classes]))


def to_uint8_rgb(image):
    """One grayscale spectrogram in [0, 1] (any shape with a 2D core) to uint8 RGB."""
    a = np.squeeze(np.asarray(image, np.float32))
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Expected one finite 2D spectrogram')
    gray = np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def load_split_images(h5_path, metadata, duration):
    """Read one duration's spectrograms for every metadata row, in metadata order.

    The HDF5 file is grouped as label / sample_type / gravityspy_id / '<duration>.png'.
    """
    import h5py
    if duration not in DURATIONS:
        raise ValueError(f'duration must be one of {DURATIONS}')
    images = []
    with h5py.File(Path(h5_path), 'r') as f:
        for label, split, gid in zip(metadata['label'], metadata['sample_type'], metadata['gravityspy_id']):
            images.append(to_uint8_rgb(f[label][split][gid][duration + '.png'][()]))
    return images
