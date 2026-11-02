"""Same v11 scene-level metrics; paired family bootstrap respects dependence."""
import math
import numpy as np


def cutoff_from_dev(rows, max_fpr=.05):
    if not rows or any(r['split'] != 'dev' for r in rows):
        raise ValueError('Calibration accepts DEV rows only')
    negatives = sorted([r['score'] for r in rows if not r['present']], reverse=True)
    if not negatives or not 0 <= max_fpr < 1:
        raise ValueError('Requires negatives and 0 <= max_fpr < 1')
    return float(np.nextafter(negatives[math.floor(max_fpr*len(negatives))], math.inf))


def summarize(rows, cutoff):
    positives = [r for r in rows if r['present']]
    negatives = [r for r in rows if not r['present']]
    tp = sum(r['score'] >= cutoff for r in positives)
    fp = sum(r['score'] >= cutoff for r in negatives)
    localized = sum(r['score'] >= cutoff and r['iou'] >= .5 for r in positives)
    auc = np.mean([[float(p['score'] > n['score'])+.5*float(p['score'] == n['score'])
        for n in negatives] for p in positives])
    ranked = sorted(rows, key=lambda r: r['score'], reverse=True)
    cumulative_tp, cumulative_predictions, ap, cursor = 0, 0, 0., 0
    while cursor < len(ranked):
        end = cursor+1
        while end < len(ranked) and ranked[end]['score'] == ranked[cursor]['score']:
            end += 1
        group_tp = sum(r['present'] and r['iou'] >= .5 for r in ranked[cursor:end])
        cumulative_tp += group_tp; cumulative_predictions += end-cursor
        ap += cumulative_tp/cumulative_predictions*group_tp/len(positives)
        cursor = end
    pairs = {}
    for r in rows:
        pair = pairs.setdefault(r['episode_id'], {})
        if bool(r['present']) in pair:
            raise ValueError('Duplicate presence state in episode')
        pair[bool(r['present'])] = r['score']
    pair_accuracy = np.mean([float(v[True] > v[False])+.5*float(v[True] == v[False]) for v in pairs.values()])
    return dict(positive_scenes=len(positives), negative_scenes=len(negatives), threshold=cutoff,
        accepted_positive=tp, false_positive=fp, localized_true_positive=localized,
        recall=tp/len(positives), false_positive_rate=fp/len(negatives),
        localized_recall_at_iou50=localized/len(positives),
        proposal_recall_at_iou50=sum(r['proposal_best_iou'] >= .5 for r in positives)/len(positives),
        auroc=float(auc), single_box_ap50=float(ap), paired_scene_ranking_accuracy=float(pair_accuracy),
        balanced_accuracy=.5*(tp/len(positives)+1-fp/len(negatives)))


def bootstrap(rows_by_method, cutoffs, samples=10000, seed=1812):
    families = sorted({r['family_id'] for rows in rows_by_method.values() for r in rows if r['split'] == 'test'})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(families), size=(samples, len(families)))
    values, distributions, result = {}, {}, {}
    for alias, rows in rows_by_method.items():
        values[alias] = np.array([np.mean([r['score'] >= cutoffs[alias] and r['iou'] >= .5
            for r in rows if r['split'] == 'test' and r['family_id'] == family and r['present']]) for family in families])
        distributions[alias] = values[alias][draws].mean(1)
        result[alias] = dict(estimate=float(values[alias].mean()),
            ci95=np.quantile(distributions[alias], [.025, .975]).tolist())
    differences = {}
    for control in ('old_cnn', 'lowres_patch'):
        differences['native_patch_minus_'+control] = dict(
            estimate=float((values['native_patch']-values[control]).mean()),
            ci95=np.quantile(distributions['native_patch']-distributions[control], [.025, .975]).tolist())
    return dict(unit='family', families=len(families), samples=samples, seed=seed,
                metric='localized_recall_at_iou50', methods=result, differences=differences,
                scope='Percentile intervals conditional on frozen DEV thresholds; no threshold refitting.')
