"""Independently recompute v09 public metrics, paired intervals and localization."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from verify_benchmark import metric, read


def verify(directory):
    directory = Path(directory)
    aggregate = read(directory/'aggregate_metrics.json')
    analysis = read(directory/'paired_analysis.json')
    scores_path = directory/'scores.jsonl'
    cohort_path = directory/'public_cohort_manifest.json'
    assert sha256(scores_path.read_bytes()).hexdigest() == aggregate['public_artifacts']['scores_sha256']
    assert sha256(cohort_path.read_bytes()).hexdigest() == aggregate['public_artifacts']['cohort_manifest_sha256']
    cohort = read(cohort_path)
    pairs = {row['id']: row for row in cohort['pairs']}
    assert len(pairs) == 720
    identities = {key: (int(row['support_image_id'].removeprefix('coco_image_')),
        int(row['query_image_id'].removeprefix('coco_image_')), row['category_id'], row['split'], row['present'])
        for key, row in pairs.items()}
    groups = defaultdict(list)
    for line in scores_path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        assert math.isfinite(row['score']) and type(row['present']) is bool
        groups[row['model']].append(row)
    assert set(groups) == set(aggregate['models']) and len(groups) == 8
    categories = sorted(aggregate['protocol']['test_categories'])
    draws = np.random.default_rng(1809).integers(0, 6, (10000, 6))
    matrices = {}
    max_error = 0.
    for alias, rows in groups.items():
        assert len(rows) == 720
        observed = {r['pair_id']: (r['support_coco_id'],r['query_coco_id'],r['category'],r['split'],r['present']) for r in rows}
        assert len(observed) == len(rows) and observed == identities
        dev, test = ([r for r in rows if r['split'] == split] for split in ('dev','test'))
        assert len(dev) == len(test) == 360
        assert len({r['category'] for r in dev}) == len({r['category'] for r in test}) == 6
        assert not {r['category'] for r in dev} & {r['category'] for r in test}
        negative = sorted((r['score'] for r in dev if not r['present']), reverse=True)
        assert len(negative) == 180
        cutoff = math.nextafter(negative[9], math.inf)
        saved = aggregate['models'][alias]
        assert saved['calibration']['threshold'] == cutoff
        assert sum(value >= cutoff for value in negative) <= 9
        for label, selected in (('dev',dev),('test',test)):
            for key, value in metric(selected, cutoff).items():
                expected = saved[label]['overall'][key]
                if value is None:
                    assert expected is None
                else:
                    error = abs(value-expected)
                    assert error < 1e-10, (alias,label,key,error)
                    max_error = max(error,max_error)
        matrix = []
        for category in categories:
            selected = [r for r in test if r['category'] == category]
            m = metric(selected,cutoff)
            assert m['n_positive'] == m['n_negative'] == 30
            matrix.append([m['tpr'],m['fpr'],m['balanced_accuracy'],m['auroc']])
        matrices[alias] = np.asarray(matrix)
        positive = [r for r in test if r['present']]
        overlaps, accepted = [], []
        for row in positive:
            box = row['bbox_xyxy']
            best = 0.
            for gt in pairs[row['pair_id']]['evaluation_annotations']['query_target_boxes']:
                if box is None:
                    continue
                left, top = max(box[0],gt[0]), max(box[1],gt[1])
                right, bottom = min(box[2],gt[2]), min(box[3],gt[3])
                intersection = max(0.,right-left)*max(0.,bottom-top)
                area = (box[2]-box[0])*(box[3]-box[1])+(gt[2]-gt[0])*(gt[3]-gt[1])-intersection
                best = max(best,intersection/area if area else 0.)
            overlaps.append(best)
            if row['score'] >= cutoff:
                accepted.append(best)
        actual = {'positive_pairs':len(positive),'accepted_positive_pairs':len(accepted),
            'all_positive_iou50':sum(x >= .5 for x in overlaps),'accepted_iou50':sum(x >= .5 for x in accepted),
            'accepted_zero_overlap':sum(x == 0 for x in accepted),
            'all_positive_mean_iou':float(np.mean(overlaps)),
            'accepted_mean_iou':float(np.mean(accepted)) if accepted else None}
        for key, value in actual.items():
            expected = analysis['localization'][alias][key]
            assert (value is None and expected is None) or abs(value-expected)<1e-12,(alias,key)
    for alias, matrix in matrices.items():
        delta = matrix-matrices['scratch_reference']
        sampled = delta[draws].mean(axis=1)
        for i,key in enumerate(('tpr','fpr','balanced_accuracy','auroc_category_macro')):
            expected = analysis['contrasts_against_scratch'][alias][key]
            values = (delta[:,i].mean(),np.quantile(sampled[:,i],.025),np.quantile(sampled[:,i],.975))
            for field,value in zip(('difference','low','high'),values):
                assert abs(float(value)-expected[field])<1e-12,(alias,key,field)
    return {'passed':True,'models':8,'prediction_rows':5760,'aggregate_partitions_checked':16,
            'max_metric_absolute_error':max_error,'paired_category_bootstrap_verified':True,
            'localization_verified':True,'all_cutoffs_recovered_from_dev_only':True,
            'score_identities_match_public_cohort':True,
            'scope':'Independent arithmetic, pairing, intervals and box diagnostics; not an inference rerun'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,default=Path('benchmarks/v0.9'))
    print(json.dumps(verify(parser.parse_args().directory),indent=2))
