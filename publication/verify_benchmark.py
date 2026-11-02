"""Independently verify published aggregate metrics from compact raw scores.

Uses scikit-learn for ranking metrics rather than HyperNet's metric functions.
This verifies arithmetic and pairing, not inference or image freshness.
"""
from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metric(rows, cutoff):
    positive = [r for r in rows if r['present']]
    negative = [r for r in rows if not r['present']]
    tp = sum(r['score'] >= cutoff for r in positive)
    fp = sum(r['score'] >= cutoff for r in negative)
    tn, fn = len(negative)-fp, len(positive)-tp
    labels, scores = [r['present'] for r in rows], [r['score'] for r in rows]
    return {'n':len(rows),'n_positive':len(positive),'n_negative':len(negative),
            'tp':tp,'fp':fp,'tn':tn,'fn':fn,'tpr':tp/len(positive),'fpr':fp/len(negative),
            'precision':tp/(tp+fp) if tp+fp else None,
            'accuracy':(tp+tn)/len(rows),'balanced_accuracy':.5*(tp/len(positive)+tn/len(negative)),
            'auroc':float(roc_auc_score(labels,scores)),
            'average_precision':float(average_precision_score(labels,scores))}


def verify(directory):
    directory = Path(directory)
    summary = read(directory/'aggregate_metrics.json')
    prediction_file = directory/'scores.jsonl'
    digest = hashlib.sha256(prediction_file.read_bytes()).hexdigest()
    assert digest == summary['public_artifacts']['scores_sha256'], 'Raw-score checksum mismatch'
    cohort_file = directory/'public_cohort_manifest.json'
    assert hashlib.sha256(cohort_file.read_bytes()).hexdigest() == summary['public_artifacts']['cohort_manifest_sha256']
    cohort = read(cohort_file)
    expected_identities = {r['id']:(int(r['support_image_id'].removeprefix('coco_image_')),
        int(r['query_image_id'].removeprefix('coco_image_')),r['category_id'],r['split'],r['present'])
        for r in cohort['pairs']}
    assert len(expected_identities)==960
    groups = defaultdict(list)
    with prediction_file.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            assert math.isfinite(row['score'])
            assert type(row['present']) is bool
            groups[row['model']].append(row)
    assert set(groups) == set(summary['models'])
    identities = None
    checked = 0
    max_error = 0.
    for name, rows in groups.items():
        assert len(rows) == 960
        observed = {r['pair_id']:(r['support_coco_id'],r['query_coco_id'],r['category'],r['split'],r['present']) for r in rows}
        assert len(observed) == len(rows), 'Duplicate pair ID'
        if identities is None:
            identities = observed
        assert identities == observed, 'Models evaluated different pairs'
        assert observed == expected_identities, 'Score identities differ from public cohort manifest'
        dev = [r for r in rows if r['split']=='dev']
        test = [r for r in rows if r['split']=='test']
        assert len(dev)==len(test)==480
        assert len({r['category'] for r in dev})==len({r['category'] for r in test})==8
        assert not {r['category'] for r in dev}&{r['category'] for r in test}
        negatives = sorted([r['score'] for r in dev if not r['present']],reverse=True)
        assert len(negatives)==240
        cutoff = math.nextafter(negatives[12],math.inf)
        saved = summary['models'][name]
        assert cutoff == saved['calibration']['threshold'], 'Cutoff was not determined by DEV order statistic'
        assert sum(s>=cutoff for s in negatives)<=12
        for label, selected in [('dev',dev),('test',test)]:
            measured = metric(selected,cutoff)
            published = saved[label]['overall']
            for key,value in measured.items():
                if value is None:
                    assert published[key] is None
                else:
                    error = abs(value-published[key])
                    assert error < 1e-10,(name,label,key,error)
                    max_error = max(max_error,error)
            checked += 1
    return {'passed':True,'models':len(groups),'prediction_rows':sum(map(len,groups.values())),
            'aggregate_partitions_checked':checked,'max_metric_absolute_error':max_error,
            'pair_identity_sets_equal':True,'all_cutoffs_recovered_from_dev_only':True,
            'score_identities_match_public_cohort':True,
            'scope':'Checks published arithmetic and model pairing; does not rerun inference or establish data provenance'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,nargs='?',default=Path('benchmarks/v0.8.1'))
    args=parser.parse_args()
    print(json.dumps(verify(args.directory),indent=2))
