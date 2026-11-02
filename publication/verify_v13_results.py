"""Independently verify published v13 metrics and intervals without photos/weights."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_photo_metrics',ROOT/'scripts/audit_real_photo.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('benchmark',type=Path,nargs='?',default=ROOT/'benchmarks/v0.13');b=parser.parse_args().benchmark
    result=read(b/'aggregate_metrics.json');manifest=read(b/'cohort_manifest.json');provenance=read(b/'public_export_provenance.json')
    for name,record in provenance['files'].items():audit.require(sha(b/name)==record['public_sha256'],'Public artifact hash changed')
    truth={q['id']:q for q in manifest['queries']};categories=sorted(r['category_id'] for r in manifest['references']);rows={};by_alias={};ious=0
    audit.require(len(truth)==288 and len(categories)==12,'Wrong cohort')
    for alias,record in result['score_files'].items():
        path=b/record['path'];audit.require(sha(path)==record['sha256'],'Original prediction log hash differs')
        records=[json.loads(line) for line in path.read_text().splitlines()];rows[alias]=records;by_alias[alias]={r['id']:r for r in records}
        audit.require(len(records)==288 and set(by_alias[alias])==set(truth),'Prediction roster differs')
        for row in records:
            q=truth[row['id']]
            for key in ['category_id','name','split','present','image_id']:audit.require(row[key]==q[key],'Query binding differs')
            actual=[audit.iou(row['bbox_xyxy'],q['gt_boxes_xyxy']),max((audit.iou(box,q['gt_boxes_xyxy']) for box in row['proposals']),default=0)]
            audit.compare_numeric(actual,[row['iou'],row['proposal_best_iou']]);ious+=2
    for identity in truth:
        for first,second in [('old_cnn','native_patch'),('old_cnn_grid','native_patch_grid')]:
            audit.require(by_alias[first][identity]['proposals']==by_alias[second][identity]['proposals'],'Matched proposal rosters differ')
    metric_checks=interval_checks=0
    for mode,saved in result['modes'].items():
        cuts=audit.TRANSFERRED if mode=='transferred' else {a:audit.dev_cutoff(values) for a,values in rows.items()}
        metric_checks+=audit.compare_numeric(cuts,saved['thresholds'])
        for alias,records in rows.items():
            for split in ['dev','test']:
                measured=audit.metric([r for r in records if r['split']==split],cuts[alias])
                metric_checks+=audit.compare_numeric(measured,saved['methods'][alias][split])
                for category in categories:
                    measured=audit.metric([r for r in records if r['split']==split and r['category_id']==category],cuts[alias])
                    metric_checks+=audit.compare_numeric(measured,saved['per_category'][str(category)][alias][split])
        methods,differences=audit.bootstrap(rows,cuts,categories)
        interval_checks+=audit.compare_numeric(methods,saved['bootstrap']['methods'])
        interval_checks+=audit.compare_numeric(differences,saved['bootstrap']['localized_recall_differences'])
    print(json.dumps({'verified':True,'prediction_rows':1152,'iou_checks':ious,'metric_numbers':metric_checks,'bootstrap_numbers':interval_checks,
        'scope':'Saved-output arithmetic only. Zero-success bootstrap intervals are degenerate, not population-equivalence guarantees.'},indent=2))
if __name__=='__main__':main()
