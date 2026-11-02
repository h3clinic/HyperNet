"""Verify published 2D metrics, localization and paired intervals without weights or pixels."""
from pathlib import Path
import argparse, importlib.util, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_2d_audit',ROOT/'scripts/audit_2d_detector.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def require(x,msg):
    if not x:raise ValueError(msg)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('benchmark',type=Path,nargs='?',default=ROOT/'benchmarks/v0.11');args=parser.parse_args();b=args.benchmark
    aggregate=read(b/'aggregate_metrics.json');analysis=read(b/'paired_analysis.json');manifest=read(b/'heldout_manifest.json')
    truth={q['id']:q for e in manifest['episodes'] for q in e['queries']}
    arms=['coarse_match','micro_aware'];rows={};cuts={};metric_values=0
    if (b/'analytic_detail_bypass_results.json').exists():
        arms.append('analytic_detail_bypass')
        aggregate['arms']['analytic_detail_bypass']=read(b/'analytic_detail_bypass_results.json')
    for arm in arms:
        records=[json.loads(l) for l in (b/f'{arm}_predictions.jsonl').read_text().splitlines() if l];rows[arm]=records
        require(len(records)==1024 and len({r['id'] for r in records})==1024,'Expected all heldout rows')
        for r in records:
            q=truth[r['id']];require(r['present']==q['present'],'Label mismatch')
            require(abs(audit.overlap(r['bbox_xyxy'],q['target_bbox_xyxy'])-r['iou'])<1e-12,'Selected box IoU mismatch')
            best=max([audit.overlap(box,q['target_bbox_xyxy']) for box in r['proposals']],default=0)
            require(abs(best-r['proposal_best_iou'])<1e-12,'Proposal IoU mismatch')
        cutoff=audit.dev_cutoff([r for r in records if r['split']=='dev']);cuts[arm]=cutoff
        for split in ['dev','test']:
            measured=audit.independent_metrics([r for r in records if r['split']==split],cutoff)
            expected=aggregate['arms'][arm]['metrics'][split]
            require(set(measured)==set(expected),'Metric schema differs')
            for key,v in measured.items():
                require(abs(v-expected[key])<1e-12,f'Metric differs: {arm}/{split}/{key}');metric_values+=1
    families=analysis['test_families'];require(len(families)==128,'128 family clusters required')
    arrays={};keys=['localized_recall','recall','false_positive_rate','balanced_accuracy','paired_ranking_accuracy']
    for arm in arms:
        a={k:[] for k in keys}
        for family in families:
            selected=[r for r in rows[arm] if r['split']=='test' and r['family_id']==family]
            pos=[r for r in selected if r['present']];neg=[r for r in selected if not r['present']]
            require(len(pos)==len(neg)==2,'Counterbalance missing')
            tpr=sum(r['score']>=cuts[arm] for r in pos)/2;fpr=sum(r['score']>=cuts[arm] for r in neg)/2
            a['localized_recall'].append(sum(r['score']>=cuts[arm] and r['iou']>=.5 for r in pos)/2)
            a['recall'].append(tpr);a['false_positive_rate'].append(fpr);a['balanced_accuracy'].append((tpr+1-fpr)/2)
            ranking=[]
            for p in pos:
                n=next(r for r in neg if r['episode_id']==p['episode_id'])
                ranking.append(float(p['score']>n['score'])+.5*float(p['score']==n['score']))
            a['paired_ranking_accuracy'].append(sum(ranking)/2)
        arrays[arm]={k:np.array(v) for k,v in a.items()}
    indices=np.random.default_rng(1811).integers(0,128,(10000,128))
    for key in keys:
        delta=arrays[arms[1]][key]-arrays[arms[0]][key];lo,hi=np.quantile(delta[indices].mean(1),[.025,.975])
        expected=analysis['contrasts_micro_aware_minus_coarse_match'][key]
        for name,v in [('difference',delta.mean()),('low',lo),('high',hi)]:require(abs(v-expected[name])<1e-12,'Paired interval differs')
    descriptive=b/'type_present_localization_diagnostic.json'
    if descriptive.exists():
        for arm,m in read(descriptive)['methods'].items():
            selected=[r for r in rows[arm] if r['split']=='test']
            correct=sum(audit.overlap(r['bbox_xyxy'],truth[r['id']]['candidate_bbox_xyxy'])>=.5 for r in selected)
            require(correct==m['correct_candidate_box'] and len(selected)==m['queries'],'Post hoc forced-choice count differs')
            require(abs(correct/len(selected)-m['fraction'])<1e-12,'Post hoc forced-choice fraction differs')
    print(json.dumps({'verified':True,'prediction_rows':1024*len(arms),'metric_values':metric_values,'paired_contrast_values':15,
        'post_hoc_type_present_counts_verified':descriptive.exists(),
        'scope':'Arithmetic verification of saved outputs; reproducing inference requires locally generated data and retrained checkpoints.'},indent=2))
if __name__=='__main__':main()
