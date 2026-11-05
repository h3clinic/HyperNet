"""Independent direct-negative-order verification of descriptive DEV points."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
result=OUT/'matched_operating_point_development.json'
d=json.loads(result.read_text(encoding='utf-8'))
sources=d['source_files'];rows=json.loads(Path(sources['records']['path']).read_text(encoding='utf-8'))
sp=json.loads(Path(sources['splits']['path']).read_text(encoding='utf-8'));ix=np.array(sp['indices']['development'])
labels=np.array([rows[i]['label'] for i in ix]);checks=[]
with np.load(sources['once_scores']['path']) as once,np.load(sources['dino_scores']['path']) as dino:
    for row in d['targets']:
        y=labels==row['target']
        s=once['scores__'+row['target']][ix] if row['model'].startswith('OnceNet') else dino['late_mean4__'+row['target']][ix]
        neg=np.sort(s[~y]);positive=s[y]
        # Pair-counting AUROC, independent from ROC integration.
        left=np.searchsorted(neg,positive,'left');right=np.searchsorted(neg,positive,'right')
        rank_auc=float(np.mean((left+.5*(right-left))/len(neg)))
        assert abs(rank_auc-row['metrics']['auroc'])<1e-12
        for point in row['operating_points']:
            budget=int(np.floor(point['fpr_budget']*len(neg)))
            threshold=neg[-budget-1]
            pred=s>threshold
            tp=int((pred&y).sum());fp=int((pred&~y).sum())
            assert tp==point['empirical_tp'] and fp==point['empirical_fp']
            checks.append({'model':row['model'],'target':row['target'],'fpr_budget':point['fpr_budget'],'tp':tp,'fp':fp,'matched':True})
audit={'passed':True,'analysis_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
    'method':'Direct strict cutoff above next disallowed negative order statistic; direct positive-negative rank-pair AUROC. No sklearn, inference, fitting, policy export or reserved reads.',
    'checks':checks,'paired_false_alarm_counts_equal_all_targets_and_budgets':all(x['achieved_fp_equal'] for x in d['achieved_false_alarm_count_comparison'])}
(OUT/'matched_operating_point_development_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(f'Independent checks passed: {len(checks)} operating points and 8 pair-count AUROCs.')
