"""Paired heldout-structural-type bootstrap, specified before diagnosis results."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARMS=['random','primary_final_4096','no_relation_aux_final_4096','raw_gray_cosine']
PLAN={'scope':'Synthetic graph grammar only; no natural-image or general semantic inference',
      'replicates':5000,'seed':16022026,'cluster':'One structural type, its reference, and both query/negative instances',
      'interval':'Two-sided95% percentile, NumPy linear quantiles',
      'metrics':['top1_accuracy','nuisance_matched_pair_accuracy'],
      'comparisons':['primary_final_4096 minus random','primary_final_4096 minus no_relation_aux_final_4096','primary_final_4096 minus raw_gray_cosine'],
      'checkpoint_selection':False}


def bootstrap(values,indices):
    values=np.asarray(values,dtype=np.float64)
    samples=values[indices].mean(1)
    return {'estimate':float(values.mean()),'ci95':[float(x)for x in np.quantile(samples,[.025,.975],method='linear')]}


def analyze():
    directory=ROOT/'experiments/synthetic_final_diagnosis'
    values={}
    for arm in ARMS:
        with np.load(directory/f'{arm}_scores.npz')as row:
            truth=row['truth'];prediction=row['prediction']
            if len(truth)!=1024 or not np.array_equal(truth,np.tile(np.arange(512),2)):
                raise ValueError('Unexpected fixed diagnostic roster')
            values[arm]={'top1_accuracy':(prediction==truth).reshape(2,512).mean(0),
                         'nuisance_matched_pair_accuracy':(row['positive_score']>row['negative_score']).reshape(2,512).mean(0)}
    rng=np.random.default_rng(PLAN['seed']);indices=rng.integers(0,512,(PLAN['replicates'],512))
    result={'plan':PLAN,'plan_sha256':hashlib.sha256((ROOT/'experiments/synthetic_uncertainty_plan.json').read_bytes()).hexdigest(),
            'clusters':512,'queries':1024,'arms':{},'paired_differences':{}}
    for arm in ARMS:result['arms'][arm]={metric:bootstrap(values[arm][metric],indices)for metric in PLAN['metrics']}
    for other in ('random','no_relation_aux_final_4096','raw_gray_cosine'):
        result['paired_differences'][f'primary_final_4096_minus_{other}']={
            metric:bootstrap(values['primary_final_4096'][metric]-values[other][metric],indices)for metric in PLAN['metrics']}
    result['score_hashes']={arm:hashlib.sha256((directory/f'{arm}_scores.npz').read_bytes()).hexdigest()for arm in ARMS}
    result['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path=directory/'paired_type_bootstrap.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    if '--write-plan'in sys.argv:
        path=ROOT/'experiments/synthetic_uncertainty_plan.json'
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(PLAN,indent=2),encoding='utf-8');print(hashlib.sha256(path.read_bytes()).hexdigest())
    else:analyze()
