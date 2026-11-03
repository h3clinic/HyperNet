"""Fixed-threshold paired photo-cluster confidence intervals; no TEST selection."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_coco_v17 import PLAN,DATA,verify_plan,read,sha,write_new
from hypernet.detection_metrics import truth_arrays

def bootstrap_outcomes(scores,overlaps,present,valid,cutoff,weights):
    pos=(present&valid).astype(float); neg=(~present&valid).astype(float)
    hits=(scores>=cutoff)&(overlaps>=.5)&present&valid
    false=(scores>=cutoff)&~present&valid
    positive_n=weights@pos; negative_n=weights@neg
    if np.any(positive_n==0) or np.any(negative_n==0):
        raise ValueError('A photo bootstrap draw has an empty category; explicitly redesign CI handling, do not silently drop it')
    return np.stack(
        [((weights@hits)/positive_n).mean(1),((weights@false)/negative_n).mean(1)],1)

def paired_report(arrays,cutoffs,present,valid,replicates=5000,seed=170031):
    rng=np.random.default_rng(seed); n=len(present)
    # Multinomial counts resample entire photos, retaining all targets/methods.
    weights=rng.multinomial(n,np.full(n,1/n),size=replicates).astype(float)
    samples={name:bootstrap_outcomes(a['scores'],a['overlaps'],present,valid,cutoffs[name],weights)
             for name,a in arrays.items()}
    observed={name:bootstrap_outcomes(a['scores'],a['overlaps'],present,valid,cutoffs[name],np.ones((1,n)))[0]
              for name,a in arrays.items()}
    names=['macro_localized_recall','macro_false_positive_rate']; results={}
    for name,sample in samples.items():
        results[name]={metric:{'estimate':float(observed[name][j]),'ci95':np.quantile(sample[:,j],[.025,.975]).tolist()}
                       for j,metric in enumerate(names)}
    comparisons={}
    pairs=[(name,'owlvit_base_patch32',name+'_minus_owlvit')for name in arrays if name!='owlvit_base_patch32']
    if 'v17' in arrays:
        pairs += [('v17',control,'v17_minus_'+control)for control in ('v17_no_geometry','random','raw_template')if control in arrays]
    for left,right,label in pairs:
        delta=samples[left]-samples[right]; point=observed[left]-observed[right]
        comparisons[label]={metric:{'estimate':float(point[j]),'ci95':np.quantile(delta[:,j],[.025,.975]).tolist()}
                                          for j,metric in enumerate(names)}
        comparisons[label]['observed_win_gate']=bool(point[0]>0 and point[1]<=0)
        comparisons[label]['gate_definition']='Higher localized recall and no higher observed macro false-positive rate; descriptive, not a universal superiority claim.'
    return {'methods':results,'paired_differences':comparisons,'bootstrap':{'unit':'query photo, all12 targets carried together',
            'replicates':replicates,'seed':seed,'interval':'95% percentile','fixed':'Single support per category and DEV-calibrated cutoffs; no support/calibration uncertainty resampling'}}

def main(args):
    plan=verify_plan(); rows=plan[args.split]
    if args.split=='test':
        acquisition=read(DATA/'acquisition.json'); keep=set(acquisition['eligible_image_ids'])
        rows=[r for r in rows if r['image_id'] in keep]
    present,valid=truth_arrays(rows,plan['targets']); arrays={}; bindings={}; calibrations={}
    for method in args.methods:
        directory=ROOT/'experiments'/f'{method}_{args.split}_v2'
        path=directory/'arrays.npz'; result=read(directory/'result.json')
        if sha(path)!=result['score_file_sha256'] or result['data_plan_sha256']!=sha(PLAN): raise ValueError('Score provenance differs')
        arr=dict(np.load(path)); arrays[method]=arr
        if arr['image_ids'].tolist()!=[r['image_id'] for r in rows]: raise ValueError('Photo pairing differs')
        if arr['category_ids'].tolist()!=[r['category_id'] for r in plan['targets']]: raise ValueError('Target pairing differs')
        calibration=ROOT/'experiments'/f'{method}_dev_v2/calibration.json'
        calibrations[method]=read(calibration); bindings[str(path)]=sha(path); bindings[str(calibration)]=sha(calibration)
    if 'owlvit_base_patch32' not in arrays: raise ValueError('Direct OWL comparator required')
    result={'data_plan_sha256':sha(PLAN),'split':args.split,'input_bindings':bindings,'photos':len(rows),
            'nominal_target_photo_pairs':int(present.size),'valid_target_photo_pairs':int(valid.sum()),
            'script_sha256':sha(__file__),'standard_coco_AP':False,'operating_points':{}}
    for operating_point in ('primary_5pct','secondary_10_per_10000'):
        cutoffs={name:c[operating_point]['cutoff'] for name,c in calibrations.items()}
        result['operating_points'][operating_point]=paired_report(arrays,cutoffs,present,valid,args.replicates)
    write_new(ROOT/'experiments'/f'paired_comparison_{args.split}_v2.json',result)
    print(json.dumps(result['operating_points']['primary_5pct']['paired_differences']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['dev','test'],default='test')
    p.add_argument('--methods',nargs='+',default=['v17','v17_no_geometry','random','owlvit_base_patch32','raw_template'])
    p.add_argument('--replicates',type=int,default=5000);main(p.parse_args())
