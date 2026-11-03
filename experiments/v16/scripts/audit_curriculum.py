"""Pretraining-only raster topology and independent-instance audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.ndimage import label
import torch
from hypernet.variation_curriculum import render,catalog,labels,component_count,hard_negative,partition


def audit(count=2048,size=96,device='cuda'):
    torch.set_num_threads(4)
    codes=catalog()[:count]
    failures=[];identical_rgb=0;identical_mask=0;negative_kinds={};masks=[]
    for start in range(0,len(codes),64):
        group=codes[start:start+64];seeds=np.arange(start,start+len(group),dtype=np.uint64)+800000000
        a=render(group,seeds,size,device);b=render(group,seeds+(1<<40),size,device)
        for batch,name in ((a,'a'),(b,'b')):
            mask=batch['mask'].cpu().numpy()
            for code,raster in zip(group,mask):
                actual=label(raster,structure=np.ones((3,3),np.uint8))[1];expected=component_count(int(code))
                if actual!=expected:failures.append({'code':int(code),'instance':name,'expected':expected,'actual':actual})
                masks.append(hashlib.sha256(raster.tobytes()).hexdigest())
        identical_rgb+=int((a['rgb']==b['rgb']).flatten(1).all(1).sum())
        identical_mask+=int((a['mask']==b['mask']).flatten(1).all(1).sum())
    matched_failures=[]
    for code in codes:
        other,kind=hard_negative(int(code));negative_kinds[kind]=negative_kinds.get(kind,0)+1
        e,d=labels([code,other])
        valid=other!=int(code) and int(partition([other])[0])!=0 and (e[0,:12].sum()==e[1,:12].sum()) and (e[0,12:].sum()==e[1,12:].sum())
        if kind!='edge_count_orientation_matched_only':valid=valid and np.array_equal(np.sort(d[0]),np.sort(d[1])) and component_count(int(code))==component_count(other)
        if not valid:matched_failures.append(int(code))
    result={'scope':'Pretraining synthetic QA; no natural images or model selection', 'size':size,'type_count':len(codes),
            'instances':2*len(codes),'component_connectivity':8,'raster_component_failures':failures,
            'independent_positive_identical_rgb':identical_rgb,'independent_positive_identical_masks':identical_mask,
            'unique_raster_masks':len(set(masks)),'negative_kind_counts':negative_kinds,'negative_statistic_failures':matched_failures,
            'train_validation_code_overlap':len(set(map(int,catalog()))&set(map(int,catalog(True)))),
            'source_sha256':hashlib.sha256((ROOT/'hypernet/variation_curriculum.py').read_bytes()).hexdigest()}
    result['passed']=not failures and not matched_failures and not identical_rgb and not identical_mask and not result['train_validation_code_overlap']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=2048);p.add_argument('--size',type=int,default=96);p.add_argument('--device',default='cuda');p.add_argument('--out',default='experiments/curriculum_pretraining_qa.json')
    a=p.parse_args();r=audit(a.count,a.size,a.device);path=ROOT/a.out;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2));sys.exit(0 if r['passed']else 1)
