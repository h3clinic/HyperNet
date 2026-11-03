"""Audit the exact prespecified synthetic diagnostic inputs without model calls."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from scipy.ndimage import label
import torch
from hypernet.variation_curriculum import catalog,render,hard_negative,component_count,partition
from diagnose_relations import PLAN

torch.set_num_threads(4)
codes=np.sort(catalog(True));n=len(codes)
seeds=np.arange(n,dtype=np.uint64)+PLAN['reference_seed_base']
query_codes=np.tile(codes,2)
query_seeds=np.concatenate([seeds+offset for offset in PLAN['query_seed_offsets']])
negative_codes=np.asarray([hard_negative(int(code))[0]for code in query_codes],np.uint32)
groups={'reference':(codes,seeds),'query':(query_codes,query_seeds),'negative':(negative_codes,query_seeds)}
result={'scope':'Exact prespecified synthetic diagnostic inputs; CPU rendering; no model or real images',
        'plan_sha256':hashlib.sha256((ROOT/'experiments/synthetic_diagnosis_plan.json').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256((ROOT/'hypernet/variation_curriculum.py').read_bytes()).hexdigest(),
        'groups':{}}
for name,(group,group_seeds)in groups.items():
    failures=[];hashes=[];pixels=[]
    for start in range(0,len(group),32):
        batch=render(group[start:start+32],group_seeds[start:start+32],96,'cpu')
        for index,mask in enumerate(batch['mask'].numpy()):
            code=int(group[start+index]);expected=component_count(code)
            actual=label(mask,structure=np.ones((3,3),np.uint8))[1]
            if actual!=expected:failures.append({'index':start+index,'code':code,'expected':expected,'actual':actual})
            hashes.append(hashlib.sha256(mask.tobytes()).hexdigest());pixels.append(int(mask.sum()))
    result['groups'][name]={'instances':len(group),'component_failures':failures,'unique_masks':len(set(hashes)),
                           'foreground_pixels_min':min(pixels),'foreground_pixels_max':max(pixels)}
result['heldout_partition_valid']=bool(np.all(partition(codes)==0))
result['passed']=result['heldout_partition_valid']and all(not row['component_failures']for row in result['groups'].values())
path=ROOT/'experiments/synthetic_diagnostic_input_audit.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
sys.exit(0 if result['passed']else 1)
