"""Check nonincident raster contacts; stronger than component-count agreement."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.ndimage import maximum_filter
import torch
from hypernet.variation_curriculum import catalog,render,EDGES

torch.set_num_threads(4)
size=96;codes=catalog()[:2048]
pixel=2*(torch.arange(size)+.5)/size-1
yy,xx=torch.meshgrid(pixel,pixel,indexing='ij');xx=xx[None];yy=yy[None]
pairs=[(i,j)for i in range(24)for j in range(i+1,24)if not set(EDGES[i])&set(EDGES[j])]
findings=[];instances=0;reconstruction_mismatch=0
for start in range(0,len(codes),32):
    group=codes[start:start+32]
    seeds=np.arange(start,start+len(group),dtype=np.uint64)+800000000
    for view,offset in (('a',0),('b',1<<40)):
        batch=render(group,seeds+offset,size,'cpu');node=batch['node_xy'];width=batch['stroke_radius'][:,None,None]
        masks=[]
        for e,(a,c)in enumerate(EDGES):
            ax=node[:,a,0,None,None];ay=node[:,a,1,None,None]
            dx=node[:,c,0,None,None]-ax;dy=node[:,c,1,None,None]-ay
            length=(dx.square()+dy.square()).sqrt().clamp_min(.1)
            t=(((xx-ax)*dx+(yy-ay)*dy)/length.square()).clamp(0,1)
            bow=batch['edge_bows'][:,e,None,None]*torch.sin(torch.pi*t)
            cx=ax+t*dx-bow*dy/length;cy=ay+t*dy+bow*dx/length
            mask=(xx-cx).square()+(yy-cy).square()<=width.square()
            for endpoint in (a,c):
                mask|=(xx-node[:,endpoint,0,None,None]).square()+(yy-node[:,endpoint,1,None,None]).square()<=(1.12*width).square()
            masks.append(mask&(batch['edges'][:,e,None,None]>0))
        all_masks=torch.stack(masks,1).numpy()
        reconstruction_mismatch+=int(np.any(np.any(all_masks,axis=1)!=batch['mask'].numpy(),axis=(1,2)).sum())
        for index,edges in enumerate(all_masks):
            dilated=maximum_filter(edges,size=(1,3,3),mode='constant')
            packed=np.packbits(edges,axis=None).reshape(24,-1)
            near=np.packbits(dilated,axis=None).reshape(24,-1)
            contact=[(a,c)for a,c in pairs if np.any(near[a]&packed[c])]
            if contact:findings.append({'code':int(group[index]),'view':view,'seed':int(seeds[index]+offset),'edge_pairs':contact})
        instances+=len(group)
result={'scope':'Post-launch CPU-only generator audit; no source/model changes or natural images',
        'method':'Reconstruct each present edge with its endpoint junction caps; test 8-connected pixel adjacency for every nonincident edge pair',
        'instances':instances,'image_side':size,'nonincident_contact_instances':len(findings),
        'contact_fraction':len(findings)/instances,'findings':findings,
        'full_mask_reconstruction_mismatches':reconstruction_mismatch,
        'source_sha256':hashlib.sha256((ROOT/'hypernet/variation_curriculum.py').read_bytes()).hexdigest(),
        'passed':not findings and not reconstruction_mismatch}
path=ROOT/'experiments/nonincident_raster_contact_audit.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));sys.exit(0 if result['passed']else 1)
