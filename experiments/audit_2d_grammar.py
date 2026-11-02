"""Audit the full4096-program canonical2D grammar before dataset freezing."""
from pathlib import Path
import hashlib,json,sys,time
from collections import defaultdict
import numpy as np
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.mathematical_shapes2d import FAMILY_COUNT,canonical_mask,family_spec

start=time.perf_counter();by_pixels=defaultdict(list);by_rotation=defaultdict(list);same_micro=[];disconnected=[];area_deltas=[]
for family in range(FAMILY_COUNT):
    masks=[canonical_mask(family,variant,96) for variant in (0,1)]
    for variant,mask in enumerate(masks):
        if label(mask,structure=np.ones((3,3)))[1]!=1:disconnected.append([family,variant])
        key=hashlib.sha256(mask.tobytes()).hexdigest()
        by_pixels[key].append([family,variant])
        rotation=min(hashlib.sha256(np.rot90(mask,k).tobytes()).hexdigest() for k in range(4))
        by_rotation[rotation].append([family,variant])
    if np.array_equal(*masks):same_micro.append(family)
    area_deltas.append(abs(int(masks[0].sum())-int(masks[1].sum()))/max(int((masks[0]|masks[1]).sum()),1))
    if (family+1)%512==0:print(f'Audited{family+1}/4096families',flush=True)
duplicates=[rows for rows in by_pixels.values() if len(rows)>1]
rotations=[rows for rows in by_rotation.values() if len(rows)>1]
result={'passed':not duplicates and not rotations and not same_micro and not disconnected,'family_programs':4096,'microstates_checked':8192,
    'canonical_resolution':96,'duplicate_canonical_masks':duplicates,'duplicate_up_to_right_angle_rotation':rotations,
    'microvariants_identical':same_micro,'unique_program_ids':len({family_spec(i)['canonical_program_id'] for i in range(FAMILY_COUNT)}),
    'disconnected_canonical_masks':disconnected,'max_paired_foreground_area_relative_difference':max(area_deltas),
    'renderer_sha256':hashlib.sha256((ROOT/'hypernet/mathematical_shapes2d.py').read_bytes()).hexdigest(),
    'elapsed_seconds':time.perf_counter()-start,
    'scope':'Distinct canonical raster masks and discrete program definitions; not a proof of semantic classes or equivalence under every continuous affine transform'}
(ROOT/'experiments/canonical_2d_grammar_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))
