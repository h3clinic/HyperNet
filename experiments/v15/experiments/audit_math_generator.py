"""Pretraining renderer QA: parameters, native rasters, warp parity and timing."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet import math_curriculum as reference,math_curriculum_torch as backend

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def cpu(value):return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else value
def summary(values):
    a=np.asarray(values,dtype=float);return dict(min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),max=float(a.max()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--families',type=int,default=4096)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/math_generator_qa.json');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing renderer QA evidence')
    torch.set_num_threads(4);device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    paths=[ROOT/'hypernet/math_curriculum.py',ROOT/'hypernet/math_curriculum_torch.py',Path(__file__)]
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    assert backend.REFERENCE_SHA256==sources['hypernet/math_curriculum.py']
    # Fixed mathematical IDs, no model outcome or real-photo selection.
    ids=np.r_[np.arange(32),reference.FAMILY_COUNT+np.arange(32),2**31-1-np.arange(32)]
    a=reference.sample_batch(ids,1515,64);b=backend.sample_batch(ids,1515,64,device=device)
    parity={}
    for name in ('view_a','view_b','mask_a','mask_b','canonical_raster','canonical_mask','detail_a','detail_b','grid_a_to_b','grid_b_to_a'):
        x=a[name].astype(np.float64);y=cpu(b[name]).astype(np.float64);error=np.abs(x-y)
        parity[name]=dict(max_abs=float(error.max()),mean_abs=float(error.mean()),different=int((error>0).sum()),elements=error.size,
                          fraction_error_over_one=float(np.mean(error>1)))
    assert parity['grid_a_to_b']['max_abs']<=2e-6 and parity['grid_b_to_a']['max_abs']<=2e-6
    assert all(parity[name]['different']/parity[name]['elements']<=1e-5 for name in ('mask_a','mask_b','canonical_mask'))
    assert all(parity[name]['fraction_error_over_one']<=1e-5 for name in ('view_a','view_b','canonical_raster'))
    del a,b
    geometry_hashes=set();mask_hashes=set();parameter_hashes=set();rgb_hashes=set();areas=[];anchors=[];counts=[0]*8
    started=time.perf_counter()
    for start in range(0,args.families,64):
        ids=np.arange(start,min(args.families,start+64));batch=backend.sample_batch(ids,1515,64,device=device)
        canonical=cpu(batch['canonical_raster']);masks=cpu(batch['canonical_mask']);rgb=cpu(batch['view_a'])
        area=cpu(batch['mask_a']).sum((1,2));detail=cpu(batch['detail_a']&batch['valid_a_to_b']).sum((1,2))
        parameters=reference.family_parameters(ids)
        for i,family in enumerate(ids):
            geometry_hashes.add(digest(canonical[i]));mask_hashes.add(digest(masks[i]));rgb_hashes.add(digest(rgb[i]))
            h=hashlib.sha256()
            for key in sorted(parameters):
                if key!='family_id':h.update(key.encode());h.update(np.asarray(parameters[key][i]).tobytes())
            parameter_hashes.add(h.hexdigest());counts[int(family)%8]+=1
        areas.extend(area.tolist());anchors.extend(detail.tolist())
    census_seconds=time.perf_counter()-started
    timings=[]
    for size,batch_size in ((64,32),(64,64),(64,128),(96,64)):
        ids=np.arange(batch_size)+50000
        backend.sample_batch(ids,1515,size,device=device)
        if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter()
        for repeat in range(5):batch=backend.sample_batch(ids+repeat*batch_size,1515,size,device=device)
        if device.type=='cuda':torch.cuda.synchronize()
        seconds=(time.perf_counter()-begin)/5
        timings.append(dict(native_size=size,batch_families=batch_size,seconds_per_batch=seconds,
            pairs_per_second=batch_size/seconds,rendered_views_per_second=2*batch_size/seconds,
            estimated_million_pair_renderer_seconds=reference.FAMILY_COUNT/batch_size*seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None))
    assert all(sha(ROOT/name)==value for name,value in sources.items())
    record=dict(valid=True,scope='Fixed preview/QA families only; zero samples counted as training consumption.',
        source_sha256=sources,device=str(device),torch_version=torch.__version__,numpy_version=np.__version__,
        gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        native_size=64,advertised_training_parameter_ids=reference.FAMILY_COUNT,
        renderer_parameter_namespace=reference.PARAMETER_NAMESPACE_SIZE,parity_ids=96,parity=parity,
        audited_parameter_ids=args.families,distinct_parameter_signatures=len(parameter_hashes),
        distinct_canonical_mask_material_rasters=len(geometry_hashes),distinct_canonical_masks=len(mask_hashes),
        distinct_firstview_rgb=len(rgb_hashes),geometry_modes={name:counts[i] for i,name in enumerate(reference.GEOMETRY_MODES)},
        target_foreground_pixels=summary(areas),valid_detail_anchor_pixels=summary(anchors),
        census_seconds=census_seconds,timings=timings,
        limitations=['A million parameter IDs is not a claim that a million finite-resolution masks or materials are unique.',
            'The full training stream must separately count actually consumed pair IDs, raw rendered views and canonical raster hashes.',
            'Continuous affine correspondence is exact; raster sampling can erase subpixel structures and boundary interpolation is not exact.',
            'Foreground correspondence masks exclude targetout-of-frame points; training should also check sampled targetforeground.',
            'NumPy/Torch/CUDA arithmetic may differ at rare threshold boundaries; measured parity is reported rather than assumed bit identity.',
            'These mathematical programs have no naturalcategory names or imported assets.'])
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('valid','device','audited_parameter_ids','distinct_parameter_signatures','distinct_canonical_mask_material_rasters','distinct_canonical_masks','census_seconds','timings')},indent=2))

if __name__=='__main__':main()
