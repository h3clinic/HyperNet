"""Exact held-out raster overlap audit, separate from parameter-ID separation."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.math_curriculum_torch import sample_batch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier audits')
    admission=json.loads((args.run/'admission.json').read_text());completed=json.loads((args.run/'completed.json').read_text())
    if not completed.get('completed') or completed['mode']!='train':raise ValueError('A completed full run is required')
    columns=completed['counts']['digest_columns']
    if columns!=['canonical_mask','canonical_material_raster','view_a_rgb','view_b_rgb']:raise ValueError('Unknown ledger layout')
    matrices=[];id_arrays=[]
    for record in completed['counts']['shards']:
        path=args.run/'ledger'/record['path']
        if sha(path)!=record['sha256']:raise ValueError('Ledger changed')
        with np.load(path,allow_pickle=False) as data:
            matrices.append(data['sha256']);id_arrays.append(data['family_ids'])
    training=np.concatenate(matrices);family_ids=np.concatenate(id_arrays)
    if not np.array_equal(family_ids,np.arange(1048576)):raise ValueError('Training ID roster is not the declared full stream')
    def packed(values):return np.ascontiguousarray(values).reshape(-1,32).view('V32').reshape(-1)
    keys=[np.sort(packed(training[:,index])) for index in (0,1)]
    rgb_keys=np.sort(packed(training[:,2:]))
    begin,end=admission['validation_family_interval'];ids=np.arange(begin,end,dtype=np.int64)
    raw=sample_batch(ids,admission['renderer_seed']+1,size=admission['side'],device=admission['device'])
    arrays={key:raw[key].cpu().numpy() for key in ('canonical_mask','canonical_raster','view_a','view_b')}
    hashes=[]
    for index in range(len(ids)):
        hashes.append(b''.join(hashlib.sha256(np.ascontiguousarray(arrays[key][index]).tobytes()).digest()
            for key in ('canonical_mask','canonical_raster','view_a','view_b')))
    validation=np.frombuffer(b''.join(hashes),np.uint8).reshape(-1,4,32)
    def membership(needles,haystack):
        values=packed(needles);positions=np.searchsorted(haystack,values);valid=positions<len(haystack)
        result=np.zeros(len(values),bool);result[valid]=haystack[positions[valid]]==values[valid]
        return result
    mask_matches=membership(validation[:,0],keys[0]);material_matches=membership(validation[:,1],keys[1])
    rgb_matches=membership(validation[:,2:],rgb_keys).reshape(-1,2)
    report={'completed':True,'source_sha256':sha(__file__),'training_completion_sha256':sha(args.run/'completed.json'),
        'admission_sha256':sha(args.run/'admission.json'),'validation_family_interval':[begin,end],
        'validation_families':len(ids),'parameter_id_overlap':int(np.isin(ids,family_ids).sum()),
        'canonical_mask_overlap_families':int(mask_matches.sum()),'canonical_material_raster_overlap_families':int(material_matches.sum()),
        'rendered_rgb_overlap_views':int(rgb_matches.sum()),'canonical_mask_overlap_family_ids':ids[mask_matches].tolist(),
        'canonical_material_overlap_family_ids':ids[material_matches].tolist(),
        'unique_validation_masks':len(np.unique(packed(validation[:,0]))),
        'unique_validation_material_rasters':len(np.unique(packed(validation[:,1]))),
        'unique_validation_rgb_views':len(np.unique(packed(validation[:,2:]))),
        'validation_sha256_matrix_sha256':hashlib.sha256(validation.tobytes()).hexdigest(),
        'interpretation':'All functional modes are shared with training. Parameter-ID separation does not imply novel geometry; exact raster overlap is reported separately. Exact-hash nonoverlap does not exclude near duplicates.',
        'natural_images_read':0,'model_inference_calls':0,'optimizer_steps':0}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
