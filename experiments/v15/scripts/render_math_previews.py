"""Candidate preview rendering only; never increments optimizer counters."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)
def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',type=int,default=64);parser.add_argument('--seed',type=int,default=15031);args=parser.parse_args()
    output=ROOT/'assets/math_preview';output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'hypernet/math_curriculum.py';source_bytes=source.read_bytes();source_sha=hashlib.sha256(source_bytes).hexdigest()
    namespace={'__name__':'math_preview_snapshot','__file__':str(source)}
    exec(compile(source_bytes,str(source),'exec'),namespace);render_pair=namespace['render_pair']
    roster=list(range(16));version=output/source_sha[:12];version.mkdir(exist_ok=True)
    (version/'renderer_snapshot.py').write_bytes(source_bytes)
    if (output/'manifest.json').exists():
        prior=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        archive=output/('manifest_'+prior['source_sha256'][:12]+'.json')
        if not archive.exists():write(archive,prior)
    manifest={'format':'hypernet.math_preview.v15','status':'candidate_preview_only_not_training_count','source_sha256':source_sha,
        'frozen_training_source':False,'preview_seed':args.seed,'native_size':args.size,'family_roster':roster,'pairs':[],
        'annotation_note':'Coordinates are analytic continuous correspondences at source pixel centers, not learned matches. Raster aliasing can change the sampled target material.'}
    started=time.perf_counter();write(output/'manifest.json',manifest)
    for family in roster:
        sample=render_pair(family,args.seed,args.size);record={'id':f'candidate_{source_sha[:12]}_{family}_{args.size}','family_id':family,
            'status':'candidate_preview_only_not_training_count','source_sha256':source_sha,'native_size':args.size,'pair_seed':args.seed,
            'parameters':plain(sample['parameters']),'correspondences':[]}
        for side in ('a','b'):
            for prefix,key in [('view','view'),('mask','mask'),('detail','detail')]:
                name=f'f{family:07d}_{prefix}_{side}_{args.size}.png';pixels=sample[key+'_'+side]
                if pixels.dtype==bool:pixels=pixels.astype(np.uint8)*255
                Image.fromarray(pixels).save(version/name);record[prefix+'_'+side]=(version/name).relative_to(ROOT).as_posix()
        points=np.argwhere(sample['valid_a_to_b']&sample['detail_a'])
        if len(points)==0:points=np.argwhere(sample['valid_a_to_b'])
        chosen=np.linspace(0,len(points)-1,min(32,len(points)),dtype=int) if len(points) else []
        for index in chosen:
            y,x=map(int,points[index]);target=(sample['grid_a_to_b'][y,x]+1)*args.size/2-.5
            record['correspondences'].append({'a':[x,y],'b':target.tolist(),'valid':True})
        record['canonical_to_view_a']=plain(sample['canonical_to_view_a']);record['canonical_to_view_b']=plain(sample['canonical_to_view_b'])
        manifest['pairs'].append(record);manifest['elapsed_seconds']=time.perf_counter()-started
        manifest['source_still_matches']=sha(ROOT/'hypernet/math_curriculum.py')==source_sha
        write(output/'manifest.json',manifest);print(json.dumps({'preview_pairs_written':len(manifest['pairs']),'optimizer_consumption_increment':0}),flush=True)
    print(json.dumps({'completed_preview_pairs':len(roster),'seconds':time.perf_counter()-started,'source_sha256':source_sha}),flush=True)
    write(output/('manifest_'+source_sha[:12]+'.json'),manifest)

if __name__=='__main__':main()
