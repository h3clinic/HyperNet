"""One-reference image-guided evaluation; all 12 targets on each photo."""
from pathlib import Path
import argparse,json,sys,time,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; WORKSPACE=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
from prepare_coco_v17 import verify_plan,load_reference,load_dev_reference,load_test,sha,read,write_new,PLAN,DATA
from hypernet.detection_metrics import iou,calibration,summarize,truth_arrays,clip_box,COMMON_BOX_RULE

def run(args):
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    plan=verify_plan(); targets=plan['targets']; rows=plan[args.split]
    if args.split=='test':
        acquisition=read(DATA/'acquisition.json')
        if not acquisition['completed'] or acquisition['data_plan_sha256']!=sha(PLAN): raise ValueError('Missing completed gated acquisition')
        protocol=read(acquisition['protocol_path'])
        if sha(acquisition['protocol_path'])!=acquisition['protocol_sha256']: raise ValueError('Protocol changed')
        from prepare_coco_v17 import verify
        verify(protocol['behavior_files']); verify(protocol['checkpoint_files'])
        keep=set(acquisition['eligible_image_ids']); rows=[r for r in rows if r['image_id'] in keep]
    output=ROOT/'experiments'/f'{args.method}_{args.split}_v2'
    if output.exists(): raise FileExistsError('Preserve completed or partial evaluation: '+str(output))
    if args.method=='owlvit_base_patch32':
        from hypernet.owl_baseline import OwlBaseline
        model=OwlBaseline(WORKSPACE/'outputs/hypernet_published_compare/checkpoints/owlvit-base-patch32',args.device)
    elif args.method=='raw_template':
        from hypernet.raw_template import RawTemplate
        model=RawTemplate()
    else:
        if args.checkpoint is None: raise ValueError('HyperNet checkpoint required')
        from hypernet.local_matcher import FrozenLocalMatcher
        model=FrozenLocalMatcher(args.checkpoint,device=args.device,max_side=args.max_side,proposal_limit=100)
    if args.split=='test':
        dev_directory=ROOT/'experiments'/f'{args.method}_dev_v2'
        required=[Path(__file__),ROOT/'hypernet/detection_metrics.py',dev_directory/'result.json',dev_directory/'calibration.json']
        required += [ROOT/'hypernet'/name for name in ('owl_baseline.py','raw_template.py','local_matcher.py','local_relation_model.py')]
        for path in required:
            if protocol['behavior_files'].get(str(path.resolve()))!=sha(path):
                raise ValueError('Required TEST behavior/DEV provenance was not frozen: '+str(path))
        previous=read(dev_directory/'result.json')
        if previous['provenance']!=json.loads(json.dumps(model.provenance())):
            raise ValueError('TEST model/checkpoint/preprocessing differs from calibrated DEV model')
    output.mkdir(parents=True)
    refs={r['category_id']:load_reference(r) for r in plan['references']}
    if args.method=='owlvit_base_patch32' and args.split=='dev':
        reference=refs[targets[0]['category_id']]; query=load_dev_reference(rows[0])
        eq=model.equivalence(reference,query); eq.update({'query_image_id':rows[0]['image_id'],'reference_category_id':targets[0]['category_id']})
        write_new(output/'equivalence.json',eq); print(json.dumps({'equivalence':eq}),flush=True)
    banks=[model.enroll(refs[t['category_id']]) for t in targets]
    scores=np.zeros((len(rows),len(targets)),dtype=float); overlaps=np.zeros_like(scores); boxes=np.zeros((*scores.shape,4),dtype=float)
    start=time.monotonic()
    with (output/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for i,row in enumerate(rows):
            rgb=load_dev_reference(row) if args.split=='dev' else load_test(row,acquisition)
            encoded=model.encode_query(rgb) if hasattr(model,'encode_query') else rgb
            for j,(target,bank) in enumerate(zip(targets,banks)):
                proposals=model.predict(bank,encoded)
                if not proposals: raise ValueError('Finite top-one proposal required')
                top=dict(max(proposals,key=lambda r:r['score']))
                top['raw_box']=list(top['box']);top['box']=clip_box(top['raw_box'],rgb.shape[1],rgb.shape[0])
                scores[i,j]=float(top['score']); boxes[i,j]=top['box']
                if not np.isfinite([scores[i,j],*boxes[i,j]]).all(): raise ValueError('Nonfinite output')
                overlaps[i,j]=iou(boxes[i,j],row['boxes_by_target'][str(target['category_id'])])
                f.write(json.dumps({'image_id':row['image_id'],'category_id':target['category_id'],**top,'iou':overlaps[i,j]},allow_nan=False)+'\n')
            f.flush()
            if (i+1)%24==0 or i+1==len(rows):
                print(json.dumps({'method':args.method,'split':args.split,'photos':i+1,'total':len(rows),'seconds':round(time.monotonic()-start,2)}),flush=True)
    model.assert_unchanged()
    np.savez_compressed(output/'arrays.npz',scores=scores,overlaps=overlaps,boxes=boxes,image_ids=[r['image_id'] for r in rows],category_ids=[t['category_id'] for t in targets])
    if args.split=='dev':
        cutoffs=calibration(scores,overlaps,rows,targets); write_new(output/'calibration.json',cutoffs)
        summary={k:r['summary'] for k,r in cutoffs.items()}
    else:
        cutoffs=read(ROOT/'experiments'/f'{args.method}_dev_v2/calibration.json')
        present,valid=truth_arrays(rows,targets)
        summary={k:summarize(scores,overlaps,present,valid,r['cutoff'],targets) for k,r in cutoffs.items()}
    result={'method':args.method,'split':args.split,'data_plan_sha256':sha(PLAN),'score_file_sha256':sha(output/'arrays.npz'),
            'predictions_sha256':sha(output/'predictions.jsonl'),'seconds':time.monotonic()-start,'provenance':model.provenance(),
            'summary':summary,'common_box_rule':COMMON_BOX_RULE,'script_sha256':sha(__file__),'metrics_sha256':sha(ROOT/'hypernet/detection_metrics.py')}
    if args.split=='test': result['protocol_sha256']=acquisition['protocol_sha256']
    write_new(output/'result.json',result)
    print(json.dumps({k:{m:v for m,v in r.items() if m!='per_target'} for k,r in summary.items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True,choices=['owlvit_base_patch32','raw_template','v17','v17_no_geometry','random'])
    p.add_argument('--split',choices=['dev','test'],default='dev'); p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cuda'); p.add_argument('--max-side',type=int,default=384)
    run(p.parse_args())
