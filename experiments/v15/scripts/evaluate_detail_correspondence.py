"""Held-out mathematical dense retrieval, separately from real-image tasks.

Each source descriptor competes against every target pixel by default. An
explicit foreground-oracle mode is a separate diagnostic, not detection.
Ground-truth correspondence labels are used only to score the retrieved point.
Repeated motifs can remain ambiguous; PCK is not semantic recognition accuracy.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.detail_encoder import DetailEncoder,FORMAT,state_hash,anchor_coordinates,sample_at_grid,conservative_validity
from hypernet.math_curriculum_torch import sample_batch
from train_details import tensor_batch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--families',type=int,default=256)
    parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--search',choices=['full','foreground_oracle'],default='full')
    parser.add_argument('--threads',type=int,default=4);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier diagnostics')
    args.output.mkdir(parents=True);torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Early, own smoke checkpoints accidentally stored torch.__version__ as its
    # TorchVersion subclass. Main-run metadata is plain str. This narrow trusted
    # compatibility allowlist does not permit arbitrary checkpoint objects.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    if saved.get('format')!=FORMAT or saved.get('natural_training_images')!=0 or saved.get('borrowed_pretrained_weights') is not False:
        raise ValueError('Expected original mathematical-only checkpoint')
    model=DetailEncoder().to(args.device);model.load_state_dict(saved['state_dict'],strict=True);model.eval().requires_grad_(False)
    before=state_hash(model)
    if before!=saved['state_sha256']:raise ValueError('Checkpoint tensor digest differs')
    source_names=['hypernet/detail_encoder.py','hypernet/math_curriculum.py','hypernet/math_curriculum_torch.py',
        'scripts/train_details.py','scripts/evaluate_detail_correspondence.py']
    sources={name:sha(ROOT/name) for name in source_names};checkpoint_sha=sha(args.checkpoint)
    side=saved['admission']['side'];renderer_seed=saved['admission']['renderer_seed']+1
    generator=torch.Generator(device=args.device).manual_seed(151599)
    rows=[];started=time.perf_counter();input_hash=hashlib.sha256()
    with torch.inference_mode():
        for offset in range(1048576,1048576+args.families,args.batch_size):
            ids=np.arange(offset,min(offset+args.batch_size,1048576+args.families),dtype=np.int64)
            raw=sample_batch(ids,renderer_seed,size=side,device=args.device)
            for key in ('view_a','view_b'):input_hash.update(raw[key].cpu().numpy().tobytes())
            batch=tensor_batch(raw,args.device);a=model(batch['rgb_a']);b=model(batch['rgb_b'])
            valid,_,coverage=conservative_validity(batch)
            for index,family in enumerate(ids):
                base={'family_id':int(family),'target_foreground_pixels':int(batch['mask_b'][index].sum()),
                    'usable_source_pixels':int(valid[index].sum())}
                if not valid[index].any():rows.append(dict(base,anchors=0,reason='no conservative bilinear correspondence support'));continue
                grid,xy=anchor_coordinates(valid[index:index+1],batch['detail_a'][index:index+1],32,generator)
                reference=sample_at_grid(a['dense'][index:index+1],grid)
                reference=torch.nn.functional.normalize(reference,dim=-1)
                candidates=b['dense'][index].flatten(1)
                scores=reference.squeeze(0)@candidates
                if args.search=='foreground_oracle':scores[:,~batch['mask_b'][index].flatten()]=-float('inf')
                winner=scores.argmax(1);predicted=torch.stack([winner%side,winner//side],-1).float()
                target=sample_at_grid(batch['grid_a_to_b'][index:index+1].permute(0,3,1,2),grid).squeeze(0)
                expected=(target+1)*side/2-.5
                distances=(predicted-expected).norm(dim=-1)
                rows.append(dict(base,anchors=len(distances),pck2_count=int((distances<=2).sum()),pck4_count=int((distances<=4).sum()),
                    endpoint_error_pixels=distances.cpu().tolist(),source_xy=xy.squeeze(0).cpu().tolist(),
                    expected_target_xy=expected.cpu().tolist(),predicted_target_xy=predicted.cpu().tolist(),
                    cosine_scores=scores.gather(1,winner[:,None]).squeeze(1).cpu().tolist()))
    if state_hash(model)!=before or sha(args.checkpoint)!=checkpoint_sha:raise RuntimeError('Diagnostic changed weights')
    if any(sha(ROOT/name)!=digest for name,digest in sources.items()):raise RuntimeError('Diagnostic source changed')
    errors=np.asarray([x for row in rows for x in row.get('endpoint_error_pixels',[])])
    summary={'completed':True,'scope':'Synthetic held-out-family dense retrieval, not real-image or category accuracy','search_mode':args.search,
        'checkpoint_name':args.checkpoint.name,'checkpoint_sha256':checkpoint_sha,'state_sha256':before,'state_unchanged':True,
        'source_sha256':sources,'device':args.device,'side':side,'families':len(rows),'families_with_anchors':sum(r['anchors']>0 for r in rows),
        'anchors':len(errors),'pck_at_2px':float(np.mean(errors<=2)),'pck_at_4px':float(np.mean(errors<=4)),
        'mean_endpoint_error_px':float(errors.mean()),'median_endpoint_error_px':float(np.median(errors)),
        'target_search':'Every image pixel, including independent background/clutter; no target annotations used for search' if args.search=='full' else 'Every ground-truth foreground pixel; generous oracle restriction, no spatial window around true answer',
        'input_stream_sha256':input_hash.hexdigest(),'seconds':time.perf_counter()-started,
        'natural_images':0,'optimizer_steps':0,'borrowed_weights':False,'rows':rows}
    (args.output/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','source_sha256')}),flush=True)


if __name__=='__main__':main()
