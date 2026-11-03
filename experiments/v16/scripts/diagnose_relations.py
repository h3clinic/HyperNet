"""Fixed heldout synthetic-type diagnostics; never selects training checkpoints."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from hypernet.variation_curriculum import catalog,render,hard_negative,EDGES
from hypernet.relational_encoder import RelationalEncoder,FORMAT,state_hash

PLAN={
    'scope':'Post-training synthetic diagnosis only. No real images, checkpoint or recipe selection.',
    'definition':'512 heldout axis-relative graph signatures; all are hash-partition disjoint from training.',
    'references_per_type':1,'query_instances_per_type':2,'image_side':96,
    'reference_seed_base':900000000,'query_seed_offsets':[1099511627776,2199023255552],
    'arms':['random','primary_final_4096','no_relation_aux_final_4096','raw_gray_cosine'],
    'main_metric':'Type top1 accuracy among all512 references; tie resolves lowest sorted reference index',
    'secondary_metrics':['mean reciprocal rank','positive-vs-nuisance-matched-negative pair accuracy','adjacency accuracy','node-degree accuracy'],
    'negative_scope':'hard_negative() preserves graph counts where declared; these negative signatures may have been seen during training. Positive heldout signatures have not.',
    'raw_baseline':'Native96x96 grayscale, subtract spatial mean, L2 normalize, cosine similarity.',
    'no_claim':'This controlled graph grammar is not semantic category recognition or a fresh real-data result.'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retrieval_metrics(scores,truth):
    scores=np.asarray(scores,dtype=np.float64);truth=np.asarray(truth,dtype=np.int64)
    prediction=scores.argmax(1)
    # Stable tie order is fixed reference index; it is identical for every arm.
    order=np.argsort(-scores,axis=1,kind='stable')
    ranks=np.argmax(order==truth[:,None],axis=1)+1
    return {'queries':len(truth),'correct':int((prediction==truth).sum()),
            'top1_accuracy':float(np.mean(prediction==truth)),
            'mean_reciprocal_rank':float(np.mean(1/ranks)),
            'chance_top1':1/scores.shape[1]},prediction,ranks


def load_model(path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['format']!=FORMAT or checkpoint['natural_training_images']!=0 or checkpoint['borrowed_pretrained_weights']:
        raise ValueError('Unsupported provenance')
    model=RelationalEncoder();model.load_state_dict(checkpoint['state_dict']);model.to(device).eval()
    if state_hash(model)!=checkpoint['state_sha256']:raise ValueError('State hash mismatch')
    return model,checkpoint


@torch.no_grad()
def representations(model,codes,seeds,device,batch_size=32):
    learned=[];raw=[];rgb_hashes=[];mask_hashes=[];degree_correct=0;edge_correct=0
    edges=torch.as_tensor(EDGES,device=device)
    for start in range(0,len(codes),batch_size):
        batch=render(codes[start:start+batch_size],seeds[start:start+batch_size],96,device)
        rgb=batch['rgb'].permute(0,3,1,2).float()/255
        gray=rgb.mean(1).flatten(1);raw.append(F.normalize(gray-gray.mean(1,keepdim=True),dim=1).cpu().numpy())
        out=model(rgb);learned.append(out['embedding'].cpu().numpy())
        degree_logits,edge_logits=model.relations(out,batch['node_xy'],edges)
        degree_correct+=int((degree_logits.argmax(-1)==batch['degrees']).sum())
        edge_correct+=int(((edge_logits>0)==batch['edges'].bool()).sum())
        rgb_hashes.extend(hashlib.sha256(image.tobytes()).hexdigest()for image in batch['rgb'].cpu().numpy())
        mask_hashes.extend(hashlib.sha256(mask.tobytes()).hexdigest()for mask in batch['mask'].cpu().numpy())
    return {'embedding':np.concatenate(learned),'raw':np.concatenate(raw),
            'rgb_sha256':rgb_hashes,'mask_sha256':mask_hashes,
            'node_degree_accuracy':degree_correct/(len(codes)*16),
            'edge_accuracy':edge_correct/(len(codes)*24)}


def run(args):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    outdir=ROOT/args.out
    if outdir.exists()and any(outdir.iterdir()):raise FileExistsError(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    codes=np.sort(catalog(True));n=len(codes)
    seeds=np.arange(n,dtype=np.uint64)+PLAN['reference_seed_base']
    query_codes=np.tile(codes,2);query_seeds=np.concatenate([seeds+offset for offset in PLAN['query_seed_offsets']])
    negative=[hard_negative(int(code))for code in query_codes]
    negative_codes=np.asarray([item[0]for item in negative],np.uint32)
    truth=np.tile(np.arange(n),2)
    paths={'random':ROOT/'training/primary_4096/random.pt',
           'primary_final_4096':ROOT/'training/primary_4096/final.pt',
           'no_relation_aux_final_4096':ROOT/'training/no_relation_aux_4096/final.pt'}
    summary={'plan':PLAN,'plan_sha256':sha(ROOT/'experiments/synthetic_diagnosis_plan.json'),
             'sources':{name:sha(ROOT/name)for name in ('hypernet/relational_encoder.py','hypernet/variation_curriculum.py','scripts/diagnose_relations.py')},'arms':{}}
    common=None
    for name,path in paths.items():
        model,checkpoint=load_model(path,args.device)
        if name!='random'and checkpoint['optimizer_steps']!=4096:raise ValueError('Only fixed final budget permitted')
        ref=representations(model,codes,seeds,args.device)
        query=representations(model,query_codes,query_seeds,args.device)
        neg=representations(model,negative_codes,query_seeds,args.device)
        hashes={'reference':ref['rgb_sha256'],'query':query['rgb_sha256'],'negative':neg['rgb_sha256']}
        if common is not None and common!=hashes:raise RuntimeError('Diagnostic inputs changed between arms')
        common=hashes
        scores=query['embedding']@ref['embedding'].T
        metrics,prediction,ranks=retrieval_metrics(scores,truth)
        positive_score=np.sum(query['embedding']*ref['embedding'][truth],axis=1)
        negative_score=np.sum(neg['embedding']*ref['embedding'][truth],axis=1)
        metrics.update(nuisance_matched_pair_accuracy=float(np.mean(positive_score>negative_score)),
                       nuisance_matched_ties=int(np.sum(positive_score==negative_score)),
                       node_degree_accuracy=query['node_degree_accuracy'],edge_accuracy=query['edge_accuracy'])
        np.savez_compressed(outdir/f'{name}_scores.npz',scores=scores,truth=truth,prediction=prediction,ranks=ranks,
                            positive_score=positive_score,negative_score=negative_score,reference_codes=codes,query_codes=query_codes,
                            negative_codes=negative_codes,reference_seeds=seeds,query_seeds=query_seeds)
        if state_hash(model)!=checkpoint['state_sha256']:raise RuntimeError('Diagnostic changed model state')
        summary['arms'][name]={'checkpoint_sha256':sha(path),'state_sha256':checkpoint['state_sha256'],
                               'optimizer_steps':checkpoint['optimizer_steps'],'metrics':metrics,
                               'scores_sha256':sha(outdir/f'{name}_scores.npz')}
        if name=='random':
            raw_scores=query['raw']@ref['raw'].T
            raw_metrics,raw_prediction,raw_ranks=retrieval_metrics(raw_scores,truth)
            raw_positive=np.sum(query['raw']*ref['raw'][truth],axis=1)
            raw_negative=np.sum(neg['raw']*ref['raw'][truth],axis=1)
            raw_metrics.update(nuisance_matched_pair_accuracy=float(np.mean(raw_positive>raw_negative)),
                               nuisance_matched_ties=int(np.sum(raw_positive==raw_negative)))
            np.savez_compressed(outdir/'raw_gray_cosine_scores.npz',scores=raw_scores,truth=truth,prediction=raw_prediction,
                                ranks=raw_ranks,positive_score=raw_positive,negative_score=raw_negative,
                                reference_codes=codes,query_codes=query_codes,negative_codes=negative_codes,
                                reference_seeds=seeds,query_seeds=query_seeds)
            summary['arms']['raw_gray_cosine']={'metrics':raw_metrics,'scores_sha256':sha(outdir/'raw_gray_cosine_scores.npz')}
        print(json.dumps({'arm':name,**metrics}),flush=True)
        del model
    summary['input_rgb_hashes']=common
    summary['negative_kind_counts']={kind:sum(item[1]==kind for item in negative)for kind in sorted(set(item[1]for item in negative))}
    summary['reference_mask_sha256']=ref['mask_sha256'];summary['query_mask_sha256']=query['mask_sha256']
    (outdir/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--out',default='experiments/synthetic_final_diagnosis');p.add_argument('--write-plan',action='store_true')
    a=p.parse_args()
    if a.write_plan:
        path=ROOT/'experiments/synthetic_diagnosis_plan.json';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(PLAN,indent=2),encoding='utf-8');print(sha(path))
    else:run(a)
