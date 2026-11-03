"""Predeclared synthetic proof: pair discrimination AND full-scene reranking.

These are separate tasks. Cropped-pair AUROC must not be compared to scene
AUROC. No natural images or v17's sealed fresh real TEST records are opened.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,argparse,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from hypernet_v18.pair_ranker import INPUT_KEYS,file_sha
from hypernet_v18.pair_curriculum import batch
from hypernet_v18.reranker import FrozenPatchReranker
from hypernet_v18.vendor_v17 import scene_curriculum as sc
from hypernet_v18.vendor_v17.local_matcher import box_iou_one
from train_pair_ranker import atomic_json

PLAN={'format':'hypernet.v18.synthetic_proof_eval.v1','splits':{
    'development':{'seed':1877,'count':256},'test':{'seed':1888,'count':256},'compositional':{'seed':1888,'count':128}},
    'nominal_development_fpr':.05,'scene_iou_threshold':.5,'proposal_limit':100,'patch_batch':32,
    'scene_query_max_side':192,'checkpoint_rule':'Final512updates only; no checkpoint selection',
    'natural_images':0,'v17_fresh_real_test_access':False,
    'interpretation':'Pair discrimination and full-scene detection are separate metrics; only same-scene v17/v18 results are directly comparable'}


def auc(positive,negative):
    p=np.asarray(positive);n=np.asarray(negative)
    return float(((p[:,None]>n[None]).sum()+.5*(p[:,None]==n[None]).sum())/(len(p)*len(n)))if len(p)*len(n)else None


def cutoff(negative):
    scores=np.sort(np.asarray(negative,np.float64))[::-1];allowed=int(.05*len(scores))
    return float(np.nextafter(scores[allowed],np.inf))


def summarize_pairs(rows,threshold):
    positive=[r['positive']for r in rows];near=[r['near_negative']for r in rows];wrong=[r['wrong_family']for r in rows]
    return {'quartets':len(rows),'near_negative_auroc':auc(positive,near),'wrong_family_auroc':auc(positive,wrong),
        'all_nonmatch_auroc':auc(positive,near+wrong),
        'paired_near_ranking_accuracy':float(np.mean(np.asarray(positive)>np.asarray(near))),
        'paired_wrong_ranking_accuracy':float(np.mean(np.asarray(positive)>np.asarray(wrong))),
        'positive_recall':float(np.mean(np.asarray(positive)>=threshold)),
        'near_negative_false_positive_rate':float(np.mean(np.asarray(near)>=threshold)),
        'wrong_reference_false_positive_rate':float(np.mean(np.asarray(wrong)>=threshold)),
        'mean_near_logit_gap':float(np.mean(np.asarray(positive)-near)),
        'mean_wrong_logit_gap':float(np.mean(np.asarray(positive)-wrong))}


def summarize_scenes(rows,method,threshold):
    p=[r for r in rows if r['present']];n=[r for r in rows if not r['present']]
    hits=sum(r[method]['score']>=threshold and r[method]['iou']>=.5 for r in p)
    false=sum(r[method]['score']>=threshold for r in n)
    return {'episodes':len(rows),'positives':len(p),'negatives':len(n),'localized_true_positives':hits,'false_alarms':false,
        'localized_recall':hits/len(p)if p else None,'false_positive_rate':false/len(n)if n else None,
        'false_alarms_per10000':10000*false/len(n)if n else None,
        'presence_auroc':auc([r[method]['score']for r in p],[r[method]['score']for r in n]),
        'unthresholded_top1_localized_recall':float(np.mean([r[method]['iou']>=.5 for r in p]))if p else None,
        'proposal_recall_at100':float(np.mean([r['proposal_max_iou']>=.5 for r in p]))if p else None}


@torch.inference_mode()
def evaluate_pairs(matcher,output):
    result={'plan':PLAN,'task':'Synthetic cropped conditional-match pairs, NOT scene detection','splits':{},'rows':{}}
    threshold=None;start=time.perf_counter()
    for split,setting in PLAN['splits'].items():
        rows=[];digest=hashlib.sha256()
        for start_index in range(0,setting['count'],8):
            b=batch(range(start_index,min(start_index+8,setting['count'])),split=split,seed=setting['seed'],device=matcher.device)
            scores=matcher.model(*(b[key]for key in INPUT_KEYS))
            for j,meta in enumerate(b['metadata']):
                for key in INPUT_KEYS:digest.update(key.encode());digest.update(bytes.fromhex(meta['image_sha256'][key]))
                rows.append({'index':meta['index'],'family':meta['family'],'program_id':meta['program_id'],
                    **{key:float(value[j])for key,value in scores.items()}})
        if split=='development':threshold=cutoff([r[k]for r in rows for k in ('near_negative','wrong_family')])
        metrics=summarize_pairs(rows,threshold)
        metrics['families']={family:summarize_pairs([r for r in rows if r['family']==family],threshold)for family in sorted({r['family']for r in rows})}
        result['splits'][split]={'threshold':threshold,'input_stream_sha256':digest.hexdigest(),'metrics':metrics};result['rows'][split]=rows
        print(json.dumps({'task':'pairs','split':split,**{k:metrics[k]for k in ('near_negative_auroc','wrong_family_auroc','paired_near_ranking_accuracy')}}),flush=True)
    matcher.assert_unchanged();result['seconds']=time.perf_counter()-start
    result['provenance']=matcher.provenance();atomic_json(output,result)


@torch.inference_mode()
def evaluate_scenes(matcher,output):
    result={'plan':PLAN,'task':'Same full synthetic scenes, frozen100v17proposals, one reference; no oracle crop inputs',
            'splits':{},'rows':{},'provenance':matcher.provenance()};thresholds={};start=time.perf_counter()
    for split,setting in PLAN['splits'].items():
        rows=[];digest=hashlib.sha256()
        for index in range(setting['count']):
            row=sc.episode(index,split=split,seed=setting['seed'])
            for key in ('support','query'):digest.update(key.encode());digest.update(hashlib.sha256(row[key].tobytes()).digest())
            bank=matcher.enroll(row['support']);query=matcher.encode_query(row['query'])
            original=matcher.stage.predict(bank['proposal_bank'],query['proposal_encoded'])
            ranked=matcher.predict(bank,query)
            # Exact same boxes survive both methods; only ordering/scores differ.
            assert sorted(tuple(p['box'])for p in original)==sorted(tuple(p['box'])for p in ranked)
            candidate_boxes=np.asarray([p['box']for p in original]);ious=box_iou_one(row['bbox'],candidate_boxes)if row['present']else np.zeros(len(original))
            record={'index':index,'family':row['family'],'present':bool(row['present']),'program_id':row['program_id'],
                    'bbox':row['bbox'].tolist(),'proposal_max_iou':float(ious.max()),
                    'reference_rgb_sha256':hashlib.sha256(row['support'].tobytes()).hexdigest(),
                    'query_rgb_sha256':hashlib.sha256(row['query'].tobytes()).hexdigest()}
            for name,predictions in (('v17',original),('v18',ranked)):
                top=predictions[0];iou=float(box_iou_one(row['bbox'],np.asarray([top['box']]))[0])if row['present']else 0.
                record[name]={'score':top['score'],'box':top['box'],'iou':iou}
            rows.append(record)
            if (index+1)%64==0:print(json.dumps({'task':'scenes','split':split,'episodes':index+1,'planned':setting['count']}),flush=True)
        if split=='development':thresholds={name:cutoff([r[name]['score']for r in rows if not r['present']])for name in ('v17','v18')}
        metrics={name:summarize_scenes(rows,name,thresholds[name])for name in ('v17','v18')}
        for name in metrics:metrics[name]['families']={family:summarize_scenes([r for r in rows if r['family']==family],name,thresholds[name])for family in sorted({r['family']for r in rows})}
        result['splits'][split]={'thresholds':thresholds,'input_stream_sha256':digest.hexdigest(),'metrics':metrics};result['rows'][split]=rows
        print(json.dumps({'task':'scenes','split':split,'metrics':{m:{k:v[k]for k in ('localized_recall','false_positive_rate','presence_auroc','unthresholded_top1_localized_recall')}for m,v in metrics.items()}}),flush=True)
        atomic_json(output.with_name('scene_progress.json'),{'completed_split':split,'seconds':time.perf_counter()-start})
    matcher.assert_unchanged();result['seconds']=time.perf_counter()-start;atomic_json(output,result)


def main(args):
    torch.set_num_threads(2);torch.manual_seed(1801);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    expected=[output/'pair_results.json',output/'scene_results.json']
    if any(p.exists()for p in expected):raise FileExistsError('Preserve earlier evaluation')
    atomic_json(output/'plan.json',{'plan':PLAN,'evaluation_source_sha256':file_sha(__file__)})
    matcher=FrozenPatchReranker(args.checkpoint,args.own_checkpoint,device=args.device,max_side=192,proposal_limit=100,patch_batch=32)
    evaluate_pairs(matcher,expected[0]);evaluate_scenes(matcher,expected[1])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'runs/proof512/final.pt'))
    p.add_argument('--own-checkpoint',default=str(ROOT.parent/'v17/runs/primary_2048/final.pt'))
    p.add_argument('--output',default='experiments/proof512_eval');p.add_argument('--device',default='cuda')
    main(p.parse_args())
