"""Fixed counterfactual tiny-set optimization, not held-out model performance."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from hypernet.scene_curriculum import episode
from hypernet.local_relation_model import LocalRelationNet,objective,decoded_boxes
from train_local import setup,collate


@torch.no_grad()
def evaluate(model,b):
    model.eval();out=model(b['support'],b['query']);_,loss=objective(out,b)
    score,index=out['center_logits'].flatten(1).max(1)
    boxes=decoded_boxes(out)[torch.arange(len(index)),index].clamp(0,192)
    gt=b['bbox'];inter=(torch.minimum(boxes[:,2:],gt[:,2:])-torch.maximum(boxes[:,:2],gt[:,:2])).clamp_min(0).prod(-1)
    union=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(-1)+(gt[:,2:]-gt[:,:2]).prod(-1)-inter
    iou=inter/union.clamp_min(1e-8);pos=b['present']>.5
    result={**loss,'top1_boxes':boxes.cpu().tolist(),'top1_iou':iou.cpu().tolist(),'top1_scores':score.cpu().tolist(),
            'present':pos.cpu().tolist(),'positive_localized_recall':float((iou[pos]>=.5).float().mean()),
            'min_positive_minus_max_negative_score':float(score[pos].min()-score[~pos].max())}
    # Same queries against a deliberately unrelated support bank, diagnostic only.
    wrong=model(b['support'].flip(0),b['query'])['center_logits']
    result['reference_changed_mean_absolute_logit_change']=float((wrong-out['center_logits']).abs().mean())
    model.train();return result


def main(output):
    path=ROOT/output;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in
        (ROOT/'hypernet/local_relation_model.py',ROOT/'hypernet/scene_curriculum.py',ROOT/'scripts/train_local.py',Path(__file__))}
    setup(1717,4);model=LocalRelationNet(True).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    rows=[episode(i,positive_probability=p)for i in (0,1)for p in (1.,0.)]
    cpu=collate(rows);b={k:v.cuda()if isinstance(v,torch.Tensor)else v for k,v in cpu.items()}
    result={'scope':'Optimization/plumbing diagnosis on FOUR repeatedly trained synthetic episodes, not held-out accuracy',
            'natural_images':0,'weights_discarded':True,'steps':384,'episode_indices':[0,0,1,1],
            'positive_probability':[1,0,1,0],'initial':evaluate(model,b),'checkpoints':[]};start=time.perf_counter()
    for step in range(1,385):
        opt.zero_grad(set_to_none=True);out=model(b['support'],b['query']);loss,stats=objective(out,b)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        if step%64==0:
            row={'step':step,'seconds':time.perf_counter()-start,**evaluate(model,b)};result['checkpoints'].append(row)
            print(json.dumps({k:row[k]for k in ('step','seconds','loss','center_loss','positive_localized_recall','min_positive_minus_max_negative_score')}),flush=True)
    result['final']=result['checkpoints'][-1]
    result['sources']=sources
    result['sources_unchanged']=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in sources.items())
    path.write_text(json.dumps(result,indent=2)+'\n');print(str(path),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='experiments/local_counterfactual_sanity_framed.json')
    main(parser.parse_args().output)
