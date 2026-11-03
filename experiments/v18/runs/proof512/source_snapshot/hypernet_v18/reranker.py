"""Frozen own-v17 proposals followed by batched one-reference patch ranking."""
from pathlib import Path
import numpy as np
import torch
from .pair_ranker import PairRanker,FORMAT,file_sha,parameter_summary
from .vendor_v17.local_relation_model import state_hash
from .vendor_v17.local_matcher import FrozenLocalMatcher,checked_rgb,prepare_image


def integer_crop(box,height,width):
    x1,y1,x2,y2=box
    left=max(0,min(width-1,int(np.floor(x1))));top=max(0,min(height-1,int(np.floor(y1))))
    right=max(left+1,min(width,int(np.ceil(x2))));bottom=max(top+1,min(height,int(np.ceil(y2))))
    return left,top,right,bottom


class FrozenPatchReranker:
    def __init__(self,checkpoint,proposal_checkpoint,device='cpu',max_side=384,proposal_limit=100,patch_batch=32):
        self.path=Path(checkpoint).resolve();self.file_hash=file_sha(self.path);self.device=str(device)
        saved=torch.load(self.path,map_location='cpu',weights_only=True)
        if saved.get('format')!=FORMAT or saved.get('natural_training_images')!=0 or saved.get('borrowed_pretrained_weights')is not False:
            raise ValueError('Expected own synthetic-only v18 checkpoint')
        self.saved=saved;self.model=PairRanker().to(device);self.model.load_state_dict(saved['state_dict'],strict=True)
        self.model.eval().requires_grad_(False);self.initial_state=state_hash(self.model)
        if self.initial_state!=saved['state_sha256']:raise ValueError('Ranker tensor hash mismatch')
        self.stage=FrozenLocalMatcher(proposal_checkpoint,device=device,max_side=max_side,proposal_limit=proposal_limit)
        if self.stage.checkpoint_sha256!=saved['own_initialization']['checkpoint_sha256']:
            raise ValueError('Proposal source must be the same own v17 checkpoint declared before training')
        self.proposal_limit=int(proposal_limit);self.patch_batch=int(patch_batch)
        if self.patch_batch<1:raise ValueError('Positive patch batch required')

    def _tensor(self,arrays):
        return torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).to(self.device,dtype=torch.float32)/255

    @torch.inference_mode()
    def enroll(self,reference):
        rgb=checked_rgb(reference);patch,_=prepare_image(rgb,support=True)
        return {'state_sha256':self.initial_state,'proposal_bank':self.stage.enroll(rgb),
                'tokens':self.model.encode_patches(self._tensor([patch])),
                'reference_images':1,'real_weight_updates':0}

    @torch.inference_mode()
    def encode_query(self,query):
        rgb=checked_rgb(query).copy()
        return {'state_sha256':self.initial_state,'rgb':rgb,'proposal_encoded':self.stage.encode_query(rgb),'patch_tokens':{}}

    @torch.inference_mode()
    def predict_many(self,banks,query):
        encoded=query if isinstance(query,dict)else self.encode_query(query)
        if encoded.get('state_sha256')!=self.initial_state:raise ValueError('Query belongs to a different ranker')
        proposals=[];keys=[];rgb=encoded['rgb'];h,w=rgb.shape[:2];cache=encoded['patch_tokens'];missing={}
        for bank in banks:
            if bank.get('state_sha256')!=self.initial_state:raise ValueError('Support belongs to a different ranker')
            p=self.stage.predict(bank['proposal_bank'],encoded['proposal_encoded'])
            proposals.append(p);rowkeys=[integer_crop(x['box'],h,w)for x in p];keys.append(rowkeys)
            for key in rowkeys:
                if key not in cache and key not in missing:
                    left,top,right,bottom=key;missing[key]=prepare_image(rgb[top:bottom,left:right],support=True)[0]
        pending=list(missing)
        for start in range(0,len(pending),self.patch_batch):
            group=pending[start:start+self.patch_batch]
            tokens=self.model.encode_patches(self._tensor([missing[key]for key in group]))
            for j,key in enumerate(group):cache[key]=tokens[j:j+1]
        results=[]
        for bank,p,rowkeys in zip(banks,proposals,keys):
            tokens=torch.cat([cache[key]for key in rowkeys],0)
            scores=self.model.score_tokens(bank['tokens'],tokens).float().cpu().numpy()
            if not np.isfinite(scores).all():raise RuntimeError('Nonfinite conditional match score')
            order=np.argsort(-scores,kind='stable')
            results.append([{'box':p[int(j)]['box'],'score':float(scores[j]),'proposal_score':p[int(j)]['score']}for j in order])
        return results

    @torch.inference_mode()
    def predict(self,bank,query):return self.predict_many([bank],query)[0]

    def provenance(self):
        return {'format':FORMAT,'checkpoint_sha256':self.file_hash,'state_sha256':self.initial_state,
                'own_initialization':self.saved['own_initialization'],'rank_training':self.saved['training'],
                'proposal_stage':self.stage.provenance(),'ranker':self.saved.get('parameter_summary',parameter_summary(self.model)),
                'ranking_score':'Class-balanced conditional-match logit; no score fusion or confidence prefilter',
                'region_input':'Frozen v17 proposed box, floor/ceil original pixels, aspect-preserving96px median-border letterbox',
                'proposal_limit':self.proposal_limit,'patch_batch':self.patch_batch,
                'cache':'Original-image integer crop key; query patch descriptors reused across references',
                'natural_training_images':0,'borrowed_pretrained_weights':False,'real_weight_updates':0}

    def assert_unchanged(self):
        self.stage.assert_unchanged()
        if state_hash(self.model)!=self.initial_state or file_sha(self.path)!=self.file_hash:
            raise RuntimeError('Frozen ranker changed')
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):raise RuntimeError('Ranker became trainable')
