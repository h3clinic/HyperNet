"""Ordered cell-pair evidence and absolutely anchored conditional-match loss.

This is a controlled patch-ranking prototype, not an established novelty claim.
Its shared encoder starts from OUR synthetic-only v17 weights. All16x16 ordered
cell similarities reach the score head; no top-half selection or global moments.
"""
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .vendor_v17.local_relation_model import LocalRelationNet,part_pool,state_hash,FORMAT as V17_FORMAT

FORMAT='hypernet.pair_ranker.v18.1'
INPUT_KEYS=('support','positive','near_negative','wrong_support')


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PairRanker(nn.Module):
    def __init__(self,own_v17_checkpoint=None):
        super().__init__();self.encoder=LocalRelationNet(geometry=True)
        self.initialization=None
        if own_v17_checkpoint is not None:
            path=Path(own_v17_checkpoint);saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved.get('format')!=V17_FORMAT or saved.get('natural_training_images')!=0 or saved.get('borrowed_pretrained_weights')is not False:
                raise ValueError('Only own synthetic-only v17 initialization is supported')
            self.encoder.load_state_dict(saved['state_dict'],strict=True)
            if state_hash(self.encoder)!=saved['state_sha256']:raise ValueError('Own v17 state hash mismatch')
            vendor=Path(__file__).parent/'vendor_v17/local_relation_model.py'
            if saved['config']['source_sha256']['hypernet/local_relation_model.py']!=file_sha(vendor):
                raise ValueError('Own v17 encoder source snapshot differs')
            self.initialization={'kind':'Own synthetic-only v17 initialization, not third-party pretrained weights',
                'checkpoint_sha256':file_sha(path),'state_sha256':saved['state_sha256'],
                'prior_optimizer_steps':saved['optimizer_steps'],'prior_training_episodes':saved['training_episodes'],
                'prior_training_views':saved['training_views'],'prior_source_sha256':saved['config']['source_sha256']}
        # The ranker uses only encode()->raw. These inherited branches are kept
        # for checkpoint transparency, but cannot receive optimization updates.
        for name in ('foreground','detail','match_head','seg_head'):
            getattr(self.encoder,name).requires_grad_(False)
        self.scorer=nn.Sequential(nn.Linear(16*16,128),nn.SiLU(),nn.Linear(128,1))

    def encode_patches(self,rgb):
        if rgb.shape[-2:]!=(96,96):raise ValueError('Ranker expects96px aspect-preserving patches')
        raw=self.encoder.encode(rgb)['raw']
        # Full ordered spatial grid, including background/letterbox. No oracle
        # masks, detached foreground extent, selected parts, or GT enter here.
        return F.normalize(part_pool(raw).flatten(2),dim=1,eps=1e-6)

    def ordered_evidence(self,support_tokens,region_tokens):
        if support_tokens.shape[0]==1 and region_tokens.shape[0]!=1:
            support_tokens=support_tokens.expand(region_tokens.shape[0],-1,-1)
        if support_tokens.shape!=region_tokens.shape or support_tokens.shape[1:]!=(32,16):
            raise ValueError('Expected aligned Bx32x16 descriptor cells')
        return torch.einsum('bcp,bcq->bpq',support_tokens,region_tokens)

    def score_tokens(self,support_tokens,region_tokens):
        evidence=self.ordered_evidence(support_tokens,region_tokens)
        return self.scorer(evidence.flatten(1)).squeeze(1)

    def forward(self,support,positive,near_negative,wrong_support):
        b=len(support)
        tokens=self.encode_patches(torch.cat([support,positive,near_negative,wrong_support],0))
        s,p,n,w=tokens.split(b,0)
        return {'positive':self.score_tokens(s,p),'near_negative':self.score_tokens(s,n),
                'wrong_family':self.score_tokens(w,p)}


def objective(scores):
    """One class-balanced logistic loss anchors absolute match/nonmatch scores.

    Positive prior has weight1/2; two distinct nonmatch controls share the other
    half. Pairwise margin diagnostics are reported but add no auxiliary losses.
    """
    positive=F.softplus(-scores['positive']).mean()
    near=F.softplus(scores['near_negative']).mean()
    wrong=F.softplus(scores['wrong_family']).mean()
    loss=.5*positive+.25*near+.25*wrong
    stats={'loss':loss,'positive_logistic':positive,'near_negative_logistic':near,'wrong_family_logistic':wrong,
           'positive_mean_logit':scores['positive'].mean(),'near_negative_mean_logit':scores['near_negative'].mean(),
           'wrong_family_mean_logit':scores['wrong_family'].mean(),
           'near_pair_accuracy':(scores['positive']>scores['near_negative']).float().mean(),
           'wrong_family_pair_accuracy':(scores['positive']>scores['wrong_family']).float().mean(),
           'near_mean_margin':(scores['positive']-scores['near_negative']).mean(),
           'wrong_family_mean_margin':(scores['positive']-scores['wrong_family']).mean()}
    return loss,{k:float(v.detach())for k,v in stats.items()}


def parameter_summary(model):
    return {'total_parameters':sum(p.numel()for p in model.parameters()),
            'trainable_parameters':sum(p.numel()for p in model.parameters()if p.requires_grad),
            'ordered_evidence_entries':256,'source_cells':16,'region_cells':16,
            'loss':'0.5 positive logistic +0.25 near-negative logistic +0.25 wrong-support logistic'}
