from pathlib import Path
import sys
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet_v18.pair_ranker import PairRanker,objective,file_sha,parameter_summary
from hypernet_v18.vendor_v17.local_relation_model import LocalRelationNet,FORMAT,state_hash
torch.set_num_threads(2)


def checkpoint(path,borrowed=False):
    torch.manual_seed(1717);model=LocalRelationNet()
    torch.save({'format':FORMAT,'state_dict':model.state_dict(),'state_sha256':state_hash(model),
        'natural_training_images':0,'borrowed_pretrained_weights':borrowed,'optimizer_steps':0,
        'training_episodes':0,'training_views':0,
        'config':{'geometry':True,'source_sha256':{'hypernet/local_relation_model.py':file_sha(ROOT/'hypernet_v18/vendor_v17/local_relation_model.py')}}},path)


def test_own_source_and_initialization_are_checked(tmp_path):
    path=tmp_path/'own.pt';checkpoint(path);model=PairRanker(path)
    assert model.initialization['checkpoint_sha256']==file_sha(path)
    assert model.initialization['state_sha256']==state_hash(model.encoder)
    checkpoint(path,True)
    with pytest.raises(ValueError,match='Only own'):PairRanker(path)


def test_all_ordered_cell_evidence_is_retained():
    torch.manual_seed(2);model=PairRanker()
    s=torch.nn.functional.normalize(torch.randn(2,32,16),dim=1)
    q=torch.nn.functional.normalize(torch.randn(2,32,16),dim=1)
    evidence=model.ordered_evidence(s,q)
    assert evidence.shape==(2,16,16)
    assert torch.allclose(evidence,torch.bmm(s.transpose(1,2),q))
    changed=model.ordered_evidence(s,q.flip(-1))
    assert torch.equal(changed,evidence.flip(-1))
    assert not torch.allclose(evidence,changed)


def test_absolute_logistic_loss_is_not_offset_invariant():
    scores={k:torch.tensor([v],requires_grad=True)for k,v in [('positive',4.),('near_negative',-4.),('wrong_family',-4.)]}
    loss,_=objective(scores);loss.backward()
    assert scores['positive'].grad.item()<0
    assert scores['near_negative'].grad.item()>0 and scores['wrong_family'].grad.item()>0
    shifted={k:v.detach()+10 for k,v in scores.items()}
    assert objective(shifted)[0]>loss*100


def test_full_encoder_and_score_head_receive_finite_gradients():
    torch.manual_seed(9);model=PairRanker()
    images=[torch.rand(1,3,96,96,requires_grad=True)for _ in range(4)]
    result=model(*images);loss,_=objective(result);loss.backward()
    assert all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None)
    assert model.encoder.stem[0].weight.grad.abs().sum()>0
    assert model.encoder.descriptor.weight.grad.abs().sum()>0
    assert model.scorer[0].weight.grad.abs().sum()>0
    assert all(x.grad.abs().sum()>0 for x in images)
    for name in ('foreground','detail','match_head','seg_head'):
        assert all(not p.requires_grad and p.grad is None for p in getattr(model.encoder,name).parameters())


def test_reference_substitution_changes_fixed_region_score():
    torch.manual_seed(19);model=PairRanker().eval()
    images=torch.rand(3,3,96,96)
    with torch.no_grad():tokens=model.encode_patches(images)
    a=model.score_tokens(tokens[:1],tokens[2:]);b=model.score_tokens(tokens[1:2],tokens[2:])
    assert (a-b).abs().item()>1e-7


def test_batch_members_do_not_change_patch_descriptor():
    torch.manual_seed(7);model=PairRanker().eval();images=torch.rand(2,3,96,96)
    with torch.no_grad():
        a=model.encode_patches(images[:1]);b=model.encode_patches(images)[:1]
    assert torch.allclose(a,b,atol=2e-5,rtol=1e-5)
    assert parameter_summary(model)['ordered_evidence_entries']==256


def test_frozen_reranker_batches_and_caches_unchanged_regions(tmp_path):
    import numpy as np
    from hypernet_v18.pair_ranker import FORMAT as RANK_FORMAT
    from hypernet_v18.reranker import FrozenPatchReranker
    own=tmp_path/'own.pt';checkpoint(own);model=PairRanker(own)
    path=tmp_path/'rank.pt'
    torch.save({'format':RANK_FORMAT,'state_dict':model.state_dict(),'state_sha256':state_hash(model),
                'own_initialization':model.initialization,'training':{'optimizer_steps':0,'quartets':0,'rgb_views':0},
                'natural_training_images':0,'borrowed_pretrained_weights':False},path)
    frozen=FrozenPatchReranker(path,own,max_side=64,proposal_limit=3,patch_batch=2)
    rng=np.random.default_rng(3);ref=rng.integers(0,256,(35,45,3),dtype=np.uint8)
    query=rng.integers(0,256,(51,83,3),dtype=np.uint8)
    bank=frozen.enroll(ref);encoded=frozen.encode_query(query)
    results=frozen.predict_many([bank,bank],encoded)
    assert results[0]==results[1] and len(results[0])==3
    count=len(encoded['patch_tokens']);assert 1<=count<=3
    assert frozen.predict(bank,encoded)==results[0]and len(encoded['patch_tokens'])==count
    for row in results[0]:
        x1,y1,x2,y2=row['box'];assert 0<=x1<x2<=83 and 0<=y1<y2<=51
        assert np.isfinite(row['score'])
    frozen.assert_unchanged()
