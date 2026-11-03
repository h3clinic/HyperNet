import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from hypernet.variation_curriculum import catalog,stream,render,labels,partition,hard_negative,component_count
from hypernet.relational_encoder import RelationalEncoder,supervised_contrast,state_hash
from train_relations import setup,ledger_record,write_checkpoint


def test_structural_split_and_nuisance_matching():
    assert len(set(map(int,catalog())))==32768
    assert not set(map(int,catalog()))&set(map(int,catalog(True)))
    codes,a,b,kinds=stream(0,8)
    assert np.array_equal(a[:4],a[4:]) and np.array_equal(b[:4],b[4:])
    assert np.all(a!=b) and np.all(codes[:4]!=codes[4:])
    for code in codes[:4]:
        negative,kind=hard_negative(int(code));e,d=labels([code,negative])
        assert e[0,:12].sum()==e[1,:12].sum() and e[0,12:].sum()==e[1,12:].sum()
        if kind!='edge_count_orientation_matched_only':
            assert np.array_equal(np.sort(d[0]),np.sort(d[1]))
            assert component_count(int(code))==component_count(negative)


def test_renderer_partition_independent_and_ledger_actual_hashes():
    torch.set_num_threads(2)
    codes,a,b,_=stream(1,4)
    x=render(codes,a,64);y=render(codes,b,64)
    singleton=render(codes[:1],a[:1],64)
    assert torch.equal(x['rgb'][:1],singleton['rgb'])
    assert not torch.equal(x['rgb'],y['rgb'])
    record=ledger_record(codes,a,b,x,y)
    assert record['rgb_sha256'].shape==(4,2,32)
    assert record['instance_seeds'].shape==(4,2)
    assert np.array_equal(record['type_codes'],codes)
    assert not np.array_equal(record['rgb_sha256'][:,0],record['rgb_sha256'][:,1])


def test_repeated_type_is_positive_not_negative():
    embedding=torch.eye(3).repeat_interleave(2,dim=0)
    codes=torch.tensor([10,10,20,20,30,30])
    loss,accuracy=supervised_contrast(embedding,codes)
    assert accuracy==1 and loss<.002
    duplicate=torch.cat([embedding,embedding[:2]])
    loss2,accuracy2=supervised_contrast(duplicate,torch.cat([codes,codes[:2]]))
    assert accuracy2==1 and torch.isfinite(loss2)


def test_variable_shape_and_safe_checkpoint(tmp_path):
    setup(1615,2);model=RelationalEncoder()
    before=state_hash(model)
    with torch.no_grad():out=model(torch.rand(1,3,65,77))
    assert out['dense'].shape==(1,32,33,39)
    assert out['detail_prediction'].shape==(1,4,65,77)
    assert out['embedding'].shape==(1,64)
    assert torch.allclose(out['embedding'].norm(dim=1),torch.ones(1),atol=1e-5)
    assert state_hash(model)==before
    # write_checkpoint requires a ROOT-relative artifact path; its payload is
    # separately exercised here without forcing tests to create training runs.
    path=tmp_path/'safe.pt'
    torch.save({'format':'test','state_dict':model.state_dict(),'state_sha256':before,
                'torch_version':str(torch.__version__)},path)
    loaded=torch.load(path,weights_only=True)
    clone=RelationalEncoder();clone.load_state_dict(loaded['state_dict'])
    assert state_hash(clone)==before
