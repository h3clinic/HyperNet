"""CPU integrity check of completed matched-arm optimizer-consumption ledgers."""
from pathlib import Path
import argparse,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from hypernet.local_matcher import FrozenLocalMatcher


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run):
    run=Path(run);done=json.loads((run/'completion.json').read_text())
    assert done['status']=='complete' and done['natural_training_images']==0 and done['borrowed_pretrained_weights']is False
    config=done['config'];stream=hashlib.sha256();seen=0;steps=0;arrays=[]
    programs=set();supports=set();queries=set();positive_count=0
    for shard in done['ledger_shards']:
        path=ROOT/shard['path'];assert sha(path)==shard['sha256']
        with np.load(path,allow_pickle=False)as file:data={k:file[k]for k in file.files}
        count=len(data['indices']);assert count==shard['steps']
        assert data['indices'].shape[1]==config['batch_size']
        assert np.array_equal(data['indices'].reshape(-1),np.arange(seen,seen+data['indices'].size))
        for s in range(count):
            for key in sorted(data):stream.update(key.encode());stream.update(data[key][s].tobytes())
        programs.update(row.tobytes()for row in data['program_sha256'].reshape(-1,32))
        supports.update(row.tobytes()for row in data['support_sha256'].reshape(-1,32))
        queries.update(row.tobytes()for row in data['query_sha256'].reshape(-1,32))
        positive_count+=int(data['present'].sum())
        seen+=data['indices'].size;steps+=count;arrays.append(data)
    assert seen==done['training_episodes']==config['steps']*config['batch_size']
    assert steps==done['optimizer_steps']==config['steps']
    assert done['training_views']==2*seen and stream.hexdigest()==done['input_stream_sha256']
    assert positive_count==done['positive_episodes'] and seen-positive_count==done['negative_episodes']
    assert len(programs)==done['unique_structural_programs']
    assert len(supports)==done['unique_support_rgb'] and len(queries)==done['unique_query_rgb']
    assert len(supports|queries)==done['unique_rgb_across_input_roles']
    for source,digest in done['source_sha256'].items():
        assert sha(run/'source_snapshot'/source)==digest
        assert sha(ROOT/source)==digest
    for checkpoint in done['checkpoints']:
        path=ROOT/checkpoint['path'];assert sha(path)==checkpoint['sha256']
        model=FrozenLocalMatcher(path,device='cpu');model.assert_unchanged()
        assert model.initial_state==checkpoint['state_sha256']
    final=done['checkpoints'][-1]
    assert final['optimizer_steps']==config['steps'] and final['state_sha256']==done['state_sha256']
    return done,arrays


def main(args):
    output=ROOT/args.output
    if output.exists():raise FileExistsError(output)
    a,aa=verify(ROOT/args.primary);b,bb=verify(ROOT/args.bag)
    assert a['config']['geometry']is True and a['config']['variant']=='primary'
    assert b['config']['geometry']is False and b['config']['variant']=='bag'
    assert {k:v for k,v in a['config'].items()if k not in ('geometry','variant')}=={k:v for k,v in b['config'].items()if k not in ('geometry','variant')}
    assert a['source_sha256']==b['source_sha256'] and a['input_stream_sha256']==b['input_stream_sha256']
    assert a['checkpoints'][0]['optimizer_steps']==b['checkpoints'][0]['optimizer_steps']==0
    assert a['checkpoints'][0]['state_sha256']==b['checkpoints'][0]['state_sha256']
    assert len(aa)==len(bb)
    for left,right in zip(aa,bb):
        assert left.keys()==right.keys()
        assert all(np.array_equal(left[k],right[k])for k in left)
    result={'status':'passed','scope':'Full saved-consumption ledger integrity and matched-arm equality; not full image regeneration',
            'steps_per_arm':a['optimizer_steps'],'episodes_per_arm':a['training_episodes'],'rgb_views_per_arm':a['training_views'],
            'input_stream_sha256':a['input_stream_sha256'],'initial_state_sha256':a['checkpoints'][0]['state_sha256'],
            'same_initial_tensors':True,'same_optimizer_input_ledger':True,'same_source_snapshots':True,
            'geometry_roles_verified':True,'source_sha256':a['source_sha256'],
            'primary_completion_sha256':sha(ROOT/args.primary/'completion.json'),
            'bag_completion_sha256':sha(ROOT/args.bag/'completion.json'),
            'verification_source_sha256':sha(__file__)}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',default='runs/primary_2048');p.add_argument('--bag',default='runs/bag_2048')
    p.add_argument('--output',default='runs/matched_pair_verification.json');main(p.parse_args())
