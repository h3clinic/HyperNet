"""Read-only training proof checks; optional exact replay of sampled inputs."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(path,expected):
    if sha(path)==expected:return
    manifest=read(ROOT/'PUBLIC_EXPORT_MANIFEST.json')
    key=str(Path(path).resolve().relative_to(ROOT)).replace('\\','/')
    binding=manifest['files'].get(key,{})
    if binding.get('original_sha256')!=expected or binding.get('exported_sha256')!=sha(path):raise ValueError('Bound artifact differs: '+str(path))
def resolve_run(value):
    path=Path(value);path=path.resolve()if path.is_absolute()else(ROOT/path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Run must be inside v19')
    return path

def verify_run(run,replay_count=0):
    run=resolve_run(run);completion=read(run/'completion.json');admission=read(run/'admission.json');config=completion['config']
    if completion['status']!='complete'or admission['config']!=config:raise ValueError('Training admission/completion differs')
    if completion['natural_training_images']!=0 or completion['borrowed_pretrained_weights']is not False or completion['deterministic_algorithms']is not True:
        raise ValueError('Training constraints differ')
    if completion['optimizer_steps']*config['batch_size']!=config['episodes']or completion['training_episodes']!=config['episodes']:
        raise ValueError('Fixed episode budget was not consumed')
    for name,expected in completion['source_sha256'].items():check(run/'source_snapshot'/name,expected)
    if completion['approval_plan_sha256']is not None:check(run/'approval_plan.json',completion['approval_plan_sha256'])
    stream=hashlib.sha256();step_count=0;next_index=0;positive=0;valid_points=0;supports=set();queries=set();programs=set();families={}
    sample_indices=set(np.linspace(0,config['episodes']-1,min(replay_count,config['episodes'])).round().astype(int).tolist())if replay_count else set()
    selected={}
    for shard in completion['ledger_shards']:
        path=ROOT/shard['path'];check(path,shard['sha256']);arrays=dict(np.load(path,allow_pickle=False))
        if arrays['indices'].shape!=(shard['steps'],config['batch_size']):raise ValueError('Ledger batch/step shape differs')
        for step in range(shard['steps']):
            record={key:value[step]for key,value in arrays.items()}
            if record['indices'].tolist()!=list(range(next_index,next_index+config['batch_size'])):raise ValueError('Actual episode sequence differs')
            if not all(np.isfinite(value).all()for value in record.values()):raise ValueError('Nonfinite recorded inputs')
            for key in sorted(record):
                a=np.ascontiguousarray(record[key]);stream.update(key.encode());stream.update(str(a.dtype).encode());stream.update(str(a.shape).encode());stream.update(a.tobytes())
            for j,index in enumerate(record['indices']):
                if int(index)in sample_indices:selected[int(index)]={key:value[j:j+1].copy()for key,value in record.items()}
            supports.update(bytes(a)for a in record['raw_support_sha256']);queries.update(bytes(a)for a in record['raw_query_sha256'])
            programs.update(bytes(a)for a in record['program_id']);positive+=int(record['positive'].sum());valid_points+=int(record['point_valid'].sum())
            for family in record['family']:families[str(int(family))]=families.get(str(int(family)),0)+1
            next_index+=config['batch_size'];step_count+=1
    if stream.hexdigest()!=completion['input_stream_sha256']:raise ValueError('Actual-consumption stream hash differs')
    measured={'optimizer_steps':step_count,'training_episodes':next_index,'positive_episodes':positive,'negative_episodes':next_index-positive,
      'valid_point_correspondences':valid_points,'unique_support_rgb':len(supports),'unique_query_rgb':len(queries),
      'unique_rgb_across_input_roles':len(supports|queries),'unique_structural_programs':len(programs),'family_episode_counts':families}
    for key,value in measured.items():
        if completion[key]!=value:raise ValueError('Completion counter differs: '+key)
    check(run/'history.jsonl',completion['history_sha256']);history_bytes=[]
    for shard in completion['history_shards']:
        path=ROOT/shard['path'];check(path,shard['sha256']);history_bytes.append(path.read_bytes())
    if b''.join(history_bytes)!=(run/'history.jsonl').read_bytes():raise ValueError('Atomic history shards differ from final history')
    rows=[json.loads(line)for line in(run/'history.jsonl').read_text().splitlines()]
    if[row['step']for row in rows]!=list(range(1,step_count+1)):raise ValueError('History steps incomplete')
    checkpoint_records=completion['checkpoints']
    if len(checkpoint_records)!=2 or checkpoint_records[0]['optimizer_steps']!=0 or checkpoint_records[-1]['optimizer_steps']!=step_count:
        raise ValueError('Expected exactly random and final checkpoints')
    for record in checkpoint_records:
        path=ROOT/record['path'];check(path,record['sha256']);saved=torch.load(path,map_location='cpu',weights_only=True)
        state=hashlib.sha256()
        for name,tensor in sorted(saved['state_dict'].items()):
            if not torch.isfinite(tensor).all():raise ValueError('Nonfinite checkpoint tensor')
            state.update(name.encode());state.update(tensor.contiguous().numpy().tobytes())
        if state.hexdigest()!=record['state_sha256']or saved['state_sha256']!=record['state_sha256']or saved['config']!=config:
            raise ValueError('Checkpoint state/config differs')
    if replay_count:
        for name,expected in completion['source_sha256'].items():check(ROOT/name,expected)
        from train_v19 import checked_collate,ledger_record
        from hypernet_v19.curriculum import episode
        for index,expected in selected.items():
            row=episode(index,split='train',seed=config['stream_seed'],difficulty=config['difficulty']);actual=ledger_record(checked_collate([row]))
            if set(actual)!=set(expected):raise ValueError('Replay ledger schema differs')
            for key in actual:
                if not np.array_equal(actual[key],expected[key]):raise ValueError(f'Exact input replay differs at episode{index}: {key}')
    return {'valid':True,'run':str(run.relative_to(ROOT)).replace('\\','/'),'completion_sha256':sha(run/'completion.json'),
      **measured,'input_stream_sha256':stream.hexdigest(),'exact_replayed_episode_indices':sorted(selected),
      'checkpoint_count':2,'history_and_ledger_shards_verified':True,'natural_images_read':0,'gpu_used':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--replay-count',type=int,default=0);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.replay_count<0:p.error('Replay count must be nonnegative')
    result=verify_run(a.run,a.replay_count)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
