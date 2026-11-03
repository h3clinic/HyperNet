"""Independent census and sampled pixel replay of the completed v15 run.

Reads the completed artifact ledger; never updates model weights or the run.
This verifies stored evidence and sampled reproducibility, not a replay of
every optimization step or a proof of semantic diversity.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hypernet.detail_encoder import DetailEncoder,FORMAT,state_hash
from hypernet.math_curriculum_torch import sample_batch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def need(value,message):
    if not value:raise ValueError(message)
def unique(values):
    return int(len(np.unique(np.ascontiguousarray(values).reshape(-1,32).view('V32'))))


def audit(run,output,device):
    run=Path(run).resolve();output=Path(output).resolve()
    need(not output.exists(),'Audit output already exists; preserve prior evidence')
    complete=read(run/'completed.json');admission=read(run/'admission.json')
    need(complete['completed'] is True and complete['format']==FORMAT,'Not a completed v15 run')
    need(complete['admission_sha256']==sha(run/'admission.json'),'Admission changed')
    need(complete['source_sha256']==admission['source_sha256'],'Source declarations differ')
    for name,expected in admission['source_sha256'].items():
        need(sha(ROOT/name)==expected,'Current frozen training source changed: '+name)
        need(sha(run/'source_snapshot'/name)==expected,'Source snapshot changed: '+name)
    need(admission['natural_training_images']==complete['natural_training_images']==0,'Natural training was declared')
    need(admission['borrowed_pretrained_weights'] is False and complete['borrowed_pretrained_weights'] is False,'Borrowed weights declared')
    records=complete['counts']['shards'];all_ids=[];all_hashes=[]
    need(set(p.name for p in (run/'ledger').glob('*.npz'))==set(r['path'] for r in records),'Ledger shard inventory differs')
    for index,record in enumerate(records):
        path=run/'ledger'/record['path']
        need(record['path']==f'shard_{index:05d}.npz','Nonconsecutive ledger shards')
        need(sha(path)==record['sha256'],'Ledger shard changed: '+path.name)
        with np.load(path,allow_pickle=False) as data:
            ids=data['family_ids'].copy();digests=data['sha256'].copy()
        need(ids.dtype==np.int64 and digests.dtype==np.uint8,'Unexpected ledger dtypes')
        need(ids.shape==(record['families'],) and digests.shape==(len(ids),4,32),'Unexpected ledger dimensions')
        all_ids.append(ids);all_hashes.append(digests)
    ids=np.concatenate(all_ids);digests=np.concatenate(all_hashes)
    target=admission['target_families']
    need(target==1048576 and np.array_equal(ids,np.arange(target,dtype=np.int64)),'Expected each ID0..1048575 exactly once in order')
    census={'family_visits':len(ids),'unique_family_ids':int(len(np.unique(ids))),
            'rendered_training_views':2*len(ids),'unique_training_rgb_sha256':unique(digests[:,2:]),
            'unique_canonical_mask_sha256':unique(digests[:,0]),'unique_canonical_raster_sha256':unique(digests[:,1])}
    for key,value in census.items():need(value==complete['counts'][key],'Census disagrees: '+key)
    rows=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    expected_steps=(target+admission['batch_size']-1)//admission['batch_size']
    need(len(rows)==complete['optimizer_steps']==expected_steps,'Optimizer step count differs')
    for index,row in enumerate(rows):
        need(row['optimizer_step']==index+1 and row['consumed_families']==min((index+1)*admission['batch_size'],target),'Nonconsecutive optimizer history')
        need(all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float))),'Nonfinite optimizer history')
    final_path=run/f'families_{target:07d}.pt'
    final=torch.load(final_path,map_location='cpu',weights_only=True)
    initial=torch.load(run/'random_initial.pt',map_location='cpu',weights_only=True)
    model=DetailEncoder();model.load_state_dict(initial['state_dict'])
    need(state_hash(model)==admission['initial_state_sha256']==complete['initial_state_sha256']==initial['state_sha256'],'Initial state mismatch')
    model.load_state_dict(final['state_dict'])
    need(state_hash(model)==complete['final_state_sha256']==final['state_sha256'],'Final state mismatch')
    need(final['state_sha256']!=initial['state_sha256'],'Weights did not change')
    need(final['optimizer_steps']==expected_steps and final['consumed_families']==target,'Checkpoint count mismatch')
    need(final['ledger_shards']==records and final['source_sha256']==admission['source_sha256'],'Checkpoint evidence differs')
    need(final['admission']==admission and initial['admission']==admission,'Checkpoint admission differs')
    need(sum(p.numel() for p in model.parameters())==admission['parameter_count'],'Parameter count differs')
    # Include both ends and all declared scaling boundaries; no outcome selection.
    replay_ids=np.array([0,1,7,65535,65536,262143,262144,1048574,1048575],np.int64)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.inference_mode():batch=sample_batch(replay_ids,seed=admission['renderer_seed'],size=admission['side'],device=device)
    replays=[]
    keys=['canonical_mask','canonical_raster','view_a','view_b']
    for i,family in enumerate(replay_ids):
        columns={}
        for column,key in enumerate(keys):
            pixels=np.ascontiguousarray(batch[key][i].detach().cpu().numpy())
            actual=hashlib.sha256(pixels.tobytes()).digest();expected=digests[family,column].tobytes()
            need(actual==expected,f'Pixel replay differs: family{family}, {key}')
            columns[key]=actual.hex()
        replays.append({'family_id':int(family),'all_four_hashes_exact':True,'hashes':columns})
    result={'format':'hypernet.v15.independent_training_audit','valid':True,
            'created_utc':datetime.now(timezone.utc).isoformat(),'admission_sha256':sha(run/'admission.json'),
            'completed_sha256':sha(run/'completed.json'),'history_sha256':sha(run/'history.jsonl'),
            'final_checkpoint_sha256':sha(final_path),'final_state_sha256':state_hash(model),
            'audit_source_sha256':sha(__file__),'source_snapshots_and_current_sources_match':True,
            'census':census,'optimizer_steps':len(rows),'parameter_count':admission['parameter_count'],
            'replay_device':device,'sampled_pixel_replays':replays,
            'scope':'Independent ledger census and9sampled pixel replays. Does not replay all optimization steps or equate parameter families with semantic categories.',
            'natural_training_images':0,'borrowed_pretrained_weights':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'valid':True,'census':census,'output':str(output),'sha256':sha(output)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();audit(a.run,a.output,a.device)
