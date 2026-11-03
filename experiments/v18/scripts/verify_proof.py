"""Verify every saved v18 proof input record/checkpoint/source hash on CPU."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from hypernet_v18.reranker import FrozenPatchReranker
from hypernet_v18.pair_ranker import INPUT_KEYS,file_sha
from train_pair_ranker import atomic_json


def main():
    run=ROOT/'runs/proof512';done=json.loads((run/'completion.json').read_text());stream=hashlib.sha256()
    assert done['status']=='complete'and done['optimizer_steps']==512
    indices=[];images=set();programs=set()
    for shard in done['ledger_shards']:
        path=ROOT/shard['path'];assert file_sha(path)==shard['sha256']
        with np.load(path,allow_pickle=False)as z:arrays={key:z[key]for key in z.files}
        assert len(arrays['indices'])==shard['steps']
        for step in range(len(arrays['indices'])):
            for key in sorted(arrays):stream.update(key.encode());stream.update(arrays[key][step].tobytes())
        indices.extend(arrays['indices'].reshape(-1).tolist())
        for key in INPUT_KEYS:images.update(row.tobytes()for row in arrays[key+'_sha256'].reshape(-1,32))
        programs.update(row.tobytes()for row in arrays['program_id'].reshape(-1,32))
    assert indices==list(range(4096)) and done['training_quartets']==4096 and done['training_rgb_views']==16384
    assert stream.hexdigest()==done['input_stream_sha256']
    assert len(images)==done['unique_input_images']and len(programs)==done['unique_target_programs']
    for name,digest in done['source_sha256'].items():
        assert file_sha(run/'source_snapshot'/name)==digest
        if file_sha(ROOT/name)!=digest:
            manifest=json.loads((ROOT/'PUBLIC_EXPORT_MANIFEST.json').read_text())
            binding=next(item for item in manifest['files'] if item['path']==name)
            assert binding['original_sha256']==digest and binding['exported_sha256']==file_sha(ROOT/name)
    own=ROOT.parent/'v17/runs/primary_2048/final.pt'
    for saved in done['checkpoints']:
        path=ROOT/saved['path'];assert file_sha(path)==saved['sha256']
        model=FrozenPatchReranker(path,own,device='cpu');model.assert_unchanged();assert model.initial_state==saved['state_sha256']
    result={'status':'passed','saved_ledger_complete':True,'source_snapshots_unchanged':True,'checkpoint_hashes_verified':True,
        'training_quartets':4096,'additional_processed_rgb_views':16384,'unique_v18_input_images':len(images),
        'unique_target_programs':len(programs),'cumulative_own_processed_rgb_exposures':done['cumulative_own_processed_rgb_views'],
        'input_stream_sha256':stream.hexdigest(),'completion_sha256':file_sha(run/'completion.json'),
        'scope':'Saved-ledger integrity; not exhaustive image regeneration or proof of semantic type identifiability',
        'source_sha256':file_sha(__file__)}
    output=run/'verification.json'
    if output.exists():raise FileExistsError(output)
    atomic_json(output,result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
