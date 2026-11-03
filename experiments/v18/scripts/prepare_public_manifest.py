"""Prepare reviewable local publication metadata; never copies or publishes."""
from pathlib import Path
import sys,json,hashlib,re,platform,importlib.metadata as metadata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def entry(path):return {'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha(path),'bytes':path.stat().st_size}


def absolute_fields(value,path='root'):
    found=[]
    if isinstance(value,str)and re.search(r'^[A-Za-z]:[\\/]|^/(Users|home|mnt|tmp)/',value):found.append(path)
    elif isinstance(value,dict):
        for key,item in value.items():found.extend(absolute_fields(item,path+'.'+str(key)))
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):found.extend(absolute_fields(item,path+'.'+str(index)))
    return found


def main():
    provenance_path=ROOT/'PUBLIC_PROVENANCE.json';allowlist_path=ROOT/'public_source_allowlist.json'
    if provenance_path.exists()or allowlist_path.exists():raise FileExistsError('Preserve prior review metadata')
    done=read(ROOT/'runs/proof512/completion.json');audit=read(ROOT/'runs/proof512/verification.json')
    pair=read(ROOT/'experiments/proof512_eval/pair_results.json');scene=read(ROOT/'experiments/proof512_eval/scene_results.json')
    real=read(ROOT/'experiments/real_dev_pilot/result.json')
    checkpoints=[]
    for name in ('initial.pt','final.pt'):
        path=ROOT/'runs/proof512'/name;saved=torch.load(path,map_location='cpu',weights_only=True)
        fields=absolute_fields({k:v for k,v in saved.items()if k!='state_dict'})
        if fields:raise RuntimeError('Checkpoint normalization requires separate exported copy: '+str(fields))
        checkpoints.append({**entry(path),'state_sha256':saved['state_sha256'],'metadata_absolute_paths_found':[],
                            'checkpoint_rewritten':False,'training':saved['training']})
    own=ROOT.parent/'v17/runs/primary_2048/final.pt'
    own_value=torch.load(own,map_location='cpu',weights_only=True)
    dependency={'workspace_path':'experiments/v17/runs/primary_2048/final.pt',
        'suggested_public_path':'checkpoints/own_v17_proposal.pt','sha256':sha(own),'state_sha256':own_value['state_sha256'],
        'purpose':'Own synthetic-only encoder initialization and separately frozen proposal stage',
        'natural_training_images':0,'borrowed_pretrained_weights':False}
    provenance={'version':'HyperNet v18 fixed512-update proof','status':'Bounded proof complete; no demonstrated real detection improvement',
        'model_format':'hypernet.pair_ranker.v18.1','checkpoints':checkpoints,'own_v17_dependency':dependency,
        'training':{'optimizer_steps':512,'batch_size':8,'quartets':4096,'additional_rgb_views':16384,
                    'unique_v18_input_images':done['unique_input_images'],'unique_target_programs':done['unique_target_programs'],
                    'prior_own_rgb_views':65536,'cumulative_processed_exposures':81920,
                    'cumulative_unique_image_count':'Not claimed; prior and current stages were not jointly deduplicated',
                    'natural_training_images':0,'borrowed_pretrained_weights':False,
                    'seconds':done['seconds'],'peak_cuda_bytes':done['peak_cuda_bytes'],
                    'ledger_input_stream_sha256':done['input_stream_sha256']},
        'runtime':{'python':platform.python_version(),'platform':platform.system(),
                   'packages':{name:metadata.version(name)for name in ('torch','numpy','opencv-python','Pillow','pytest')}},
        'source_sha256':done['source_sha256'],'ledger_verification':audit['status'],
        'synthetic_pair_test':pair['splits']['test']['metrics'],
        'same_scene_synthetic_test':scene['splits']['test']['metrics'],
        'natural_pilot':{'scope':'Old DEVELOPMENT only; descriptive within-pilot thresholds, not fresh validation',
                         'photos':48,'references':12,'nominal_pairs':576,'metrics':real['metrics'],
                         'proposal_recall_at100':real['proposal_recall_at100'],
                         'v17_fresh_real_TEST_access':False,'real_weight_updates':0},
        'limitations':['Coarse wrong-family rejection is much stronger than one-edit type discrimination.',
                      'No claim of beating OWL-ViT or establishing algorithmic novelty.',
                      'Visible paired differences do not prove type manifolds remain disjoint under all nuisance variations.',
                      'Positive patch labels mean type presence, not tight-boxIoU quality.',
                      'No raw natural photographs are included in the allowlist.']}
    provenance_path.write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    source=[]
    for folder in ('hypernet_v18','scripts','tests'):
        source.extend(p for p in (ROOT/folder).rglob('*.py')if '__pycache__'not in p.parts)
    source.extend([ROOT/'README.md',ROOT/'requirements.txt'])
    artifacts=[provenance_path,ROOT/'assets/pair_preflight.json']
    for relative in ('runs/proof512/admission.json','runs/proof512/completion.json','runs/proof512/verification.json',
                     'runs/proof512/history.jsonl','runs/proof512/initial.pt','runs/proof512/final.pt',
                     'experiments/proof512_eval/plan.json','experiments/proof512_eval/pair_results.json','experiments/proof512_eval/scene_results.json',
                     'experiments/real_dev_pilot/admission.json','experiments/real_dev_pilot/result.json',
                     'experiments/real_dev_pilot/arrays.npz','experiments/real_dev_pilot/predictions.jsonl'):
        artifacts.append(ROOT/relative)
    artifacts.extend((ROOT/'runs/proof512/ledger').glob('*.npz'))
    artifacts.extend((ROOT/'runs/proof512/source_snapshot').rglob('*.py'))
    allowlist={'status':'Prepared locally for root review; nothing copied to external repository or published',
        'source_files':[entry(p)for p in sorted(source)],'audited_artifacts':[entry(p)for p in sorted(set(artifacts))],
        'explicit_own_checkpoint_dependency':dependency,
        'excluded':['Natural photographs and galleries','data/real_dev_only.json workspace-layout manifest',
                    'v17 fresh real TEST records/images/predictions','temporary smokes and discarded fit weights','__pycache__'],
        'export_rules':['Do not mutate original trained checkpoints or frozen source snapshots.',
                        'The v18 checkpoint metadata contains no absolute filesystem paths; preserve bytes and SHA256.',
                        'If any future metadata normalization is required, edit only an exported copy and record original/exported SHA256 plus verified identical tensor state hash.',
                        'Preserve negative results and DEVELOPMENT labels; do not present cropped-pair AUROC as scene detection accuracy.']}
    allowlist_path.write_text(json.dumps(allowlist,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'source_files':len(source),'audited_artifacts':len(set(artifacts)),
                      'provenance_sha256':sha(provenance_path),'allowlist_sha256':sha(allowlist_path)}),flush=True)

if __name__=='__main__':main()
