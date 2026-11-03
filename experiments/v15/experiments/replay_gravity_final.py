"""Exact independent inference replay of the two predeclared fresh GS events.

No calibration, metric arithmetic, recipe selection, cache reuse, or training.
Require all four primary runs to be complete before constructing any model.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import argparse,json,sys,time,math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.detail_features import FrozenDetails
from hypernet.pattern_matcher import PatternMatcher,RECIPES,raw_pixel_vector
from hypernet.gravity_spy_images import GravitySpyCohort
from hypernet.gravity_spy_admission import verify_test_admission

DATA=ROOT/'data/gravity_spy_v15_final'
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(condition,message):
    if not condition:raise ValueError(message)
def bank_fingerprint(bank):
    value=sha256()
    for name in ('descriptors','analytic','xy','weights','cells'):
        array=np.ascontiguousarray(getattr(bank,name));value.update(name.encode());value.update(str(array.dtype).encode());value.update(str(array.shape).encode());value.update(array.tobytes())
    value.update(bank.reference_rgb_sha256.encode());value.update(str(bank.source_shape).encode())
    return value.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--result',action='append',required=True,help='Four explicit model=results.json bindings')
    args=parser.parse_args();start=time.perf_counter();output=ROOT/'experiments/gravity_spy_inference_replay.json'
    require(not output.exists(),'Preserve existing replay')
    plan_path=ROOT/'experiments/gravity_spy_replay_plan.json';roster_path=ROOT/'experiments/gravity_spy_exact_replay_roster.json'
    plan=read(plan_path);selection=read(roster_path);protocol_path=ROOT/'experiments/gravity_spy_protocol_freeze.json';protocol=read(protocol_path)
    before={path:digest(path)for path in (plan_path,roster_path,protocol_path,DATA/'quality_test_audit.json',DATA/'plan.json',
        DATA/'prior_exposure_admission.json',ROOT/'experiments/gravity_spy_final_data_independent_audit.json')}
    before[Path(__file__).resolve()]=digest(__file__)
    require(plan['declared_before_any_test_scores']and selection['declared_before_any_test_scores'],'Replay was not predeclared')
    require(selection['selection_plan_sha256']==digest(plan_path)and selection['blind_test_quality_sha256']==digest(DATA/'quality_test_audit.json'),'Selection binding changed')
    require(plan['protocol_sha256']==digest(protocol_path),'Protocol changed')
    require(plan['device']=='cuda'and plan['batch_size']==64 and plan['sampling_stride']==4 and plan['threads']==4 and plan['tolerance']==0,'Replay configuration differs')
    admission=ROOT/'experiments/gravity_spy_test_admission.json';verify_test_admission(DATA,admission);before[admission]=digest(admission)
    for field in ('behavior_files','checkpoint_files'):
        for name,expected in protocol[field].items():
            path=(ROOT/name).resolve();require(digest(path)==expected,'Frozen binding changed '+name);before[path]=expected
    raw=read(DATA/'plan.json');prior=read(DATA/'prior_exposure_admission.json');qc=read(DATA/'quality_test_audit.json')
    valid=set(prior['primary_test_event_ids'])-set(qc['excluded_event_ids'])
    effective=[event for event in raw['events']if event['split']=='test'and event['event_id']in valid]
    selected=selection['events'];ids=[event['event_id']for event in selected]
    require(ids==[event['event_id']for event in effective[:2]]and len(ids)==2,'Fixed first-two roster differs')
    for chosen,original in zip(selected,effective):
        require(all(chosen[key]==value for key,value in original.items()),'Selected metadata changed')
    bindings={}
    for item in args.result:
        alias,path=item.split('=',1);require(alias not in bindings,'Duplicate result alias');bindings[alias]=Path(path).resolve()
    require(set(bindings)==set(plan['models'])==set(protocol['models']),'All four completed runs required')
    completed={};saved={}
    for alias,path in bindings.items():
        result=read(path);before[path]=digest(path)
        require(result['completed']and result['mode']=='test'and result['model_state_unchanged'],'Primary TEST run incomplete')
        require(result['events']==len(effective)and set(result['methods'])==set(protocol['models'][alias]['test_methods']),'Run roster/methods differ')
        require(result['calibration_results_sha256']==digest(ROOT/protocol['models'][alias]['result_path']),'Calibrated run binding differs')
        require(all(digest(ROOT/name)==value for name,value in result['source_sha256'].items()),'Saved inference source binding differs')
        scorepath=path.parent/'event_scores.jsonl';require(digest(scorepath)==result['scores_sha256'],'Saved score log changed');before[scorepath]=digest(scorepath)
        keep={};order=[]
        with scorepath.open(encoding='utf-8')as stream:
            for line in stream:
                row=json.loads(line);order.append(row['event_id'])
                if row['event_id']in ids:keep[row['event_id']]=row
        require(order==[event['event_id']for event in effective]and set(keep)==set(ids),'Saved TEST order/IDs differ')
        completed[alias]=result;saved[alias]=keep
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cohort=GravitySpyCohort(DATA,admission);targets=cohort.target_labels
    require(len(targets)==19,'Target count changed')
    numeric_count=0;maxerror=0.;mismatches=[];comparisons=[];provenance=[]
    def compare(actual,expected,path):
        nonlocal numeric_count,maxerror
        if isinstance(expected,dict):
            require(isinstance(actual,dict)and set(actual)==set(expected),'Prediction keys differ '+path)
            for key in expected:compare(actual[key],expected[key],path+'/'+key)
        elif isinstance(expected,list):
            require(isinstance(actual,list)and len(actual)==len(expected),'Prediction list differs '+path)
            for index,(a,b)in enumerate(zip(actual,expected)):compare(a,b,path+'/'+str(index))
        elif isinstance(expected,(int,float))and not isinstance(expected,bool):
            require(isinstance(actual,(int,float))and not isinstance(actual,bool)and math.isfinite(actual),'Invalid numeric prediction '+path)
            numeric_count+=1;error=abs(float(actual)-float(expected));maxerror=max(maxerror,error)
            if error and len(mismatches)<100:mismatches.append({'path':path,'actual':actual,'expected':expected,'absolute_error':error})
        elif actual!=expected and len(mismatches)<100:mismatches.append({'path':path,'actual':actual,'expected':expected})
    for alias in plan['models']:
        result=completed[alias];require(result['target_labels']==list(targets),'Target ordering differs')
        checkpoint=(ROOT/protocol['models'][alias]['checkpoint_path']).resolve()
        extractor=FrozenDetails(checkpoint,device='cuda',batch_size=64);initial=extractor.provenance()
        require(initial==result['checkpoint_provenance'],'Loaded model differs from saved run')
        require(not extractor.model.training and all(not parameter.requires_grad for parameter in extractor.model.parameters()),'Model is trainable')
        matcher=PatternMatcher(extractor);images=[cohort.reference(label)for label in targets]
        banks=[matcher.enroll(image)for image in images];raw_refs=np.stack([raw_pixel_vector(image)for image in images])
        bank_hashes=[bank_fingerprint(bank)for bank in banks]
        identity=[{'label':label,'rgb_sha256':bank.reference_rgb_sha256,'anchors':len(bank.xy)}for label,bank in zip(targets,banks)]
        require(identity==result['reference_banks'],'Reference bank identity differs')
        methods=protocol['models'][alias]['test_methods'];kinds=('learned','analytic')if any(method.startswith('analytic:')for method in methods)else('learned',)
        for event in selected:
            event_id=event['event_id'];rgb=cohort.load_rgb(event_id);pixel_sha=sha256(rgb.tobytes()).hexdigest()
            require(pixel_sha==event['crop_rgb_sha256']==saved[alias][event_id]['pixel_sha256'],'Query pixels changed')
            encoded=extractor.encode(rgb,sampling_stride=4);predicted={}
            for kind in kinds:
                values=matcher.score_many(banks,encoded,recipes=RECIPES,feature_kind=kind)
                for recipe in RECIPES:
                    method=kind+':'+recipe;predicted[method]={}
                    for label,item in zip(targets,values):
                        item=item[recipe];keep=('anchor_count','contributing_anchors','reference_cells','covered_reference_cells','reference_cell_coverage','translation_xy','abstain')
                        predicted[method][label]={'score':item['score'],'evidence':{key:value for key,value in item['evidence'].items()if key in keep}}
            raw_scores=raw_refs@raw_pixel_vector(rgb)
            predicted['raw_pixels48']={label:{'score':float(value),'evidence':{'control':'fixed48x48gray centeredL2 dot; noCNN'}}for label,value in zip(targets,raw_scores)}
            require(set(predicted)==set(methods),'Replayed method roster differs')
            compare(predicted,saved[alias][event_id]['scores'],alias+'/'+event_id)
            comparisons.append({'model':alias,'event_id':event_id,'methods':methods,'target_predictions':len(methods)*len(targets),'pixel_sha256':pixel_sha})
        require(extractor.provenance()==initial and [bank_fingerprint(bank)for bank in banks]==bank_hashes,'Model/reference bank changed')
        provenance.append({'model':alias,'checkpoint_provenance':initial,'reference_bank_hashes':bank_hashes,'state_unchanged':True})
        del banks,images,raw_refs,matcher,extractor;torch.cuda.empty_cache()
        print(json.dumps({'replayed_model':alias,'queries':2,'target_predictions':2*len(methods)*len(targets)}),flush=True)
    for path,expected in before.items():require(digest(path)==expected,'Bound artifact changed during replay '+str(path))
    verify_test_admission(DATA,admission)
    record={'format':'hypernet.gravity_spy.inference_replay.v15','valid':not mismatches,'created_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':digest(__file__),'elapsed_seconds':time.perf_counter()-start,'query_ids':ids,'model_queries':len(comparisons),
        'target_prediction_comparisons':sum(item['target_predictions']for item in comparisons),'numeric_values_compared':numeric_count,
        'maximum_absolute_error':maxerror,'tolerance':0,'mismatches':mismatches,'comparisons':comparisons,'models':provenance,
        'bound_artifacts':{str(path):value for path,value in before.items()},'prediction_cache_reads_or_writes':False,
        'gradient_enabled':torch.is_grad_enabled(),'training_or_calibration_calls':0,'original_sources_models_data_results_unchanged':True,
        'scope':'Exactly the two predeclared QC-valid TEST events, all four checkpoints, every frozen method and19target scores/evidence. Independent metric arithmetic is a separate parent-owned audit.'}
    with output.open('x',encoding='utf-8')as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'path':str(output),'sha256':digest(output),'valid':record['valid'],'target_predictions':record['target_prediction_comparisons'],'max_error':maxerror}))
    require(record['valid'],'Replay mismatch; preserve this failure without tuning')
if __name__=='__main__':main()
