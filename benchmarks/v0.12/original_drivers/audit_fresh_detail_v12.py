"""Independent fresh-data metrics, paired uncertainty and exact six-method replay."""
from datetime import datetime,timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT.parent/'hypernet_v11'
DATA=ROOT/'data/shapes2d_fresh_v12'
RESULTS=ROOT/'results/fresh_eval'
OUTPUT=ROOT/'experiments/fresh_detail_run_audit.json'
digest=lambda path:sha256(Path(path).read_bytes()).hexdigest()
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))

# Reuse the separately tested independent reviewer, never production metrics.
helper_path=PREVIOUS/'scripts/audit_2d_detector.py'
assert digest(helper_path)=='3cf0235827e377424a851dd5bad5b7bba01aa214322bacea70fb73df00621fdb'
spec=importlib.util.spec_from_file_location('frozen_independent_metrics_v11',helper_path)
independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
sys.path.insert(0,str(ROOT))


def require(condition,message):
    if not condition:raise ValueError(message)


def state_digest(model):
    h=sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def bounds(values):
    values=np.sort(values)
    output=[]
    for fraction in (.025,.975):
        position=(len(values)-1)*fraction;low=int(position);high=min(low+1,len(values)-1)
        output.append(float(values[low]+(position-low)*(values[high]-values[low])))
    return output


def verify_bootstrap(rows,cutoffs,reported):
    families=sorted({r['family_id'] for values in rows.values() for r in values if r['split']=='test'})
    require(len(families)==128 and reported['families']==128 and reported['samples']==10000 and reported['seed']==1812,
            'Bootstrap cluster roster/protocol differs')
    require(reported['unit']=='family' and reported['metric']=='localized_recall_at_iou50','Bootstrap metric differs')
    draws=np.random.default_rng(1812).integers(0,128,(10000,128))
    distributions={};values={};verified={};checks=0
    for alias,records in rows.items():
        parts=[]
        for family in families:
            positive=[r for r in records if r['split']=='test' and r['family_id']==family and r['present']]
            require(len(positive)==2,'Expected two positive enrollments per family')
            parts.append(sum(r['score']>=cutoffs[alias] and r['iou']>=.5 for r in positive)/2)
        values[alias]=np.array(parts)
        distributions[alias]=np.sum(values[alias][draws],axis=1)/128
        actual={'estimate':float(sum(parts)/128),'ci95':bounds(distributions[alias])}
        require(abs(actual['estimate']-reported['methods'][alias]['estimate'])<1e-12,'Bootstrap method point mismatch')
        require(np.max(np.abs(np.array(actual['ci95'])-reported['methods'][alias]['ci95']))<1e-12,'Bootstrap method interval mismatch')
        checks+=3;verified[alias]=actual
    differences={}
    for control in ('old_cnn','lowres_patch'):
        name='native_patch_minus_'+control
        actual={'estimate':float(np.sum(values['native_patch']-values[control])/128),
                'ci95':bounds(distributions['native_patch']-distributions[control])}
        require(abs(actual['estimate']-reported['differences'][name]['estimate'])<1e-12,'Paired point mismatch')
        require(np.max(np.abs(np.array(actual['ci95'])-reported['differences'][name]['ci95']))<1e-12,'Paired interval mismatch')
        checks+=3;differences[name]=actual
    return {'verified_values':checks,'families':128,'samples':10000,'seed':1812,'methods':verified,'differences':differences}


def main():
    import torch
    from PIL import Image
    from hypernet.detail_pipeline import DetailPipeline,METHODS,PRIMARY,CHECKPOINT_SHA256
    from hypernet.one_shot_detector import detect as legacy_detect
    result_path=RESULTS/'fresh_results.json';results=read(result_path)
    manifest_path=DATA/'manifest.json';manifest=read(manifest_path)
    data_audit_path=ROOT/'experiments/fresh_data_audit.json';data_audit=read(data_audit_path)
    admission_path=ROOT/'experiments/fresh_generation_admission.json';admission=read(admission_path)
    protocol_path=Path(admission['protocol_path']);protocol=read(protocol_path)
    replay_path=ROOT/'experiments/fresh_replay_plan.json';replay=read(replay_path)
    require(results['completed'] and manifest['completed'] and data_audit['valid'],'Completed audited evaluation required')
    require(results['scope']=='FROZEN FRESH 2D SYNTHETIC EVALUATION','Wrong evaluation scope')
    require(digest(manifest_path)==results['manifest_sha256']==data_audit['manifest_sha256'],'Manifest binding differs')
    require(digest(protocol_path)==results['protocol_sha256']==admission['protocol_sha256']==data_audit['model_protocol_sha256'],
            'Protocol bindings differ')
    require(protocol['model_selection_complete'] and protocol['fresh_test_generated_before_freeze'] is False,
            'Fresh generation preceded selected model/protocol freeze')
    require(admission['no_fresh_images_before_admission'] and admission['model_selection_complete_before_generation'],
            'Fresh generation admission failed')
    require(digest(DATA/'plan.json')==protocol['fresh_plan_sha256']==replay['fresh_plan_sha256'],'Fresh plan binding differs')
    require(replay['fresh_images_existed_at_selection'] is False and replay['apply_to_all_protocol_methods'] is True,
            'Replay selection must precede fresh generation and cover all methods')
    require(set(results['methods'])==set(METHODS) and results['primary_method']==PRIMARY=='native_patch','Method roster differs')
    require(set(protocol['methods'])==set(METHODS) and protocol['primary_method']==PRIMARY,'Frozen method selection differs')
    require(results['learned_weight_updates_this_version']==0 and results['reference_images_per_enrollment']==1,
            'One-reference/read-only budget violated')
    require(results['natural_training_images']==0 and results['externally_pretrained_weights'] is False,'External training budget violated')
    require(results['gradients_created'] is False and results['model_state_before']==results['model_state_after'],'Evaluation mutated CNN')
    protected={str(p):digest(p) for p in (result_path,manifest_path,data_audit_path,admission_path,protocol_path,replay_path,
                                         DATA/'plan.json',DATA/'prior_rgb_inventory.json',helper_path)}
    for path,expected in protocol['behavior_files'].items():
        path=Path(path) if Path(path).is_absolute() else ROOT/path
        require(digest(path)==expected,'Frozen scoring behavior changed: '+str(path));protected[str(path)]=expected
    for path,expected in protocol['development_evidence'].items():
        path=Path(path) if Path(path).is_absolute() else ROOT/path
        require(digest(path)==expected,'Frozen pre-fresh development evidence changed');protected[str(path)]=expected
    data_freeze_path=ROOT/'experiments/fresh_data_freeze.json';data_freeze=read(data_freeze_path)
    protected[str(data_freeze_path)]=digest(data_freeze_path)
    require(digest(data_freeze_path)==admission['data_freeze_sha256']==replay['fresh_data_freeze_sha256']==protocol['fresh_data_freeze_sha256'],'Data freeze differs')
    for group in ('behavior_files','prior_evidence_files'):
        for path,expected in data_freeze[group].items():
            require(digest(path)==expected,'Frozen generation/prior evidence changed: '+path);protected[path]=expected
    checkpoint=PREVIOUS/'checkpoints/detector_micro_aware.pt'
    require(digest(checkpoint)==CHECKPOINT_SHA256==results['checkpoint_sha256'],'Own v11 checkpoint changed')
    protected[str(checkpoint)]=digest(checkpoint)
    for file in ('one_shot_detector.py','training_v3.py'):
        require(digest(ROOT/'hypernet'/file)==digest(PREVIOUS/'hypernet'/file),'Original CNN baseline code changed')
    annotations={q['id']:(e,q) for e in manifest['episodes'] for q in e['queries']}
    require(len(annotations)==1024 and len(manifest['episodes'])==512,'Fresh-only roster differs')
    require(all(e['split'] in ('dev','test') for e in manifest['episodes']),'Fresh training rows found')
    rows={};row_maps={};cutoffs={};metric_checks=localization_checks=0;method_metrics={}
    image_paths={}
    for episode in manifest['episodes']:
        for record in [episode['support']]+episode['queries']:
            image_paths[str(DATA/record['rgb_path'])]=record['rgb_sha256']
    require(len(image_paths)==1024,'Unexpected unique fresh RGB count')
    for path,expected in image_paths.items():require(digest(path)==expected,'Fresh source pixel file changed')
    for alias in METHODS:
        score_path=RESULTS/results['score_files'][alias]['path']
        require(digest(score_path)==results['score_files'][alias]['sha256'],'Saved score file changed')
        protected[str(score_path)]=digest(score_path)
        records=[json.loads(line) for line in score_path.read_text(encoding='utf-8').splitlines() if line]
        require(len(records)==1024 and {r['id'] for r in records}==set(annotations),'Saved score roster differs')
        for row in records:
            episode,target=annotations[row['id']]
            require(row['method']==alias and row['split']==episode['split'] and row['episode_id']==episode['episode_id'],
                    'Saved score identity differs')
            require(row['family_id']==episode['family_id'] and row['enrolled_variant']==episode['enrolled_variant'] and
                    row['present']==target['present'],'Saved score label binding differs')
            require(row['support_path']==episode['support']['rgb_path'] and row['query_path']==target['rgb_path'],'Saved RGB binding differs')
            gt=target['target_bbox_xyxy']
            actual_iou=independent.overlap(row['bbox_xyxy'],gt)
            actual_proposal=max((independent.overlap(box,gt) for box in row['proposals']),default=0.)
            require(abs(actual_iou-row['iou'])<1e-12 and abs(actual_proposal-row['proposal_best_iou'])<1e-12,'Saved localization differs')
            require(len(row['proposals'])==len(row['proposal_scores'])==len(row['all_evidence']),'Proposal evidence length mismatch')
            require(row['proposal_scores']==[e['score'] for e in row['all_evidence']],'Per-proposal score evidence differs')
            if row['proposals']:
                index=int(np.argmax(row['proposal_scores']))
                require(row['score']==row['proposal_scores'][index] and row['bbox_xyxy']==row['proposals'][index] and
                        row['evidence']==row['all_evidence'][index],'Selected top proposal differs')
            else:
                require(row['bbox_xyxy'] is None and row['evidence']=={} and row['score']==(-1. if alias=='old_cnn' else 0.),
                        'Empty-proposal fallback differs')
            localization_checks+=1
        rows[alias]=records;row_maps[alias]={r['id']:r for r in records}
        cutoff=independent.dev_cutoff([r for r in records if r['split']=='dev']);cutoffs[alias]=cutoff
        method_metrics[alias]={}
        for split in ('dev','test'):
            computed=independent.independent_metrics([r for r in records if r['split']==split],cutoff)
            for key,value in computed.items():
                require(abs(value-results['methods'][alias][split][key])<1e-12,'Metric differs: '+alias+'/'+split+'/'+key)
                metric_checks+=1
            method_metrics[alias][split]=computed
        require(method_metrics[alias]['dev']['false_positive_rate']<=.05,'Fresh DEV FPR target exceeded')
    control_checks=0
    for query_id in annotations:
        baseline=row_maps['old_cnn'][query_id]
        for alias in METHODS:require(row_maps[alias][query_id]['proposals']==baseline['proposals'],'Methods use different CNN proposals')
        for native,low in zip(row_maps['native_patch'][query_id]['all_evidence'],row_maps['lowres_patch'][query_id]['all_evidence']):
            for key in ('outline','angle','patch_count'):
                require(native.get(key)==low.get(key),'Low-resolution control changed native geometry/patch selection')
            control_checks+=1
    bootstrap=verify_bootstrap(rows,cutoffs,results['bootstrap'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    def blocked(*args,**kwargs):raise RuntimeError('Training is forbidden during fresh inference audit')
    torch.Tensor.backward=blocked;torch.optim.AdamW.step=blocked
    engine=DetailPipeline(checkpoint);before=state_digest(engine.model)
    require(before==results['model_state_before'],'Loaded CNN differs from evaluated state')
    replayed=[];legacy_cases=0
    for query_id in replay['query_ids']:
        episode,target=annotations[query_id]
        with Image.open(DATA/episode['support']['rgb_path']) as image:reference=np.array(image.convert('RGB'))
        with Image.open(DATA/target['rgb_path']) as image:query=np.array(image.convert('RGB'))
        predictions=engine.detect(reference,episode['support']['bbox_xyxy'],query)
        require(set(predictions)==set(METHODS),'Replay method roster differs')
        for alias,prediction in predictions.items():
            expected=row_maps[alias][query_id]
            require(prediction=={key:expected[key] for key in prediction},'Full prediction replay differs: '+alias+'/'+query_id)
            replayed.append({'method':alias,'id':query_id,'exact':True,'score':prediction['score'],'bbox_xyxy':prediction['bbox_xyxy']})
        ref_tensor=torch.from_numpy(reference.copy()).permute(2,0,1)[None].float()/255
        query_tensor=torch.from_numpy(query.copy()).permute(2,0,1)[None].float()/255
        original=legacy_detect(engine.model,ref_tensor,query_tensor,episode['support']['bbox_xyxy'])
        baseline=row_maps['old_cnn'][query_id]
        for key,other in [('score','score'),('bbox_xyxy','bbox_xyxy'),('proposals','proposals'),('scores','proposal_scores')]:
            require(original[key]==baseline[other],'Original v11 CNN direct replay differs')
        legacy_cases+=1
    require(len(replayed)==96 and legacy_cases==16,'Replay case budget mismatch')
    require(state_digest(engine.model)==before and not any(p.grad is not None for p in engine.model.parameters()),'Audit mutated CNN/gradients')
    for path,expected in protected.items():require(digest(path)==expected,'Audit changed protected evidence: '+path)
    output={'valid':True,'created_utc':datetime.now(timezone.utc).isoformat(),'auditor_sha256':digest(__file__),
            'results_sha256':digest(result_path),'manifest_sha256':digest(manifest_path),'protocol_sha256':digest(protocol_path),
            'replay_plan_sha256':digest(replay_path),'metric_values_checked':metric_checks,'localization_rows_checked':localization_checks,
            'unique_fresh_rgb_files_verified':len(image_paths),'exact_full_prediction_replays':len(replayed),
            'exact_original_cnn_compatibility_replays':legacy_cases,'shared_proposals_all_methods':True,
            'matched_native_geometry_lowres_evidence_checks':control_checks,'bootstrap':bootstrap,
            'one_reference_per_enrollment':True,'fresh_training_images':0,'external_pretrained_weights':False,
            'new_training_during_audit':False,'weights_and_gradients_unchanged':True,'protected_files':protected,
            'methods':method_metrics,'replays':replayed,
            'scope':'Fresh synthetic whole grammar combinations; shared building blocks. Own synthetic-trained CNN proposals plus analytic scoring, not newly trained CNN weights.'}
    with OUTPUT.open('x',encoding='utf-8') as stream:json.dump(output,stream,indent=2)
    print(json.dumps({key:value for key,value in output.items() if key not in ('protected_files','methods','replays')},indent=2),flush=True)


if __name__=='__main__':main()
