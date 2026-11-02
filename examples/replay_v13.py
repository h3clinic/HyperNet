"""Replay the published real-photo transfer result, including its failures.

Default: first sorted positive and negative TEST query, four methods each.
--full: all288 queries. Both operating modes reuse published thresholds;
no threshold fitting, weight training, or claim of a new heldout experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from numbers import Real
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BENCHMARK=ROOT/'benchmarks/v0.13'
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def select_queries(queries,full=False):
    if full:return list(queries)
    selected=[]
    for present in (True,False):
        options=sorted((q for q in queries if q['split']=='test' and q['present']==present),key=lambda q:q['id'])
        if not options:raise ValueError('Default replay requires positive and negative TEST queries')
        selected.append(options[0])
    return selected


def validate_manifest(manifest,published):
    if manifest.get('schema_version')!='hypernet-real-photo-v13-v1':raise ValueError('Unexpected dataset schema')
    for key in ('references','queries','categories','counts'):
        if manifest.get(key)!=published.get(key):raise ValueError('Dataset differs from published cohort: '+key)


def compare(actual,expected,atol,stats,path='value'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Different fields: '+path)
        for k,v in expected.items():compare(actual[k],v,atol,stats,path+'.'+k)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('Different list: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,atol,stats,path+f'[{i}]')
    elif isinstance(expected,bool) or isinstance(actual,bool):
        if type(actual)!=type(expected) or actual!=expected:raise ValueError('Boolean mismatch: '+path)
    elif isinstance(actual,Real) and isinstance(expected,Real):
        if not math.isfinite(actual) or not math.isfinite(expected):raise ValueError('Nonfinite value: '+path)
        error=abs(float(actual)-float(expected))
        stats['numeric_values_checked']+=1
        stats['max_absolute_error']=max(stats['max_absolute_error'],error)
        if error>atol or (isinstance(expected,int) and actual!=expected):raise ValueError('Number differs: '+path)
    elif type(actual)!=type(expected) or actual!=expected:raise ValueError('Value differs: '+path)


def compare_row(row,saved,atol,stats):
    # Float tolerance never permits changing the selected geometry or labels.
    for key in ('id','category_id','image_id','present','bbox_xyxy','proposals'):
        if row[key]!=saved[key]:raise ValueError('Discrete/box prediction changed: '+key)
    compare(row,saved,atol,stats)


def load_runner():
    spec=importlib.util.spec_from_file_location('published_v13_driver',ROOT/'scripts/evaluate_real_photo.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_sources():
    provenance=read(BENCHMARK/'public_export_provenance.json')
    for name in ('cohort_manifest.json','aggregate_metrics.json','evidence/real_photo_protocol_freeze.json'):
        if sha(BENCHMARK/name)!=provenance['files'][name]['public_sha256']:raise ValueError('Published evidence changed: '+name)
    protocol=read(BENCHMARK/'evidence/real_photo_protocol_freeze.json')
    bindings={name:value for name,value in protocol['behavior_files'].items()
              if name.startswith('hypernet/') or name=='scripts/evaluate_real_photo.py'}
    if 'hypernet/real_photo.py' not in bindings or 'scripts/evaluate_real_photo.py' not in bindings:
        raise ValueError('Missing frozen inference bindings')
    for name,value in bindings.items():
        if sha(ROOT/name)!=value:raise ValueError('Inference source differs: '+name)
    return bindings


def evaluate_selected(runner,engine,manifest,selected,data):
    refs={r['category_id']:r for r in manifest['references']}
    categories=sorted({q['category_id'] for q in selected})
    enrolled={c:engine.enroll(runner.load_rgb(data,refs[c]),refs[c]['bbox_xyxy']) for c in categories}
    rows={a:[] for a in runner.ALIASES}
    for query in selected:
        rgb=runner.load_rgb(data,query);category=query['category_id']
        predictions=engine.predict(enrolled[category],rgb)
        # Ground-truth boxes are accessed only after inference, for scoring.
        for alias,prediction in predictions.items():
            runner.validate_prediction(prediction,query['width'],query['height'])
            targets=query['gt_boxes_xyxy']
            rows[alias].append(dict(id=query['id'],category_id=category,name=query['name'],split=query['split'],
                present=query['present'],method=alias,image_id=query['image_id'],reference_image_id=refs[category]['image_id'],
                reference_path=refs[category]['rgb_path'],query_path=query['rgb_path'],
                original_query_size=[query['width'],query['height']],**prediction,
                iou=runner.max_box_iou(prediction['bbox_xyxy'],targets),
                proposal_best_iou=max((runner.max_box_iou(b,targets) for b in prediction['proposals']),default=0.)))
    return rows


def run(data,checkpoint,output,full=False,atol=1e-6):
    if not math.isfinite(atol) or not 0<=atol<=1e-6:raise ValueError('--atol must be within0..1e-6, with zero relative tolerance')
    data,checkpoint,output=map(lambda p:Path(p).resolve(),(data,checkpoint,output))
    if output.exists() and any(output.iterdir()):raise FileExistsError('Preserve existing replication evidence: use an empty output directory')
    bindings=verify_sources()
    manifest=read(data/'manifest.json');published=read(BENCHMARK/'cohort_manifest.json')
    validate_manifest(manifest,published)
    aggregate=read(BENCHMARK/'aggregate_metrics.json')
    if aggregate.get('completed') is not True:raise ValueError('Published result incomplete')
    if sha(checkpoint)!=aggregate['checkpoint_sha256']:raise ValueError('Requires the exact project-trained CNN checkpoint')
    selected=select_queries(manifest['queries'],full)
    expected={}
    for alias,info in aggregate['score_files'].items():
        path=BENCHMARK/info['path']
        if sha(path)!=info['sha256']:raise ValueError('Published prediction log changed: '+alias)
        expected[alias]={r['id']:r for r in (json.loads(line) for line in path.read_text().splitlines())}
    runner=load_runner()
    import torch,numpy,cv2
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    engine=runner.RealPhotoDetector(checkpoint)
    before=runner.state_hash(engine.model)
    if before!=aggregate['model_state_before']:raise ValueError('Loaded weights differ from published tensors')
    start=time.perf_counter()
    rows=evaluate_selected(runner,engine,manifest,selected,data)
    stats={'numeric_values_checked':0,'max_absolute_error':0.}
    for alias,records in rows.items():
        for row in records:
            saved=expected[alias][row['id']]
            compare_row(row,saved,atol,stats)
            for mode in aggregate['modes'].values():
                cutoff=mode['thresholds'][alias]
                if (row['score']>=cutoff)!=(saved['score']>=cutoff):raise ValueError('Published threshold decision changed')
    after=runner.state_hash(engine.model)
    if after!=before or sha(checkpoint)!=aggregate['checkpoint_sha256'] or any(p.grad is not None for p in engine.model.parameters()):
        raise RuntimeError('Replication changed weights or produced gradients')
    if any(sha(ROOT/name)!=value for name,value in bindings.items()):raise RuntimeError('Source changed during replay')
    full_metrics=None
    if full:
        full_metrics={}
        for mode,recorded in aggregate['modes'].items():
            cuts=recorded['thresholds']
            methods={alias:{split:runner.summarize([r for r in records if r['split']==split],cuts[alias])
                            for split in ('dev','test')} for alias,records in rows.items()}
            bootstrap=runner.category_bootstrap(rows,cuts,samples=10000,seed=1313)
            compare(methods,recorded['methods'],atol,stats,mode+'/metrics')
            compare(bootstrap,recorded['bootstrap'],atol,stats,mode+'/bootstrap')
            full_metrics[mode]={'methods':methods,'bootstrap':bootstrap,'thresholds':cuts}
    output.mkdir(parents=True,exist_ok=True)
    score_files={}
    for alias,records in rows.items():
        path=output/(alias+'_replication_scores.jsonl')
        with path.open('x',encoding='utf-8') as stream:
            for row in records:stream.write(json.dumps(row,allow_nan=False)+'\n')
        score_files[alias]={'path':path.name,'sha256':sha(path)}
    summary={'format':'hypernet.portable_replication.v13','replication':True,'verified':True,
        'original_experiment_freeze_recreated':False,'full_cohort':bool(full),'selected_query_ids':[q['id'] for q in selected],
        'query_images':len(selected),'reference_enrollments':len({q['category_id'] for q in selected}),
        'method_prediction_rows':sum(len(records) for records in rows.values()),
        'threshold_source':'Published transferred synthetic and secondary realDEV thresholds, both fixed; never refitted',
        'acceptance_decisions_match_both_modes':True,'selected_boxes_and_proposals_match_exactly':True,
        'float_absolute_tolerance':atol,'float_relative_tolerance':0.,'comparison':stats,
        'checkpoint_sha256':sha(checkpoint),'model_state_before':before,'model_state_after':after,
        'manifest_sha256':sha(data/'manifest.json'),'published_aggregate_sha256':sha(BENCHMARK/'aggregate_metrics.json'),
        'source_bindings':bindings,'learned_weight_updates':0,'new_threshold_fitting':False,
        'reference_images_per_enrollment':1,'score_files':score_files,'full_metrics':full_metrics,
        'elapsed_seconds':time.perf_counter()-start,
        'runtime':{'python':platform.python_version(),'numpy':numpy.__version__,'torch':torch.__version__,'opencv':cv2.__version__,'torch_threads':4},
        'scope':'Replication of published natural-photo transfer failure; not a new heldout evaluation or a claim of successful natural category recognition.'}
    with (output/'replication_summary.json').open('x',encoding='utf-8') as stream:json.dump(summary,stream,indent=2,allow_nan=False)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/coco_real_v13')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/detector_micro_aware.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--atol',type=float,default=1e-6)
    args=parser.parse_args()
    result=run(args.data,args.checkpoint,args.output,args.full,args.atol)
    print(json.dumps({k:result[k] for k in ('replication','verified','full_cohort','query_images','method_prediction_rows','comparison','elapsed_seconds')},indent=2))
