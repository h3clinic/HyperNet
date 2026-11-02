"""Reproduce published v0.12 predictions without inventing original freeze records.

Default: four deterministic enrollments (two DEV, two TEST). --full replays all
512 enrollments. Float comparisons use absolute tolerance <=1e-6, zero relative
tolerance; IDs, boxes, labels and acceptance decisions must match exactly.
Published DEV thresholds are reused: this command never refits thresholds.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from numbers import Integral,Real
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BENCHMARK=ROOT/'benchmarks/v0.12'
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def select_episodes(episodes,limit=4,full=False):
    if full:return list(episodes)
    if isinstance(limit,bool) or not isinstance(limit,int) or limit<1 or limit>len(episodes):
        raise ValueError('--limit-episodes must be between1 and the published enrollment count')
    dev=sorted((e for e in episodes if e['split']=='dev'),key=lambda e:(e['family_id'],e['enrolled_variant'],e['episode_id']))
    test=sorted((e for e in episodes if e['split']=='test'),key=lambda e:(e['family_id'],e['enrolled_variant'],e['episode_id']))
    if len(dev)+len(test)!=len(episodes):raise ValueError('Replication accepts only published DEV/TEST enrollments')
    # Interleave the two partitions. Four default episodes include both
    # enrollment variants of the first family in each partition, without scores.
    ordered=[]
    for index in range(max(len(dev),len(test))):
        for group in (dev,test):
            if index<len(group):ordered.append(group[index])
    return ordered[:limit]


def compare_json(actual,expected,atol,stats,path='value'):
    if isinstance(actual,dict) and isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('Different fields at '+path)
        for key in actual:compare_json(actual[key],expected[key],atol,stats,path+'.'+str(key))
        return
    if isinstance(actual,(list,tuple)) and isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):raise ValueError('Different list length at '+path)
        for index,(a,b) in enumerate(zip(actual,expected)):compare_json(a,b,atol,stats,f'{path}[{index}]')
        return
    if isinstance(actual,bool) or isinstance(expected,bool):
        if type(actual) is not type(expected) or actual!=expected:raise ValueError('Boolean mismatch at '+path)
    elif isinstance(actual,Real) and isinstance(expected,Real):
        if isinstance(actual,Integral) or isinstance(expected,Integral):
            if actual!=expected:raise ValueError('Integer/coordinate mismatch at '+path)
        else:
            a,b=float(actual),float(expected)
            if not math.isfinite(a) or not math.isfinite(b):raise ValueError('Nonfinite value at '+path)
            difference=abs(a-b)
            stats['float_values_checked']+=1
            stats['max_absolute_float_error']=max(stats['max_absolute_float_error'],difference)
            stats['nonzero_float_differences']+=int(difference!=0.)
            if difference>atol:raise ValueError(f'Float mismatch at {path}: error {difference} exceeds {atol}')
    elif actual!=expected or type(actual) is not type(expected):
        raise ValueError('Value mismatch at '+path)
    stats['leaf_values_checked']+=1


def validate_manifest(manifest,published):
    if manifest.get('completed') is not True:raise ValueError('Dataset manifest is incomplete')
    # Absolute provenance paths differ between original and portable exports;
    # the full enrollment roster, labels, image paths and hashes must not differ.
    for key in ('episodes','counts','family_partitions'):
        if manifest.get(key)!=published.get(key):raise ValueError('Dataset differs from published cohort: '+key)


def load_runner():
    spec=importlib.util.spec_from_file_location('published_v12_evaluation_driver',ROOT/'scripts/evaluate_detail.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_public_sources():
    provenance=read(BENCHMARK/'public_export_provenance.json')
    for relative in ('aggregate_metrics.json','cohort_manifest.json','evidence/detail_protocol_freeze.json'):
        if sha(BENCHMARK/relative)!=provenance['files'][relative]['public_sha256']:
            raise ValueError('Published evidence hash changed: '+relative)
    protocol=read(BENCHMARK/'evidence/detail_protocol_freeze.json')
    required=('hypernet/detail_pipeline.py','hypernet/gradient_detail.py','hypernet/native_detail.py',
              'hypernet/one_shot_detector.py','hypernet/training_v3.py','hypernet/detail_metrics.py','scripts/evaluate_detail.py')
    bindings={}
    for relative in required:
        expected=protocol['behavior_files'][relative]
        if sha(ROOT/relative)!=expected:raise ValueError('Published inference source changed: '+relative)
        bindings[relative]=expected
    return bindings


def run(data,checkpoint,output,limit=4,full=False,atol=1e-6):
    if not math.isfinite(atol) or not 0<=atol<=1e-6:
        raise ValueError('--atol must lie between0 and1e-6; use0 for bit-exact scalar replay')
    data,checkpoint,output=Path(data).resolve(),Path(checkpoint).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use an empty output directory; existing replication evidence is preserved')
    source_bindings=verify_public_sources()
    manifest=read(data/'manifest.json');published=read(BENCHMARK/'cohort_manifest.json')
    validate_manifest(manifest,published)
    aggregate=read(BENCHMARK/'aggregate_metrics.json')
    if not aggregate['completed']:raise ValueError('Published comparison is incomplete')
    if sha(checkpoint)!=aggregate['checkpoint_sha256']:raise ValueError('Checkpoint differs from the project’s own published CNN')
    selected=select_episodes(manifest['episodes'],limit,full)
    expected={}
    for alias,info in aggregate['score_files'].items():
        path=BENCHMARK/info['path']
        if sha(path)!=info['sha256']:raise ValueError('Published prediction log hash changed: '+alias)
        expected[alias]={row['id']:row for row in (json.loads(line) for line in path.read_text().splitlines() if line)}
    runner=load_runner()
    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    engine=runner.DetailPipeline(checkpoint)
    before=runner.state_hash(engine.model)
    if before!=aggregate['model_state_before']:raise ValueError('Loaded tensor state differs from publication')
    original_checkpoint_sha=sha(checkpoint)
    start=time.perf_counter()
    # The unchanged driver validates raw PNG and decoded RGB hashes while
    # loading each image, then calls the unchanged inference API before labels.
    rows=runner.evaluate_episodes(engine,selected,data)
    stats={'float_values_checked':0,'leaf_values_checked':0,'max_absolute_float_error':0.,'nonzero_float_differences':0}
    count=0;cutoffs={alias:aggregate['methods'][alias]['dev']['threshold'] for alias in rows}
    for alias,records in rows.items():
        for row in records:
            saved=expected[alias][row['id']]
            compare_json(row,saved,atol,stats,alias+'/'+row['id'])
            if (row['score']>=cutoffs[alias])!=(saved['score']>=cutoffs[alias]):
                raise ValueError('Published acceptance decision changed: '+alias+'/'+row['id'])
            count+=1
    after=runner.state_hash(engine.model)
    if after!=before or sha(checkpoint)!=original_checkpoint_sha or any(p.grad is not None for p in engine.model.parameters()):
        raise RuntimeError('Replication changed weights or created gradients')
    for relative,expected_sha in source_bindings.items():
        if sha(ROOT/relative)!=expected_sha:raise RuntimeError('Inference source changed during replay')
    metrics=None;bootstrap=None
    if full:
        metrics={alias:{split:runner.summarize([r for r in records if r['split']==split],cutoffs[alias])
                        for split in ('dev','test')} for alias,records in rows.items()}
        compare_json(metrics,aggregate['methods'],atol,stats,'aggregate_metrics')
        bootstrap=runner.bootstrap(rows,cutoffs)
        compare_json(bootstrap,aggregate['bootstrap'],atol,stats,'paired_bootstrap')
    output.mkdir(parents=True,exist_ok=True)
    score_files={}
    for alias,records in rows.items():
        path=output/f'{alias}_replication_scores.jsonl'
        with path.open('x',encoding='utf-8') as stream:
            for row in records:stream.write(json.dumps(row,allow_nan=False)+'\n')
        score_files[alias]={'path':path.name,'sha256':sha(path)}
    summary={'format':'hypernet.portable_replication.v12','replication':True,'verified':True,
             'original_experiment_freeze_recreated':False,'full_cohort':bool(full),
             'selected_episode_ids':[e['episode_id'] for e in selected],'enrollments':len(selected),
             'reference_query_rows':sum(len(e['queries']) for e in selected),'method_prediction_rows':count,
             'float_absolute_tolerance':atol,'float_relative_tolerance':0.,'acceptance_decisions_match':True,
             'comparison':stats,'threshold_source':'Published fresh DEV thresholds; no refitting',
             'checkpoint_sha256':original_checkpoint_sha,'model_state_before':before,'model_state_after':after,
             'dataset_manifest_sha256':sha(data/'manifest.json'),'published_manifest_sha256':sha(BENCHMARK/'cohort_manifest.json'),
             'published_aggregate_sha256':sha(BENCHMARK/'aggregate_metrics.json'),'source_bindings':source_bindings,
             'learned_weight_updates':0,'new_threshold_fitting':False,'reference_images_per_enrollment':1,
             'elapsed_seconds':time.perf_counter()-start,
             'runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'opencv':cv2.__version__,'torch_threads':4},
             'score_files':score_files,'full_metrics':metrics,'full_bootstrap':bootstrap,
             'scope':'Replay of already published synthetic images, not a new heldout evaluation or new evidence of natural-image accuracy.'}
    with (output/'replication_summary.json').open('x',encoding='utf-8') as stream:json.dump(summary,stream,indent=2,allow_nan=False)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/shapes2d_fresh_v12')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/detector_micro_aware.pt')
    parser.add_argument('--output',type=Path,required=True)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--limit-episodes',type=int,default=None)
    selection.add_argument('--full',action='store_true')
    parser.add_argument('--atol',type=float,default=1e-6,help='Absolute float tolerance0..1e-6 (no relative tolerance); discrete decisions must match exactly')
    args=parser.parse_args()
    summary=run(args.data,args.checkpoint,args.output,4 if args.limit_episodes is None else args.limit_episodes,args.full,args.atol)
    print(json.dumps({key:summary[key] for key in ('replication','verified','full_cohort','enrollments','method_prediction_rows','comparison','elapsed_seconds')},indent=2))


if __name__=='__main__':main()
