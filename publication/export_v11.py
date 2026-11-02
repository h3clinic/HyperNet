"""Publish exact v11 source, then audited pixel-free results in a separate stage."""
from pathlib import Path
import argparse, hashlib, json, re, shutil
PUBLIC=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sanitize(v,root):
    if isinstance(v,dict):return {sanitize(k,root):sanitize(x,root) for k,x in v.items()}
    if isinstance(v,list):return [sanitize(x,root) for x in v]
    if isinstance(v,str) and (re.match(r'^[A-Za-z]:[/\\]',v) or v.startswith('\\\\')):return Path(v).resolve().relative_to(root).as_posix()
    return v

def prepare(root):
    snapshot=read(PUBLIC/'publication/source_snapshot.json')
    previous=read(PUBLIC/'publication/source_snapshot_v0_10.json') if (PUBLIC/'publication/source_snapshot_v0_10.json').exists() else snapshot
    if previous['core_file_count']!=69:raise ValueError('Expected v10 snapshot')
    for p,h in previous['core_files'].items():
        if sha(PUBLIC/p)!=h:raise ValueError('Previously published code changed: '+p)
    if not (PUBLIC/'publication/source_snapshot_v0_10.json').exists():
        shutil.copyfile(PUBLIC/'publication/source_snapshot.json',PUBLIC/'publication/source_snapshot_v0_10.json')
        shutil.copyfile(PUBLIC/'publication/validation.json',PUBLIC/'publication/validation_v0_10.json')
    freeze=root/'experiments/detector_training_freeze.json'
    for p,h in read(freeze)['sources'].items():
        if sha(root/p)!=h:raise ValueError('Frozen source changed: '+p)
    for p,h in read(root/'experiments/shapes2d_data_freeze.json')['behavior_files'].items():
        if sha(p)!=h:raise ValueError('Frozen data source changed: '+p)
    names=['hypernet/mathematical_shapes2d.py','hypernet/one_shot_detector.py',
      'scripts/prepare_2d_data.py','scripts/train_2d_detector.py','scripts/audit_2d_detector.py',
      'scripts/serve_2d_gallery.py','scripts/report_2d_detector.py','two_dimensional_gallery.html',
      'research/two_dimensional_primary_sources.json','research/two_dimensional_research.html',
      'data/SHAPES2D_CONTRACT.json','experiments/audit_2d_grammar.py','experiments/analytic_detail_bypass.py']
    names += [p.relative_to(root).as_posix() for p in sorted((root/'tests').glob('test_*.py'))]
    exported={}
    for name in names:
        out=PUBLIC/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,out)
        exported[name]=sha(out)
        if exported[name]!=sha(root/name):raise ValueError('Copy changed bytes')
    core={p.relative_to(PUBLIC).as_posix():sha(p) for p in sorted((PUBLIC/'hypernet').rglob('*.py'))}
    write(PUBLIC/'publication/source_snapshot.json',{'snapshot':'HyperNet v0.11.0 procedural 2D exemplar detection',
      'created_date':'2026-11-02','core_file_count':len(core),'core_files':core,
      'prior_snapshot':'publication/source_snapshot_v0_10.json','unchanged_prior_core_files':69,
      'experimental_freeze_sha256':sha(freeze),'new_frozen_behavior_files':read(freeze)['sources'],
      'additional_exported_files':exported,'evaluation_status':'Final audit and result export required before release.',
      'notes':['The two new package files and training/data/audit drivers are byte-identical to the experiment.',
               'No generated image pixels or trained weights are exported.']})
    p=PUBLIC/'pyproject.toml';s=p.read_text().replace('version = "0.10.0"','version = "0.11.0"')
    s=s.replace('dev = ["pytest>=8", "scikit-learn>=1.5"]','dev = ["pytest>=8", "scikit-learn>=1.5", "matplotlib>=3.8"]');p.write_text(s)
    print(json.dumps({'prepared_core':len(core),'new_exports':len(names)}))

def export_results(root):
    audit_path=root/'experiments/detector_run_audit.json';audit=read(audit_path)
    result_path=root/'results/detector_results.json';results=read(result_path)
    freeze=root/'experiments/detector_training_freeze.json'
    if audit.get('valid') is not True or audit['results_sha256']!=sha(result_path) or audit['freeze_sha256']!=sha(freeze):raise ValueError('Valid bound independent audit required')
    if not results['completed'] or not results['sources_unchanged']:raise ValueError('Incomplete run')
    for p,h in audit['protected_files'].items():
        if sha(p)!=h:raise ValueError('Audited artifact changed: '+p)
    dest=PUBLIC/'benchmarks/v0.11'
    if dest.exists():raise FileExistsError('Preserve published results')
    dest.mkdir(parents=True)
    inputs={'results/detector_results.json':'aggregate_metrics.json',
      'experiments/detector_training_freeze.json':'training_protocol.json',
      'experiments/detector_batch_plan.json':'batch_plan.json',
      'experiments/detector_replay_plan.json':'replay_plan.json',
      'experiments/detector_run_audit.json':'inference_metric_audit.json',
      'experiments/detector_analysis.json':'paired_analysis.json',
      'experiments/shapes2d_data_audit.json':'data_audit.json',
      'experiments/shapes2d_data_freeze.json':'data_protocol.json',
      'experiments/canonical_2d_grammar_audit.json':'canonical_grammar_audit.json',
      'data/shapes2d_v11/plan.json':'dataset_plan.json'}
    for name in ['shapes2d_visibility_audit','shapes2d_support_fragments','2d_gallery_full_audit','detector_analysis_independent_review','analytic_detail_bypass_plan','analytic_detail_bypass_replay_audit','analytic_detail_bypass_replay_plan','analytic_detail_bypass_independent_review','shapes2d_evidence_bundle','type_present_localization_diagnostic','type_present_localization_independent_review']:
        if (root/f'experiments/{name}.json').exists():inputs[f'experiments/{name}.json']=name+'.json'
    originals={}
    for source,target in inputs.items():write(dest/target,sanitize(read(root/source),root));originals[source]=sha(root/source)
    manifest=read(root/'data/shapes2d_v11/manifest.json')
    episodes=[]
    for e in manifest['episodes']:
        if e['split']=='train':continue
        entry={k:e[k] for k in ['episode_id','split','family_id','enrolled_variant','grammar_digits']}
        entry['support']={k:v for k,v in e['support'].items() if k in ('rgb_path','bbox_xyxy','rgb_sha256','rgb_array_sha256')}
        entry['queries']=[{k:q[k] for k in ['id','present','candidate_variant','rgb_path','target_bbox_xyxy','candidate_bbox_xyxy','rgb_sha256','rgb_array_sha256']} for q in e['queries']]
        episodes.append(entry)
    write(dest/'heldout_manifest.json',{'original_manifest_sha256':sha(root/'data/shapes2d_v11/manifest.json'),'episodes':episodes,
       'scope':'DEV and TEST reference/query labels, pixel hashes and boxes. Full generation seeds and partitions are in dataset_plan.json.'})
    originals['data/shapes2d_v11/manifest.json']=sha(root/'data/shapes2d_v11/manifest.json')
    export_arms=['coarse_match','micro_aware']
    if (root/'results/analytic_detail_bypass_results.json').is_file():
        hybrid=read(root/'results/analytic_detail_bypass_results.json')
        if not hybrid['completed'] or not hybrid['weights_and_frozen_sources_unchanged']:raise ValueError('Incomplete diagnostic')
        export_arms.append('analytic_detail_bypass')
        write(dest/'analytic_detail_bypass_results.json',sanitize(hybrid,root))
        originals['results/analytic_detail_bypass_results.json']=sha(root/'results/analytic_detail_bypass_results.json')
    for arm in export_arms:
        source=root/'results'/f'{arm}_predictions.jsonl'
        expected=(hybrid if arm=='analytic_detail_bypass' else results['arms'][arm])['predictions_sha256']
        if sha(source)!=expected:raise ValueError('Prediction hash mismatch')
        fields=['id','episode_id','split','family_id','enrolled_variant','present','score','bbox_xyxy','proposals','proposal_scores','iou','proposal_best_iou']
        rows=[{k:r[k] for k in fields} for r in (json.loads(l) for l in source.read_text().splitlines() if l)]
        (dest/source.name).write_text('\n'.join(json.dumps(r,separators=(',',':'),allow_nan=False) for r in rows)+'\n',encoding='utf-8')
        originals['results/'+source.name]=sha(source)
    write(dest/'export_provenance.json',{'original_artifact_sha256':originals,
      'public_artifact_sha256':{p.name:sha(p) for p in dest.iterdir()},
      'sanitization':'Private absolute paths replaced by logical experiment-relative identifiers; prediction paths omitted. Original hashes retained.',
      'pixels_exported':False,'learned_parameters_exported':False})
    p=PUBLIC/'publication/source_snapshot.json';snapshot=read(p)
    snapshot['evaluation_status']='Audited 2D procedural detection; see benchmarks/v0.11 for both arms and paired uncertainty. No natural-photo transfer claim.';write(p,snapshot)
    print(json.dumps({'benchmark':str(dest.relative_to(PUBLIC)),'prediction_rows':1024*len(export_arms)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--stage',choices=['prepare','results'],required=True);a=p.parse_args()
    (prepare if a.stage=='prepare' else export_results)(a.root.resolve())
