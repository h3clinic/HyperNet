"""Export v13 frozen code and audited pixel-free real-photo results."""
from pathlib import Path
import argparse,hashlib,json,re,shutil
PUBLIC=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def clean(value,root):
    if isinstance(value,dict):return {clean(k,root):clean(v,root) for k,v in value.items()}
    if isinstance(value,list):return [clean(v,root) for v in value]
    if isinstance(value,str):
        if re.search(r'data:image/[^;]+;base64,[A-Za-z0-9+/]{32}',value):raise ValueError('Image pixels cannot be published')
        normalized=value.replace('\\','/')
        for base,prefix in [(root,''),(root.parent/'hypernet_v12','previous_v12/'),(root.parent/'hypernet_v11','previous_v11/'),(PUBLIC,'public_source/'),(root.parent,'previous_experiments/'),(root.parent.parent,'workspace/')]:
            base=str(base).replace('\\','/').rstrip('/')
            if normalized.lower().startswith(base.lower()+'/'):return prefix+normalized[len(base)+1:]
        if re.match(r'^[A-Za-z]:/',normalized) or normalized.startswith('//'):raise ValueError('Unknown private provenance path')
    return value
def export_json(root,name,dest,outname,bindings):
    source=root/name;out=dest/outname;write(out,clean(read(source),root))
    bindings[outname]={'original_logical_path':name,'original_sha256':sha(source),'public_sha256':sha(out),'transformation':'JSON serialization and absolute provenance path remapping only; no embedded image pixels'}
def verify_frozen(root):
    protocol=root/'experiments/real_photo_protocol_freeze.json'
    for name,digest in read(protocol)['behavior_files'].items():
        source=Path(name) if Path(name).is_absolute() else root/name
        if sha(source)!=digest:raise ValueError('Frozen behavior changed: '+name)
    return protocol
def prepare(root):
    protocol=verify_frozen(root);prior=PUBLIC/'publication/source_snapshot_v0_12.json'
    if not prior.exists():
        shutil.copyfile(PUBLIC/'publication/source_snapshot.json',prior)
        shutil.copyfile(PUBLIC/'publication/validation.json',PUBLIC/'publication/validation_v0_12.json')
    previous=read(prior)
    for name,digest in previous['core_files'].items():
        if sha(PUBLIC/name)!=digest:raise ValueError('Previous public core changed')
    names=[p.relative_to(root).as_posix() for p in sorted((root/'hypernet').rglob('*.py'))]
    names += ['scripts/evaluate_real_photo.py','scripts/prepare_real_photos.py','scripts/recover_real_photo_transport.py',
        'scripts/freeze_real_photo_protocol.py','scripts/audit_real_photo.py','scripts/serve_photo_lab.py','scripts/report_real_photo.py','photo_lab.html']
    names += [p.relative_to(root).as_posix() for p in sorted((root/'tests').glob('test_*.py'))]
    exported={}
    for name in names:
        out=PUBLIC/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,out)
        if sha(out)!=sha(root/name):raise ValueError('Source copy changed bytes')
        exported[name]=sha(out)
    core={p.relative_to(PUBLIC).as_posix():sha(p) for p in sorted((PUBLIC/'hypernet').rglob('*.py'))}
    write(PUBLIC/'publication/source_snapshot.json',{'snapshot':'HyperNet v0.13.0 natural-photo transfer','core_file_count':len(core),
        'core_files':core,'unchanged_prior_core_files':len(previous['core_files']),'prior_snapshot':'publication/source_snapshot_v0_12.json',
        'original_protocol_sha256':sha(protocol),'additional_exported_files':exported,
        'notes':['All package files match the frozen v13 source byte for byte.','The inherited internal package version string is preserved; distribution metadata identifies0.13.0.','Natural photos and manual uploads are excluded from public artifacts.']})
    p=PUBLIC/'pyproject.toml';p.write_text(p.read_text().replace('version = "0.12.0"','version = "0.13.0"'))
    dest=PUBLIC/'benchmarks/v0.13';bindings={}
    for name,outname in [('data/coco_real_v13/manifest.json','cohort_manifest.json'),('data/coco_real_v13/plan.json','dataset_plan.json'),
        ('data/coco_real_v13/source_licenses.json','source_licenses.json'),('experiments/real_photo_protocol_freeze.json','evidence/real_photo_protocol_freeze.json')]:
        export_json(root,name,dest,outname,bindings)
    write(dest/'public_export_provenance.json',{'files':bindings,'original_protocol_sha256':sha(protocol),'warning':'Original audit hashes refer to original local JSON bytes. Sanitized public files have their own hashes; no scoring code changes.'})
    print(json.dumps({'prepared_package_files':len(core),'unchanged_prior_files':len(previous['core_files']),'natural_photos_exported':0}))
def results(root):
    protocol=verify_frozen(root);path=root/'results/real_eval/real_results.json';result=read(path)
    audit_paths=[p for p in (root/'experiments').glob('*.json') if 'audit' in p.name and read(p).get('valid') is True and read(p).get('results_sha256')==sha(path)]
    if not result['completed'] or not audit_paths:raise ValueError('Completed independently audited results required')
    dest=PUBLIC/'benchmarks/v0.13';bindings=read(dest/'public_export_provenance.json')['files']
    export_json(root,'results/real_eval/real_results.json',dest,'aggregate_metrics.json',bindings)
    for p in sorted((root/'experiments').glob('*.json')):export_json(root,p.relative_to(root).as_posix(),dest,'evidence/'+p.name,bindings)
    for alias,info in result['score_files'].items():
        source=path.parent/info['path'];out=dest/info['path']
        if sha(source)!=info['sha256']:raise ValueError('Prediction hash changed')
        for line in source.read_text(encoding='utf-8').splitlines():
            value=json.loads(line)
            if clean(value,root)!=value:raise ValueError('Expected portable score rows')
        shutil.copyfile(source,out);bindings[out.name]={'original_logical_path':source.relative_to(root).as_posix(),'original_sha256':sha(source),'public_sha256':sha(out),'transformation':'none: exact bytes'}
    # Only these two scientific plots contain aggregates, never natural photos.
    for name in ['real_photo_operating_points.png','real_photo_score_distributions.png']:
        out=dest/'figures'/name;out.parent.mkdir(exist_ok=True);shutil.copyfile(root/'figures'/name,out)
    report=(root/'index.html').read_text(encoding='utf-8')
    report=re.sub(r'<figure><img src="assets/.*?</figure>','',report,flags=re.S)
    report=report.replace('href="photo_lab.html"','href="../../photo_lab.html"').replace('href="results/real_eval/real_results.json"','href="aggregate_metrics.json"').replace('href="data/coco_real_v13/manifest.json"','href="cohort_manifest.json"').replace('href="experiments/','href="evidence/')
    if re.search(r'data:image/[^;]+;base64,',report) or 'src="assets/' in report:raise ValueError('Natural image content in public report')
    (dest/'report.html').write_text(report,encoding='utf-8')
    write(dest/'public_export_provenance.json',{'files':bindings,'original_protocol_sha256':sha(protocol),
        'natural_photo_pixels_exported':0,'user_upload_pixels_exported':0,
        'warning':'Original hashes refer to original local JSON bytes; public JSON has separate hashes after path remapping. Images must be acquired from original licensed sources; no model/scoring code changes.'})
    print(json.dumps({'audited_methods':len(result['score_files']),'score_rows':1152,'natural_photos_exported':0,'metadata_files':len(bindings)}))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--stage',choices=['prepare','results'],required=True);args=parser.parse_args()
    (prepare if args.stage=='prepare' else results)(args.root.resolve())
if __name__=='__main__':main()
