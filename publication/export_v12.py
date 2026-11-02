"""Export audited v12 sources/results; rewrite only public provenance path strings."""
from pathlib import Path
import argparse,hashlib,json,re,shutil,zipfile
PUBLIC=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sanitize(v,root):
    if isinstance(v,dict):return {sanitize(k,root):sanitize(x,root) for k,x in v.items()}
    if isinstance(v,list):return [sanitize(x,root) for x in v]
    if isinstance(v,str):
        value=v.replace('\\','/')
        for base,prefix in [(root,''),(root.parent/'hypernet_v11','previous_v11/'),(root.parent,'previous_experiments/'),(PUBLIC,'public_source/')]:
            base=str(base).replace('\\','/').rstrip('/')
            if value.lower().startswith(base.lower()+'/'):return prefix+value[len(base)+1:]
        if re.match(r'^[A-Za-z]:/',value) or value.startswith('//'):raise ValueError('Unrecognized private path')
    return v
def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--assets',type=Path,required=True);args=ap.parse_args();root=args.root.resolve()
    protocol=read(root/'experiments/detail_protocol_freeze.json')
    for section in ['behavior_files','development_evidence']:
        for path,expected in protocol[section].items():
            source=Path(path) if Path(path).is_absolute() else root/path
            if sha(source)!=expected:raise ValueError('Frozen binding changed: '+path)
    result=read(root/'results/fresh_eval/fresh_results.json');audit=read(root/'experiments/fresh_detail_run_audit.json')
    if not audit['valid'] or audit['results_sha256']!=sha(root/'results/fresh_eval/fresh_results.json'):raise ValueError('Independent audit missing')
    previous=PUBLIC/'publication/source_snapshot_v0_11.json'
    if not previous.exists():
        shutil.copyfile(PUBLIC/'publication/source_snapshot.json',previous)
        shutil.copyfile(PUBLIC/'publication/validation.json',PUBLIC/'publication/validation_v0_11.json')
    prior=read(previous)
    for path,expected in prior['core_files'].items():
        if sha(PUBLIC/path)!=expected:
            if path!='hypernet/__init__.py' or sha(PUBLIC/path)!=sha(root/path):raise ValueError('Prior core changed: '+path)
    names=[p.relative_to(root).as_posix() for p in (root/'hypernet').rglob('*.py')]
    names+=['scripts/evaluate_detail.py','scripts/prepare_fresh_2d_v12.py','scripts/freeze_detail_protocol.py',
      'scripts/serve_fresh_gallery.py','scripts/report_fresh_detail.py','fresh_gallery.html',
      'research/detail_matching_primary_sources.json','research/detail_matching_research.html',
      'data/SHAPES2D_FRESH_CONTRACT.json','tests/test_detail_pipeline.py','tests/test_gradient_detail.py','tests/test_fresh_gallery.py']
    exported={}
    for name in names:
        out=PUBLIC/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,out)
        if sha(out)!=sha(root/name):raise ValueError('Copy modified source')
        exported[name]=sha(out)
    dest=PUBLIC/'benchmarks/v0.12';dest.mkdir(parents=True,exist_ok=True);bindings={}
    inputs={'results/fresh_eval/fresh_results.json':'aggregate_metrics.json',
      'data/shapes2d_fresh_v12/manifest.json':'cohort_manifest.json','data/shapes2d_fresh_v12/plan.json':'dataset_plan.json'}
    for p in (root/'experiments').glob('*.json'):inputs[p.relative_to(root).as_posix()]='evidence/'+p.name
    inputs['experiments/old_dev_stability/fresh_results.json']='evidence/old_dev_stability.json'
    for name,outname in inputs.items():
        source=root/name;out=dest/outname;write(out,sanitize(read(source),root))
        bindings[outname]={'original_logical_path':name,'original_sha256':sha(source),'public_sha256':sha(out),'transformation':'JSON serialization and absolute provenance paths mapped to logical identifiers only'}
    for alias,record in result['score_files'].items():
        source=root/'results/fresh_eval'/record['path'];out=dest/record['path']
        if sha(source)!=record['sha256']:raise ValueError('Score binding changed')
        # These full logs already have relative paths and retain every evidence field.
        text=source.read_text();sanitized=[sanitize(json.loads(line),root) for line in text.splitlines()]
        if any(json.loads(line)!=clean for line,clean in zip(text.splitlines(),sanitized)):raise ValueError('Unexpected nonportable score log')
        shutil.copyfile(source,out);bindings[out.name]={'original_logical_path':source.relative_to(root).as_posix(),'original_sha256':sha(source),'public_sha256':sha(out),'transformation':'none: exact bytes'}
    archive=dest/'original_drivers';archive.mkdir(exist_ok=True)
    for name in ['scripts/audit_fresh_detail_v12.py','tests/test_fresh_detail_audit.py','tests/test_fresh_2d_data.py']:
        shutil.copyfile(root/name,archive/Path(name).name)
    for folder in ['figures','assets']:
        for p in (root/folder).glob('*.png'):
            out=dest/folder/p.name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
    report=(root/'index.html').read_text(encoding='utf-8')
    report=report.replace('href="fresh_gallery.html"','href="README.md#reproduce-inference"').replace('href="research/','href="../../research/').replace('href="results/fresh_eval/fresh_results.json"','href="aggregate_metrics.json"').replace('href="data/shapes2d_fresh_v12/manifest.json"','href="cohort_manifest.json"').replace('href="experiments/','href="evidence/')
    (dest/'report.html').write_text(report,encoding='utf-8')
    write(dest/'public_export_provenance.json',{'original_protocol_sha256':sha(root/'experiments/detail_protocol_freeze.json'),'path_remapping':{'v12_workspace':'relative experiment paths','v11_workspace':'previous_v11/','other_experiment_workspace':'previous_experiments/','public_checkout':'public_source/'},'files':bindings,'warning':'Original hashes refer to original local JSON bytes; sanitized public JSON has separate public hashes. No model or scoring source changes.'})
    core={p.relative_to(PUBLIC).as_posix():sha(p) for p in sorted((PUBLIC/'hypernet').rglob('*.py'))}
    unchanged=sum(core.get(path)==expected for path,expected in prior['core_files'].items())
    write(PUBLIC/'publication/source_snapshot.json',{'snapshot':'HyperNet v0.12.0 fresh native-detail hybrid','core_file_count':len(core),'core_files':core,'prior_snapshot':'publication/source_snapshot_v0_11.json','unchanged_prior_core_files':unchanged,'prior_version_metadata_change':'hypernet/__init__.py: __version__ changes from 0.3.0 to 0.12.0, identical to frozen v12 source','original_protocol_sha256':sha(root/'experiments/detail_protocol_freeze.json'),'additional_exported_files':exported,'evaluation_status':'Independent fresh audit passed; exact source export.','notes':['Synthetic-only proposal CNN plus analytic local patch matching.','One reference per enrollment; no v12 optimizer updates.','Original protocol paths sanitized only in public evidence JSON; source bytes unchanged.']})
    p=PUBLIC/'pyproject.toml';p.write_text(p.read_text().replace('version = "0.11.0"','version = "0.12.0"'))
    ignore=PUBLIC/'.gitignore';s=ignore.read_text()
    if '!data/SHAPES2D_FRESH_CONTRACT.json' not in s:ignore.write_text(s+'\n!data/SHAPES2D_FRESH_CONTRACT.json\n')
    args.assets.mkdir(parents=True,exist_ok=True)
    checkpoint=root.parent/'hypernet_v11/checkpoints/detector_micro_aware.pt';shutil.copyfile(checkpoint,args.assets/checkpoint.name)
    manifest=read(dest/'cohort_manifest.json');rgb={record['rgb_path'] for ep in manifest['episodes'] for record in [ep['support']]+ep['queries']}
    with zipfile.ZipFile(args.assets/'fresh_v12_rgb.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,source in [('manifest.json',dest/'cohort_manifest.json')]+[(name,root/'data/shapes2d_fresh_v12'/name) for name in sorted(rgb)]:
            info=zipfile.ZipInfo(name,date_time=(2026,11,2,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,source.read_bytes())
    write(dest/'release_assets.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size,'url':'https://github.com/h3clinic/HyperNet/releases/download/v0.12.0/'+p.name} for p in sorted(args.assets.iterdir()) if p.is_file()})
    print(json.dumps({'core_files':len(core),'prior_core_unchanged':unchanged,'evidence_files':len(inputs),'prediction_rows':6144,'release_assets':read(dest/'release_assets.json')},indent=2))
if __name__=='__main__':main()
