"""Recompute six-method metrics, localization and paired family intervals without images."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_v11_metrics',ROOT/'scripts/audit_2d_detector.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def read(p):return json.loads(Path(p).read_text())
def require(value,message):
    if not value:raise ValueError(message)
def bounds(x):
    values=sorted(x);output=[]
    for p in [.025,.975]:
        index=(len(values)-1)*p;lo=int(index);hi=min(lo+1,len(values)-1)
        output.append(values[lo]+(index-lo)*(values[hi]-values[lo]))
    return output
def main():
    ap=argparse.ArgumentParser();ap.add_argument('benchmark',type=Path,nargs='?',default=ROOT/'benchmarks/v0.12');b=ap.parse_args().benchmark
    result=read(b/'aggregate_metrics.json');manifest=read(b/'cohort_manifest.json');provenance=read(b/'public_export_provenance.json')
    for name,record in provenance['files'].items():require(hashlib.sha256((b/name).read_bytes()).hexdigest()==record['public_sha256'],'Public export hash changed')
    truth={q['id']:q for ep in manifest['episodes'] for q in ep['queries']};rows={};cuts={};metrics=0
    for alias,record in result['score_files'].items():
        path=b/record['path'];require(hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],'Original score hash changed')
        values=[json.loads(s) for s in path.read_text().splitlines()];rows[alias]=values
        require(len(values)==len(truth)==1024 and {r['id'] for r in values}==set(truth),'Missing or duplicate prediction')
        for r in values:
            q=truth[r['id']];require(r['present']==q['present'],'Wrong label')
            require(abs(audit.overlap(r['bbox_xyxy'],q['target_bbox_xyxy'])-r['iou'])<1e-12,'Selected IoU differs')
            require(abs(max([audit.overlap(box,q['target_bbox_xyxy']) for box in r['proposals']],default=0)-r['proposal_best_iou'])<1e-12,'Proposal IoU differs')
        cuts[alias]=audit.dev_cutoff([r for r in values if r['split']=='dev'])
        for split in ['dev','test']:
            actual=audit.independent_metrics([r for r in values if r['split']==split],cuts[alias]);expected=result['methods'][alias][split]
            require(set(actual)==set(expected),'Metric schema changed')
            for key,value in actual.items():require(abs(value-expected[key])<1e-12,f'Metric differs: {alias}/{split}/{key}');metrics+=1
    families=sorted({r['family_id'] for r in rows['native_patch'] if r['split']=='test'});require(len(families)==128,'Wrong clusters')
    draws=np.random.default_rng(1812).integers(0,128,(10000,128));values={};distributions={};checks=0
    for alias,records in rows.items():
        values[alias]=np.array([sum(r['score']>=cuts[alias] and r['iou']>=.5 for r in records if r['split']=='test' and r['family_id']==f and r['present'])/2 for f in families])
        distributions[alias]=values[alias][draws].mean(1)
        actual=[float(values[alias].mean()),*bounds(distributions[alias])];expected=result['bootstrap']['methods'][alias]
        require(np.max(np.abs(np.array(actual)-[expected['estimate'],*expected['ci95']]))<1e-12,'Method interval differs');checks+=3
    for control in ['old_cnn','lowres_patch']:
        expected=result['bootstrap']['differences']['native_patch_minus_'+control]
        actual=[float((values['native_patch']-values[control]).mean()),*bounds(distributions['native_patch']-distributions[control])]
        require(np.max(np.abs(np.array(actual)-[expected['estimate'],*expected['ci95']]))<1e-12,'Paired interval differs');checks+=3
    print(json.dumps({'verified':True,'prediction_rows':6144,'metric_values':metrics,'bootstrap_values':checks,'scope':'Arithmetic verification of unchanged saved predictions; use examples/replay_v12.py for inference.'},indent=2))
if __name__=='__main__':main()
