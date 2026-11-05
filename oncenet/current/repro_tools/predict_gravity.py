"""Portable CPU query interface for the frozen strict-FP32 OnceNet bundle.

Default: verify all original payload hashes and replay saved USED-fixture
descriptors only. --query explicitly processes four native Gravity Spy PNGs
from one event. --fixture-test explicitly checks CPU numerical portability on
the existing 12 USED events; it is not a scientific performance evaluation.
No encoder training, target fitting, downloads or threshold changes occur.
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

DUR=('d05','d1','d2','d4')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def jsonable(x):
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x

def image_features(bundle,queries):
    import torch
    from PIL import Image
    torch.set_num_threads(2);torch.backends.mkldnn.enabled=True
    model=module('frozen_portable_architecture',bundle/'model.py')
    prep=module('frozen_portable_preprocessing',bundle/'preprocess.py')
    net=model.load_encoder(bundle/'encoder.pt','cpu').eval()
    if sum(p.numel() for p in net.parameters())!=42480:raise RuntimeError('Wrong encoder size')
    features={};inputs={}
    with torch.inference_mode():
        for d in DUR:
            pairs=[]
            for view in ['full','low']:
                arrays=[]
                for q in queries:
                    with Image.open(q[d]) as im:
                        if im.format!='PNG' or im.size!=(800,600):raise ValueError('Expected native 800x600 Gravity Spy PNG')
                        rgb=np.asarray(im.convert('RGB'))
                    prep.validate_renderer(rgb)
                    arrays.append(prep.preprocess(rgb,'energy_'+view,128)[:,:,0])
                a=np.stack(arrays).astype(np.float32);inputs[view+'__'+d]=a
                chunks=[]
                for i in range(0,len(a),32):
                    x=torch.from_numpy(a[i:i+32].copy()[:,None]).repeat(1,3,1,1)
                    chunks.append(model.descriptor_base(net.net(x)).numpy())
                pairs.append(np.concatenate(chunks))
            z=np.concatenate(pairs,1)
            features[d]=(z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)).astype(np.float32)
    return features,inputs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True,help='Full strict-FP32 kaggle_pilot directory')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--query',action='store_true');mode.add_argument('--fixture-test',action='store_true')
    for d in DUR:p.add_argument('--'+d,type=Path)
    p.add_argument('--output',type=Path,help='Optional new JSON file; never overwritten')
    a=p.parse_args();bundle=a.bundle.resolve()
    if a.output and a.output.exists():raise ValueError('Output exists; refusing overwrite')
    runtime=module('frozen_portable_runtime',bundle/'run_pilot.py')
    manifest,events,plan=runtime.verify_payload(bundle)
    cached=runtime.saved_feature_check(bundle,manifest)
    base={'payload_verified':True,'cached_feature_check':cached,'primary_alpha':.001,
        'historical_bundle_secondary_alpha':.0005,
        'note':'The paper confirmation separately evaluated alpha0.01; this wrapper preserves the original pilot bundle alphas.',
        'encoder_parameters':42480,'training_updates':0,'target_refits':0}
    if not(a.query or a.fixture_test):
        out=base|{'neural_inference':False,'reserved_access':False}
    else:
        if a.fixture_test:
            queries=[{d:bundle/e['windows'][d]['path'] for d in DUR} for e in events]
        else:
            queries=[{d:getattr(a,d) for d in DUR}]
            if not all(queries[0].values()):raise ValueError('--query requires --d05 --d1 --d2 --d4')
            if len({p.resolve() for p in queries[0].values()})!=4:raise ValueError('Four distinct authentic duration PNGs are required')
        features,inputs=image_features(bundle,queries)
        scored=runtime.score_features(bundle,manifest,features)
        if a.fixture_test:
            expected=np.load(bundle/'expected.npz',allow_pickle=False)
            feature_errors={d:float(np.max(np.abs(features[d]-expected['features__'+d]))) for d in DUR}
            input_errors={k:float(np.max(np.abs(v-expected['input__'+k]))) for k,v in inputs.items()}
            tol=plan['tolerances'];fail=[]
            for k,v in input_errors.items():
                if v>tol['preprocess_atol']:fail.append('preprocessing '+k)
            for d in DUR:
                if not np.allclose(features[d],expected['features__'+d],atol=tol['feature_atol'],rtol=tol['feature_rtol']):fail.append('features '+d)
            for k,v in scored.items():
                if k.startswith('scores__'):
                    if not np.allclose(v,expected[k],atol=tol['score_atol'],rtol=tol['score_rtol']):fail.append(k)
                elif not np.array_equal(v,expected[k]):fail.append(k)
            out=base|{'passed':not fail,'cpu_fixture_only':True,'already_used_events':12,
                'feature_max_errors':feature_errors,'input_max_errors':input_errors,
                'failures':fail,'benchmark_run':False,'reserved_access':False}
            if fail:raise RuntimeError('CPU portability check failed; do not adjust thresholds or tolerances: '+repr(out))
        else:
            out=base|{'query_events':1,'neural_inference':True,'scores':scored,
                'interpretation':'Fixed target resemblance and calibrated alarms, not confirmed scientific identities or discoveries.'}
    out=jsonable(out)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2,allow_nan=False))

if __name__=='__main__':main()
