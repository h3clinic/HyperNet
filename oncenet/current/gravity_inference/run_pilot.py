"""Frozen Gravity Spy portability pilot. No training, downloads or new benchmark."""
from pathlib import Path
from fractions import Fraction
import argparse, hashlib, importlib.util, json, multiprocessing as mp, os, sys, time, traceback
import numpy as np

DURATIONS=('d05','d1','d2','d4')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(value,message):
    if not value:raise RuntimeError(message)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def verify_payload(root):
    manifest=read(root/'manifest.json')
    require(sha(root/'manifest.json')==(root/'manifest.sha256').read_text().split()[0],'Manifest digest mismatch')
    for name,digest in manifest['file_hashes'].items():
        p=(root/name).resolve();require(p.is_relative_to(root.resolve()),'Payload path escapes root')
        require(p.is_file() and sha(p)==digest,'Payload mismatch: '+name)
    events=read(root/'events.json');plan=read(root/'fixture_plan.json')
    require(len(events)==12 and sum(e['role']=='query' for e in events)==8,'Wrong fixture population')
    require([e['event_id'] for e in events[:8]]==plan['query_ids'],'Query plan changed')
    require([e['event_id'] for e in events[8:]]==plan['support_ids'],'Support plan changed')
    require(len({e['event_id'] for e in events})==12,'Repeated event')
    require(plan['scores_not_read_before_this_plan'] is True and plan['reserved_closed'] is True,'Invalid fixture declaration')
    for target in manifest['targets']:
        require(sum(e['role']=='query' and e['label']==target for e in events)==2,'Unbalanced fixture')
        require(sum(e['role']=='support' and e['label']==target for e in events)==1,'Expected one existing positive support per target')
    return manifest,events,plan

def calibrate(cal,scores,alpha):
    cal=np.asarray(cal,np.float64);scores=np.asarray(scores,np.float64)
    require(np.isfinite(cal).all() and np.isfinite(scores).all(),'Nonfinite calibration/score')
    require(len(cal)>0 and np.all(cal[:-1]<=cal[1:]),'Unsorted calibration')
    a=Fraction(str(alpha));k=(len(cal)+1)*a.numerator//a.denominator
    nums=1+len(cal)-np.searchsorted(cal,scores,side='left')
    return nums/(len(cal)+1),nums<=k,float(cal[-k]) if k else None

def score_features(root,manifest,features):
    for d in DURATIONS:
        require(features[d].dtype==np.float32 and features[d].shape[1]==4032,'Wrong descriptor dtype/shape')
        require(np.isfinite(features[d]).all(),'Nonfinite descriptors')
    joint=(np.concatenate([features[d] for d in DURATIONS],axis=1)/2).astype(np.float64)
    states=np.load(root/'matcher.npz',allow_pickle=False);cal=np.load(root/'calibration.npz',allow_pickle=False)
    result={}
    for target in manifest['targets']:
        weight=states[target+'__weight'];require(weight.shape==(16128,),'Wrong matcher shape')
        require(int(states[target+'__effective_dimension'])==4032,'Effective fit dimension changed')
        score=np.einsum('ij,j->i',joint,weight,optimize=False)+states[target+'__offset']
        result['scores__'+target]=score
        for alpha in manifest['alphas']:
            p,decision,cut=calibrate(cal[target+'__scores'],score,alpha)
            result['pvalues__'+target]=p
            result['decisions__'+target+'__'+str(alpha)]=decision
    return result

def saved_feature_check(root,manifest):
    z=np.load(root/'expected.npz',allow_pickle=False)
    result=score_features(root,manifest,{d:z['features__'+d] for d in DURATIONS})
    maxerr=0.
    for key,value in result.items():
        if key.startswith('scores__'):
            err=float(np.max(np.abs(value-z[key])));maxerr=max(maxerr,err)
            require(np.allclose(value,z[key],atol=2e-7,rtol=2e-8),'Saved feature score mismatch '+key)
        else:require(np.array_equal(value,z[key]),'Saved decision/pvalue mismatch '+key)
    _,tie,_=calibrate(np.array([1.,2.,3.,4.]),np.array([4.,4.0001]),.2)
    require(np.array_equal(tie,[False,True]),'Conservative tie test failed')
    return dict(passed=True,event_count=12,target_alpha_rows=8,max_score_error=maxerr,
                all_saved_decisions_and_pvalues_exact=True,neural_inference=False,reserved_access=False)

def worker(root_text,output_text,rank,workers,repeats):
    root=Path(root_text);output=Path(output_text);started=time.perf_counter()
    try:
        os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
        import torch
        from PIL import Image
        torch.set_num_threads(2);torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        manifest=read(root/'manifest.json');events=read(root/'events.json');plan=read(root/'fixture_plan.json')
        indices=np.array([i for i in range(len(events)) if i%workers==rank],int)
        expected=np.load(root/'expected.npz',allow_pickle=False)
        preprocessing=module('pilot_preprocess_'+str(rank),root/'preprocess.py')
        arch=module('pilot_arch_'+str(rank),root/'model.py')
        net=arch.load_encoder(root/'encoder.pt','cpu').to('cuda:'+str(rank)).eval()
        require(sum(p.numel() for p in net.parameters())==42480,'Wrong encoder parameter count')
        arrays={};preprocess_start=time.perf_counter();max_input_error=0.
        for d in DURATIONS:
            for view in ['full','low']:
                images=[]
                for i in indices:
                    with Image.open(root/events[i]['windows'][d]['path']) as im:
                        require(im.format=='PNG' and im.size==(800,600),'Not a native Gravity Spy PNG')
                        rgb=np.asarray(im.convert('RGB'))
                    images.append(preprocessing.preprocess(rgb,'energy_'+view,128)[:,:,0])
                arr=np.stack(images).astype(np.float32);arrays[(d,view)]=arr
                err=float(np.max(np.abs(arr-expected['input__'+view+'__'+d][indices])));max_input_error=max(max_input_error,err)
                require(err<=plan['tolerances']['preprocess_atol'],'PNG preprocessing mismatch')
        preprocessing_seconds=time.perf_counter()-preprocess_start
        features={};pass_seconds=[]
        with torch.inference_mode():
            for repeat in range(repeats):
                torch.cuda.synchronize(rank);tick=time.perf_counter()
                for d in DURATIONS:
                    parts=[]
                    for view in ['full','low']:
                        x=torch.from_numpy(arrays[(d,view)][:,None].copy()).repeat(1,3,1,1).to(rank)
                        parts.append(arch.descriptor_base(net.net(x)).cpu().numpy())
                    z=np.concatenate(parts,axis=1)
                    features[d]=(z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)).astype(np.float32)
                torch.cuda.synchronize(rank);pass_seconds.append(time.perf_counter()-tick)
        scored=score_features(root,manifest,features);checks=[];ok=True;max_feature_error=0.
        for d in DURATIONS:
            target=expected['features__'+d][indices];err=float(np.max(np.abs(features[d]-target)))
            passed=bool(np.allclose(features[d],target,atol=plan['tolerances']['feature_atol'],rtol=plan['tolerances']['feature_rtol']))
            max_feature_error=max(max_feature_error,err);ok=ok and passed
            checks.append(dict(kind='features',duration=d,passed=passed,max_absolute_error=err))
        for key,value in scored.items():
            target=expected[key][indices]
            if key.startswith('scores__'):
                err=float(np.max(np.abs(value-target)))
                passed=bool(np.allclose(value,target,atol=plan['tolerances']['score_atol'],rtol=plan['tolerances']['score_rtol']))
            else:
                err=float(np.max(np.abs(value.astype(float)-target.astype(float))))
                passed=bool(np.array_equal(value,target))
            ok=ok and passed;checks.append(dict(kind=key,passed=passed,max_absolute_error=err))
        np.savez_compressed(output/f'gpu_{rank}_features_scores.npz',indices=indices,
                            **{'features__'+d:v for d,v in features.items()},**scored)
        write(output/f'gpu_{rank}.json',dict(status='completed',passed=ok,rank=rank,device=torch.cuda.get_device_name(rank),
            event_indices=indices.tolist(),event_ids=[events[i]['event_id'] for i in indices],
            event_count=len(indices),png_count=len(indices)*4,encoder_parameters=42480,
            preprocess_seconds=preprocessing_seconds,max_preprocess_error=max_input_error,
            inference_pass_seconds=pass_seconds,repeats=repeats,max_feature_error=max_feature_error,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(rank),elapsed_seconds=time.perf_counter()-started,
            torch_version=torch.__version__,cuda_version=torch.version.cuda,checks=checks,
            calibration_or_weight_updates=0,scope='Portability and tiny repeated-fixture timing, not new accuracy or archive throughput'))
        if not ok:raise RuntimeError('GPU parity checks failed; retain output and do not adjust tolerances or frozen artifacts')
    except Exception as exc:
        write(output/f'gpu_{rank}_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,default=Path.cwd()/'pilot_results')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--max-gpus',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    require(1<=args.max_gpus<=2,'This pilot permits at most two independent GPU workers')
    require(1<=args.repeats<=5,'One to five timing passes permitted')
    manifest,events,plan=verify_payload(root);cached=saved_feature_check(root,manifest)
    output.mkdir(parents=True,exist_ok=True)
    require(not (output/'pilot_result.json').exists(),'Completed output exists; choose a new output directory if intentionally repeating')
    binding=dict(manifest_sha256=sha(root/'manifest.json'),pilot_code_sha256=sha(Path(__file__)),fixture_plan_sha256=sha(root/'fixture_plan.json'))
    if args.verify_only:
        result=dict(status='cpu_verification_only',passed=True,binding=binding,cached=cached,gpu_jobs=0,neural_inference=False)
        write(output/'pilot_result.json',result);print(json.dumps(result));return
    import torch
    devices=torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not devices:
        result=dict(status='no_gpu_cpu_verification_only',passed=True,binding=binding,cached=cached,gpu_jobs=0,neural_inference=False,
                    next_action='Enable a GPU and use a new output folder; CPU fallback deliberately does not run neural inference.')
        write(output/'pilot_result.json',result);print(json.dumps(result));return
    workers=min(args.max_gpus,devices);started=time.perf_counter();ctx=mp.get_context('spawn')
    jobs=[ctx.Process(target=worker,args=(str(root),str(output),rank,workers,args.repeats)) for rank in range(workers)]
    for job in jobs:job.start()
    for job in jobs:job.join()
    reports=[read(output/f'gpu_{rank}.json') for rank in range(workers) if (output/f'gpu_{rank}.json').exists()]
    indices=[i for report in reports for i in report['event_indices']]
    passed=(all(j.exitcode==0 for j in jobs) and len(reports)==workers and sorted(indices)==list(range(12)) and all(r['passed'] for r in reports))
    result=dict(status='completed' if passed else 'failed',passed=passed,binding=binding,cached=cached,gpu_jobs=workers,
        gpu_reports=reports,wall_seconds=time.perf_counter()-started,event_count=12,query_events=8,support_events=4,pngs=48,
        no_training=True,no_refit=True,no_downloads=True,reserved_closed=True,
        limitation='This is an already-used tiny conversion fixture. Do not report accuracy, recall, discoveries, or archive ETA from it.')
    write(output/'pilot_result.json',result);print(json.dumps(result,indent=2));require(passed,'Pilot failed; inspect saved reports')

if __name__=='__main__':main()
