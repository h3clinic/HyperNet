"""Bounded NOAA Bering search. Default is a two-file, two-GPU portability pilot.

Never imports the original workspace, fits a threshold, or writes its outputs.
Candidate scores are not calibrated detections, accuracy, or discoveries.
"""
from pathlib import Path
import argparse, base64, hashlib, json, os, shutil, signal, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from urllib.parse import quote

HERE=Path(__file__).resolve().parent
ALLOWED={'mbarc_lmr_bering01-m2','mbarc_lmr_bering04-m4'}
sys.path.insert(0,str(HERE/'whale/code'))

def now():return datetime.now(timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def verify_bundle():
    expected=json.loads((HERE/'asset_hashes.json').read_text())
    for name,sha in expected.items():
        p=(HERE/name).resolve()
        assert p.is_relative_to(HERE) and p.is_file() and digest(p)==sha, f'Asset mismatch: {name}'
    return digest(HERE/'asset_hashes.json')

def manifest():
    m=json.loads((HERE/'manifest.json').read_text())
    assert set(m['allowed_deployments'])==ALLOWED
    seen=set()
    for n,r in enumerate(m['items']):
        assert n==r['index'] and r['deployment'] in ALLOWED and r['gpu_rank']==n%2
        assert r['object_key'].startswith(f"mbarc_bering/audio/{r['deployment']}/audio/")
        assert r['object_key'].endswith('.flac') and '..' not in r['object_key']
        assert r['object_key'] not in seen;seen.add(r['object_key'])
    return m

def complete_valid(output,item,asset_hash):
    meta=Path(output)/item['output_id']/'commit.json'
    if not meta.exists():return False
    r=json.loads(meta.read_text())
    return r.get('status')=='completed' and r['object_key']==item['object_key'] and r['asset_hashes_sha256']==asset_hash and all((meta.parent/name).is_file() and digest(meta.parent/name)==sha for name,sha in r['output_sha256'].items())

def download(item,dest,progress,seconds=600,retries=2,max_bytes=4_000_000_000):
    end=time.monotonic()+seconds;errors=[]
    url='https://storage.googleapis.com/noaa-passive-bioacoustic/'+quote(item['object_key'],safe='/')
    heads=HERE/'pilot_source_head.json'
    fixed=next((r for r in json.loads(heads.read_text()) if r.get('object_key')==item['object_key'] and r.get('http_status')==200),None) if heads.exists() else None
    if fixed:url+='?generation='+quote(fixed['generation'],safe='')
    for attempt in range(retries+1):
        if time.monotonic()>=end:break
        total=0;md5=hashlib.md5();t=time.monotonic();last=t
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'HyperNet-W-bounded-research-pilot/1.0'})
            with urllib.request.urlopen(req,timeout=min(30,max(1,end-time.monotonic()))) as r,open(dest,'wb') as f:
                length=int(r.headers['Content-Length']) if r.headers.get('Content-Length') else None
                if length is not None and length>max_bytes:raise ValueError('Object exceeds byte cap')
                if fixed and (length!=fixed['bytes'] or r.headers.get('x-goog-generation')!=fixed['generation']):raise ValueError('Pinned pilot object changed')
                response={k:r.headers.get(k) for k in ['ETag','Content-Length','x-goog-generation','x-goog-hash','Last-Modified']}
                expected_md5=next((s.split('=',1)[1] for s in (r.headers.get('x-goog-hash') or '').replace(' ','').split(',') if s.startswith('md5=')),None)
                expected_crc=next((s.split('=',1)[1] for s in (r.headers.get('x-goog-hash') or '').replace(' ','').split(',') if s.startswith('crc32c=')),None)
                crc=None
                if expected_crc:
                    import google_crc32c
                    crc=google_crc32c.Checksum()
                while True:
                    if time.monotonic()>end:raise TimeoutError('Total download deadline')
                    b=r.read(1<<20)
                    if not b:break
                    total+=len(b)
                    if total>max_bytes:raise ValueError('Object exceeds byte cap')
                    f.write(b);md5.update(b)
                    if crc is not None:crc.update(b)
                    if time.monotonic()-last>5:
                        progress('downloading',bytes=total,expected_bytes=length,attempt=attempt,elapsed_seconds=time.monotonic()-t);last=time.monotonic()
                f.flush();os.fsync(f.fileno())
            if length is not None and total!=length:raise IOError(f'Incomplete download {total}/{length}')
            if expected_md5 and base64.b64encode(md5.digest()).decode()!=expected_md5:raise IOError('GCS MD5 mismatch')
            if expected_crc and base64.b64encode(crc.digest()).decode()!=expected_crc:raise IOError('GCS CRC32C mismatch')
            return {'url':url,'bytes':total,'sha256':digest(dest),'response_headers':response,'server_md5_verified':bool(expected_md5),'server_crc32c_verified':bool(expected_crc),'generation_pinned_before_run':bool(fixed),'seconds':time.monotonic()-t,'attempts':attempt+1,'previous_errors':errors}
        except Exception as e:
            errors.append(repr(e));progress('download_retry',bytes=total,attempt=attempt,error=repr(e))
            if attempt<retries and time.monotonic()<end:time.sleep(min(2*(attempt+1),max(0,end-time.monotonic())))
    raise TimeoutError('Download attempts/deadline exhausted: '+repr(errors))

class SharedScorer:
    """Same frozen query tokens reused only for same-speed targets."""
    def __init__(self,device):
        import numpy as np,torch
        import oneshot as O
        self.np=np;self.torch=torch;self.O=O
        O.DEV=device
        torch.set_num_threads(4)
        if device.startswith('cuda'):
            torch.backends.mha.set_fastpath_enabled(False)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
        torch.set_float32_matmul_precision('highest')
        self.device=device;self.matchers={};self.speeds={}
        for target in ('sei','narw','bloop'):
            e=np.load(HERE/f'whale/examples/{target}.npz',allow_pickle=False)
            self.matchers[target]=O.OneShot(e['window'],tuple(map(int,e['box'])))
            self.speeds[target]=int(e['speed'])
        assert self.speeds=={'sei':1,'narw':1,'bloop':8}
        # No target-specific encoder edits. Verify equality before sharing tokens.
        anchor=self.matchers['sei'].enc.state_dict()
        assert all(torch.equal(anchor[k],m.enc.state_dict()[k]) for m in self.matchers.values() for k in anchor)

    def score(self,windows,targets):
        torch=self.torch;np=self.np;O=self.O;F=O.F;E=O.E
        assert len({self.speeds[t] for t in targets})==1
        with torch.no_grad():
            q=torch.as_tensor(np.asarray(windows)).float().div(255).to(self.device)
            anchor=self.matchers[targets[0]]
            tokens={m:E.raw_layer_tokens(anchor.m,anchor.enc,q,'final' if m=='v3' else 'pixels') for m in ('v3','pixels')}
            result={}
            for target in targets:
                obj=self.matchers[target];out={}
                for meth in obj.methods:
                    if meth=='xcorr':
                        k,r0,r1=obj.kern;L_=k.shape[1];qq=q[:,r0:r1]
                        num=F.conv1d(qq,k[None])[:,0]
                        ones=torch.ones(1,qq.shape[1],L_,device=self.device)
                        s1=F.conv1d(qq,ones)[:,0];s2=F.conv1d(qq*qq,ones)[:,0]
                        den=(s2-s1*s1/(qq.shape[1]*L_)).clamp_min(1e-6).sqrt()
                        out[meth]=(num/den).amax(1).cpu().numpy()
                    else:
                        _,reps=obj.refs[meth]
                        out[meth]=torch.stack([E.block_cosine(tokens[meth],rd,mk) for rd,mk in reps]).amax(0).cpu().numpy()
                result[target]=out
            return result

def save_npz(path,arrays):
    import numpy as np
    tmp=path.with_suffix('.tmp.npz')
    with open(tmp,'wb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)==set(arrays)
        assert all(z[k].dtype==np.float32 and np.array_equal(z[k],arrays[k]) and np.isfinite(z[k]).all() for k in z.files)

def run_item(args):
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    import legacy_audio as A,spectro as S
    assets=verify_bundle();m=manifest();item=m['items'][args.one_item]
    assert item['gpu_rank']==args.rank
    out=Path(args.output).resolve();work=out/'temporary'/item['output_id'];dst=out/item['output_id']
    if complete_valid(out,item,assets):return
    work.mkdir(parents=True,exist_ok=True);dst.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic()
    def progress(stage,**extra):
        atomic_json(out/f'progress_gpu{args.rank}.json',dict(updated_utc=now(),index=item['index'],object_key=item['object_key'],stage=stage,wall_seconds=time.monotonic()-t0,**extra))
    raw=work/'original.flac';cache=work/'resampled.flac'
    try:
        dl=download(item,raw,progress,args.download_timeout,args.retries)
        progress('converting',download=dl)
        t=time.monotonic();seconds,peak,orig_seconds=A.convert(str(raw),str(cache));convert_s=time.monotonic()-t
        cached_sha=digest(cache)
        x,sr=sf.read(cache,dtype='float32');assert sr==1000 and x.ndim==1
        progress('loading_model',audio_seconds=seconds,bytes=dl['bytes'])
        assert torch.cuda.is_available(), 'Cloud runner requires the assigned CUDA GPU'
        scorer=SharedScorer('cuda:0');outputs={};stages={};windows={}
        for speed,targets in [(1,['sei','narw']),(8,['bloop'])]:
            t=time.monotonic();signal=x if speed==1 else resample_poly(x,1,speed).astype(np.float32)
            w=A.spectro_chunks(signal)
            starts=np.arange(0,max(0,w.shape[1]-S.WIN_COLS+1),80)
            if len(starts)==0:raise ValueError('Audio has no complete 4-second window')
            slices=np.lib.stride_tricks.sliding_window_view(w,S.WIN_COLS,axis=1)
            accum={target:{method:[] for method in ('v3','pixels','xcorr')} for target in targets}
            prep_s=time.monotonic()-t;infer=time.monotonic()
            for a in range(0,len(starts),args.batch_size):
                win=slices[:,starts[a:a+args.batch_size]].transpose(1,0,2)
                scores=scorer.score(win,targets)
                for target in targets:
                    for method in accum[target]:accum[target][method].append(scores[target][method])
                progress('scoring',speed=speed,audio_seconds=seconds,bytes=dl['bytes'],windows_done=min(a+args.batch_size,len(starts)),windows_total=len(starts))
            infer_s=time.monotonic()-infer
            for target in targets:
                arrays={'start_s':(starts*S.HOP/S.SR*speed).astype(np.float32),**{k:np.concatenate(v).astype(np.float32) for k,v in accum[target].items()}}
                score_path=dst/f'{target}.npz';save_npz(score_path,arrays)
                outputs[score_path.name]=digest(score_path);windows[target]=len(starts)
            stages[str(speed)]={'preprocess_seconds':prep_s,'inference_seconds':infer_s,'windows':len(starts),'targets':targets}
        result={'status':'completed','completed_utc':now(),'object_key':item['object_key'],'deployment':item['deployment'],'index':item['index'],'gpu_rank':args.rank,
                'asset_hashes_sha256':assets,'manifest_sha256':digest(HERE/'manifest.json'),'protocol_sha256':digest(HERE/'protocol.json'),
                'source':dl,'cached_1khz_sha256':cached_sha,'audio_seconds':seconds,'original_audio_seconds':orig_seconds,'normalization_peak':peak,
                'convert_seconds':convert_s,'stages':stages,'output_sha256':outputs,'score_dtype':'float32','windows':windows,
                'wall_seconds':time.monotonic()-t0,'hardware':torch.cuda.get_device_name(0),'torch':torch.__version__,'numpy':np.__version__,
                'python':sys.version,'batch_size':args.batch_size,'tf32':False,'autocast':False,
                'one_positive_reference_per_target':True,'training':'Existing HyperNet-W v3, 77000 real unlabeled windows, no borrowed weights',
                'interpretation':'Uncalibrated candidate rankings; no accuracy, false-alarm, species confirmation, or discovery claim',
                'start_s_convention':'Matches legacy window-column offsets; not a corrected GPS timestamp. Recording-time metadata and STFT-center offsets require downstream review.'}
        atomic_json(dst/'commit.json',result)
        assert complete_valid(out,item,assets)
        progress('completed',audio_seconds=seconds,bytes=dl['bytes'],result=str(dst/'commit.json'))
    except BaseException as e:
        atomic_json(dst/'failure.json',{'status':'failed','object_key':item['object_key'],'updated_utc':now(),'error':repr(e),'wall_seconds':time.monotonic()-t0})
        raise
    finally:
        # Scores and source checksum are committed before successful input cleanup.
        # Failed attempts have a failure record; temporary audio is never retained indefinitely.
        if work.is_relative_to(out/'temporary') and work.exists():shutil.rmtree(work)

def worker(args):
    assets=verify_bundle();m=manifest();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    items=[r for r in m['items'][:args.max_files] if r['gpu_rank']==args.rank]
    rows=[]
    for item in items:
        remaining=args.deadline_epoch-time.time()
        if remaining<30:break
        if complete_valid(out,item,assets):
            rows.append({'index':item['index'],'status':'already_completed'});continue
        command=[sys.executable,str(HERE/'scanner.py'),'--one-item',str(item['index']),'--rank',str(args.rank),'--output',str(out),'--batch-size',str(args.batch_size),'--download-timeout',str(args.download_timeout),'--retries',str(args.retries)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.rank),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        log=out/f'item_{item["index"]:05d}.log'
        t=time.monotonic()
        try:
            with open(log,'a',encoding='utf-8') as f:
                run=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=min(args.file_timeout,remaining))
            state='completed' if run.returncode==0 and complete_valid(out,item,assets) else 'failed'
            error=None if state=='completed' else f'exit {run.returncode}'
        except subprocess.TimeoutExpired:
            state='timed_out';error='Per-file or overall deadline reached'
        rows.append({'index':item['index'],'status':state,'error':error,'wall_seconds':time.monotonic()-t})
        atomic_json(out/f'worker_{args.rank}.json',{'updated_utc':now(),'status':'advancing','rank':args.rank,'results':rows})
        # Timed-out children cannot run their finally block; remove only this explicit scratch path.
        scratch=out/'temporary'/item['output_id']
        if scratch.exists() and scratch.resolve().is_relative_to((out/'temporary').resolve()):shutil.rmtree(scratch)
    atomic_json(out/f'worker_{args.rank}.json',{'updated_utc':now(),'status':'finished','rank':args.rank,'results':rows})

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='/kaggle/working/bering_results');p.add_argument('--max-files',type=int,default=2)
    p.add_argument('--wall-seconds',type=int,default=14400);p.add_argument('--file-timeout',type=int,default=1800)
    p.add_argument('--download-timeout',type=int,default=600);p.add_argument('--retries',type=int,default=2);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--worker',action='store_true');p.add_argument('--one-item',type=int)
    p.add_argument('--deadline-epoch',type=float,default=0);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    verify_bundle();m=manifest()
    assert 1<=args.max_files<=len(m['items']) and 0<args.wall_seconds<=14400
    assert args.rank in (0,1) and 1<=args.batch_size<=512
    assert 30<=args.download_timeout<=900 and 60<=args.file_timeout<=1800 and 0<=args.retries<=3
    if args.verify_only:print(json.dumps({'status':'verified','keys':len(m['items']),'pilot':m['pilot_indices'],'no_inference':True}));return
    if args.one_item is not None:run_item(args);return
    if args.worker:worker(args);return
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'active.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Output already claimed; verify old job stopped before removing active.lock')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.time();deadline=start+args.wall_seconds;children=[]
    atomic_json(out/'run_manifest.json',{'started_utc':now(),'manifest_sha256':digest(HERE/'manifest.json'),'asset_hashes_sha256':digest(HERE/'asset_hashes.json'),'indices':list(range(args.max_files)),'max_wall_seconds':args.wall_seconds,'no_threshold_fitting':True})
    try:
        for rank in (0,1):
            cmd=[sys.executable,str(HERE/'scanner.py'),'--worker','--rank',str(rank),'--output',str(out),'--max-files',str(args.max_files),'--deadline-epoch',str(deadline),'--file-timeout',str(args.file_timeout),'--download-timeout',str(args.download_timeout),'--retries',str(args.retries),'--batch-size',str(args.batch_size)]
            children.append(subprocess.Popen(cmd,start_new_session=(os.name=='posix')))
        while any(c.poll() is None for c in children) and time.time()<deadline:time.sleep(.5)
        for c in children:
            if c.poll() is None:
                if os.name=='posix':os.killpg(c.pid,signal.SIGTERM)
                else:c.terminate()
        for c in children:
            try:c.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name=='posix':os.killpg(c.pid,signal.SIGKILL)
                else:c.kill()
                c.wait()
        reports=[json.loads((out/f'worker_{r}.json').read_text()) for r in (0,1) if (out/f'worker_{r}.json').exists()]
        commits=[json.loads((out/r['output_id']/'commit.json').read_text()) for r in m['items'][:args.max_files] if complete_valid(out,r,digest(HERE/'asset_hashes.json'))]
        summary={'status':'finished' if len(commits)==args.max_files else 'partial_or_failed','finished_utc':now(),'wall_seconds':time.time()-start,'requested_files':args.max_files,'completed_files':len(commits),'audio_hours':sum(r['audio_seconds'] for r in commits)/3600,'download_bytes':sum(r['source']['bytes'] for r in commits),'workers':reports,'discoveries_confirmed':0,'accuracy_evaluation':False}
        atomic_json(out/'run_summary.json',summary)
        print(json.dumps(summary),flush=True)
    finally:
        for c in children:
            if c.poll() is None:
                if os.name=='posix':os.killpg(c.pid,signal.SIGTERM)
                else:c.terminate()
        for c in children:
            try:c.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name=='posix':os.killpg(c.pid,signal.SIGKILL)
                else:c.kill()
                c.wait()
        # Parent deadlines/interrupts can prevent child finally blocks executing.
        for item in m['items'][:args.max_files]:
            scratch=out/'temporary'/item['output_id']
            if scratch.exists() and scratch.resolve().is_relative_to((out/'temporary').resolve()):shutil.rmtree(scratch)
        lock.unlink(missing_ok=True)
    if summary['status']!='finished':raise SystemExit(2)

if __name__=='__main__':main()
