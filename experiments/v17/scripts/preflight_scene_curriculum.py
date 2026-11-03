"""Render the exact planned curriculum indices on CPU before training starts."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json,time,sys,traceback
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hypernet.scene_curriculum import episode,config_hash,bbox_from_mask


def check_job(job):
    cv2.setNumThreads(1)
    split,seed,start,end=job;failures=[];families=Counter();programs=set();digest=hashlib.sha256()
    rendered=0;paired=0;min_detail=10**9;min_rgb=10**9;min_visibility=1.;rejected=0;max_retry=0;retry_histogram=Counter();retry_reasons=Counter()
    for index in range(start,end):
        try:
            r=episode(index,split=split,seed=seed)
            rendered+=1
            rejected+=r['metadata']['ambiguous_nuisance_draws_rejected']
            max_retry=max(max_retry,r['metadata']['ambiguous_nuisance_draws_rejected'])
            retry_histogram[str(r['metadata']['ambiguous_nuisance_draws_rejected'])]+=1
            retry_reasons.update(r['metadata']['ambiguous_nuisance_rejection_reasons'])
            assert np.array_equal(r['target_mask']>0,r['query_parts']>0)
            assert np.array_equal(r['support_mask']>0,r['support_parts']>0)
            assert np.array_equal(r['bbox'],bbox_from_mask(r['target_mask']))
            assert np.all(r['target_mask']<=r['amodal_target_mask'])
            assert r['query_foreground_mask'].any() and np.all(r['target_mask']<=r['query_foreground_mask'])
            visible_detail=min(r['metadata'][key] for key in (
                'visible_hard_negative_detail_pixels','visible_absence_detail_pixels',
                'visible_counterfactual_rgb_difference_pixels_lower_bound'))
            assert visible_detail>=8;min_detail=min(min_detail,visible_detail)
            if r['present']:
                assert r['visibility']>=.65;min_visibility=min(min_visibility,r['visibility'])
            else:
                assert not r['target_mask'].any() and not r['bbox'].any()
            families[r['family']]+=1;programs.add(r['program_id'])
            digest.update(index.to_bytes(8,'little'));digest.update(r['support'].tobytes());digest.update(r['query'].tobytes())
            if index%256==0:
                other=episode(index,split=split,seed=seed,positive_probability=0 if r['present'] else 1)
                paired+=1
                difference=np.max(np.abs(r['query'].astype(np.int16)-other['query'].astype(np.int16)),axis=2)>=3
                assert difference.sum()>=r['metadata']['visible_counterfactual_rgb_difference_pixels_lower_bound']>=8
                min_rgb=min(min_rgb,int(difference.sum()))
                assert np.array_equal(r['support'],other['support'])
                assert len(r['metadata']['instances'])==len(other['metadata']['instances'])
                slot=next(v for v in r['metadata']['instances'] if v['role'] in ('target','matched_absence_replacement'))
                x,y,w,h=slot['placement_xywh'];outside=np.ones((192,192),bool);outside[y:y+h,x:x+w]=False
                assert np.array_equal(r['query'][outside],other['query'][outside])
                digest.update(other['query'].tobytes())
        except Exception as exc:
            failures.append({'split':split,'seed':seed,'index':index,'error':str(exc),'traceback':traceback.format_exc()})
    return {'split':split,'seed':seed,'index_start':start,'index_end_exclusive':end,
            'primary_rendered':rendered,'paired_complements_rendered':paired,'failures':failures,
            'rejected_ambiguous_nuisance_draws_primary':rejected,'maximum_retry_count':max_retry,
            'retry_histogram':dict(retry_histogram),'retry_reasons':dict(retry_reasons),
            'family_counts':dict(families),'program_hashes':sorted(programs),'raster_stream_sha256':digest.hexdigest(),
            'minimum_visible_detail_pixels':min_detail,'minimum_measured_pair_rgb_difference_pixels':min_rgb,
            'minimum_target_visibility':min_visibility}


def main(train_count,workers):
    out=ROOT/'assets'/'generator_preflight.json';source_hash=hashlib.sha256((ROOT/'hypernet/scene_curriculum.py').read_bytes()).hexdigest()
    plan=[('train',1701,train_count),('development',1777,256),('test',1777,512),('compositional',1777,512)]
    jobs=[(split,seed,start,min(start+512,count)) for split,seed,count in plan for start in range(0,count,512)]
    start=time.perf_counter();results=[]
    def save(status):
        rendered=sum(r['primary_rendered'] for r in results);paired=sum(r['paired_complements_rendered'] for r in results)
        report={'format':'hypernet.v17.actual_episode_preflight','status':status,'created_utc':datetime.now(timezone.utc).isoformat(),
                'source_sha256':source_hash,'config_sha256':config_hash(),'cpu_workers':workers,'gpu_used':False,
                'training_has_run':False,'optimizer_consumed_images':0,
                'planned_primary_episodes':sum(n for _,_,n in plan),'primary_episodes_rendered':rendered,
                'additional_paired_complements_rendered':paired,'total_actual_episodes_rendered':rendered+paired,
                'rejected_ambiguous_nuisance_draws_primary':sum(r['rejected_ambiguous_nuisance_draws_primary'] for r in results),
                'maximum_retry_count':max((r['maximum_retry_count'] for r in results),default=0),
                'retry_histogram':dict(sum((Counter(r['retry_histogram']) for r in results),Counter())),
                'retry_reasons':dict(sum((Counter(r['retry_reasons']) for r in results),Counter())),
                'elapsed_seconds':time.perf_counter()-start,'plans':[{'split':s,'seed':seed,'index_start':0,'index_end_exclusive':n} for s,seed,n in plan],
                'failures':[f for r in results for f in r['failures']],
                'minimum_visible_detail_pixels':min((r['minimum_visible_detail_pixels'] for r in results),default=None),
                'minimum_measured_pair_rgb_difference_pixels':min((r['minimum_measured_pair_rgb_difference_pixels'] for r in results),default=None),
                'minimum_target_visibility':min((r['minimum_target_visibility'] for r in results),default=None),
                'chunks':[{k:v for k,v in r.items() if k not in ('program_hashes','failures')} for r in sorted(results,key=lambda r:(r['split'],r['index_start']))],
                'distinct_whole_programs':{split:len(set(p for r in results if r['split']==split for p in r['program_hashes']))for split,_,_ in plan}}
        report['episodes_per_second']=(rendered+paired)/max(report['elapsed_seconds'],.001)
        out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        return report
    save('validating')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(check_job,job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result());report=save('validating')
            print(json.dumps({'rendered':report['total_actual_episodes_rendered'],'primary_planned':report['planned_primary_episodes'],
                              'failures':len(report['failures']),'elapsed_seconds':round(report['elapsed_seconds'],2)}),flush=True)
    if hashlib.sha256((ROOT/'hypernet/scene_curriculum.py').read_bytes()).hexdigest()!=source_hash:
        raise RuntimeError('Generator source changed during preflight')
    report=save('failed' if any(r['failures'] for r in results) else 'passed')
    print(json.dumps({k:report[k]for k in ('status','source_sha256','total_actual_episodes_rendered','episodes_per_second','minimum_measured_pair_rgb_difference_pixels')}),flush=True)
    return 1 if report['failures'] else 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train-count',type=int,default=32768);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();raise SystemExit(main(a.train_count,a.workers))
