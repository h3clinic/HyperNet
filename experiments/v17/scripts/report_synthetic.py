"""Post-hoc illustrations of saved synthetic TEST predictions; no inference."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,html,json,sys,time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.scene_curriculum import episode
from hypernet.detection_metrics import iou

COLORS={'truth':'#4ee3a1','primary':'#ffac55','bag':'#79b8ff'}
FONT=Path('C:/Windows/Fonts/arial.ttf')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def font(size):return ImageFont.truetype(str(FONT),size)
def caption(score,cutoff,overlap):
    return f"score {score:.3f} / cutoff {cutoff:.3f} / IoU {overlap:.3f} / "+('ACCEPT' if score>=cutoff else 'REJECT')


def annotate(rgb,truth,predictions):
    image=Image.fromarray(rgb).resize((384,384),Image.Resampling.NEAREST)
    draw=ImageDraw.Draw(image)
    for key,box in [('truth',truth),*predictions.items()]:
        if box is None:continue
        x1,y1,x2,y2=box
        draw.rectangle((x1*2,y1*2,max(x1*2,x2*2-1),max(y1*2,y2*2-1)),outline=COLORS[key],width=3 if key=='truth' else 2)
    return image


def main():
    cv2.setNumThreads(1);started=time.perf_counter();assets=ROOT/'assets';assets.mkdir(exist_ok=True)
    primary_path=ROOT/'experiments/synthetic_primary/results.json'
    bag_path=ROOT/'experiments/synthetic_bag/results.json'
    raw_path=ROOT/'experiments/synthetic_raw_template/results.json'
    primary=read(primary_path);bag=read(bag_path);raw=read(raw_path)
    sources={path.relative_to(ROOT).as_posix():sha(path) for path in (primary_path,bag_path,raw_path,ROOT/'hypernet/scene_curriculum.py')}
    assert primary['plan']==bag['plan']==raw['plan']
    plan=primary['plan'];assert plan['seed']==1777 and plan['splits']['test']==512
    assert primary['splits']['test']['input_stream_sha256']==bag['splits']['test']['input_stream_sha256']==raw['splits']['test']['input_stream_sha256']
    for result in (primary,bag):
        recorded={key.replace('\\','/'):value for key,value in result['source_sha256'].items()}
        assert recorded['hypernet/scene_curriculum.py']==sources['hypernet/scene_curriculum.py']
    cutoff=primary['splits']['test']['threshold'];bag_cutoff=bag['splits']['test']['threshold']
    rows=sorted(primary['rows']['test'],key=lambda row:row['index']);bag_rows={row['index']:row for row in bag['rows']['test']}
    selectors=[
        ('accepted_localized_hit','Accepted and correctly localized',lambda row:row['present'] and row['score']>=cutoff and row['iou']>=.5,
         'present and primary score >= cutoff and primary IoU >= 0.5'),
        ('good_box_rejected','Correct location rejected by cutoff',lambda row:row['present'] and row['score']<cutoff and row['iou']>=.5,
         'present and primary score < cutoff and primary IoU >= 0.5'),
        ('absent_false_alarm','False alarm on an absent scene',lambda row:not row['present'] and row['score']>=cutoff,
         'absent and primary score >= cutoff'),
        ('positive_bad_top_box','Target present, incorrect top box',lambda row:row['present'] and row['iou']<.5,
         'present and primary IoU < 0.5')]
    with np.load(ROOT/'experiments/synthetic_raw_template/test.npz',allow_pickle=False) as raw_arrays:
        selected=[]
        for kind,title,predicate,rule in selectors:
            candidates=[row for row in rows if predicate(row)]
            if not candidates:raise ValueError('No saved case matches '+kind)
            p=candidates[0];b=bag_rows[p['index']]
            e=episode(p['index'],split='test',seed=1777,support_size=96,query_size=192)
            assert e['program_id']==p['program_id']==b['program_id']
            assert e['present']==p['present']==b['present']
            assert np.array_equal(e['bbox'],p['truth_box']) and np.array_equal(e['bbox'],b['truth_box'])
            assert np.isclose(iou(p['box'],[e['bbox']]) if e['present'] else 0.,p['iou'],atol=1e-6,rtol=0)
            hashes={key:hashlib.sha256(e[key].tobytes()).hexdigest() for key in ('support','query','support_mask','target_mask','query_foreground_mask')}
            for key,digest in hashes.items():
                assert np.frombuffer(bytes.fromhex(digest),np.uint8).tolist()==raw_arrays['input_sha256_'+key][p['index']].tolist()
            prefix='synthetic_case_'+str(p['index']).zfill(3)
            filenames={name:'assets/'+prefix+'_'+name+'.png' for name in ('reference','query','annotated')}
            Image.fromarray(e['support']).save(ROOT/filenames['reference'])
            Image.fromarray(e['query']).save(ROOT/filenames['query'])
            annotated=annotate(e['query'],e['bbox'].tolist() if e['present'] else None,{'primary':p['box'],'bag':b['box']})
            annotated.save(ROOT/filenames['annotated'])
            selected.append({'case':kind,'title':title,'selection_rule':'Lowest TEST episode index satisfying: '+rule,
                'matching_cases_in_saved_test':len(candidates),'index':p['index'],'family':p['family'],
                'program_id':p['program_id'],'present':p['present'],'primary':p,'bag':b,'input_sha256':hashes,
                'hash_verification':'Five exact generated input arrays match raw NPZ; raw/primary/bag share the same TEST input-stream hash.',
                'images':filenames,'_support':e['support'],'_annotated':annotated})
    # Each input pixel is enlarged by exactly 2x, without blur/interpolation.
    canvas=Image.new('RGB',(1520,1236),'#0d1521');draw=ImageDraw.Draw(canvas)
    draw.text((24,18),'HyperNet: where the saved synthetic TEST predictions succeed and fail',font=font(27),fill='#f0f5ff')
    draw.text((24,58),'Post-hoc illustrations, not accuracy estimates. First matching episode per declared case; no new inference.',font=font(18),fill='#b9ccdf')
    x=24
    for name,label in [('truth','Ground truth'),('primary','Primary top box'),('bag','Bag ablation top box')]:
        draw.rectangle((x,96,x+19,110),outline=COLORS[name],width=3);draw.text((x+29,91),label,font=font(18),fill=COLORS[name]);x+=250
    draw.text((774,93),'All top proposals shown, including rejected ones.',font=font(17),fill='#b9ccdf')
    cards=[]
    for number,case in enumerate(selected):
        x=24+(number%2)*752;y=140+(number//2)*536
        draw.rounded_rectangle((x,y,x+728,y+514),radius=10,fill='#172335',outline='#3a4c62',width=1)
        draw.text((x+18,y+15),case['title'],font=font(22),fill='#f1f6ff')
        draw.text((x+18,y+47),f"TEST #{case['index']}  |  {case['family']}  |  target "+('present' if case['present'] else 'absent'),font=font(16),fill='#b9ccdf')
        draw.text((x+18,y+77),'One reference',font=font(17),fill='#dbe7f5')
        canvas.paste(Image.fromarray(case['_support']).resize((192,192),Image.Resampling.NEAREST),(x+18,y+104))
        draw.text((x+326,y+77),'Query: stored top boxes',font=font(17),fill='#dbe7f5')
        canvas.paste(case['_annotated'],(x+326,y+104))
        p=case['primary'];b=case['bag']
        for offset,method,row,threshold,color in [(318,'Primary',p,cutoff,COLORS['primary']),(406,'Bag ablation',b,bag_cutoff,COLORS['bag'])]:
            draw.text((x+18,y+offset),method,font=font(19),fill=color)
            draw.text((x+18,y+offset+25),f"score {row['score']:.3f} / cutoff {threshold:.3f}",font=font(15),fill='#dbe7f5')
            draw.text((x+18,y+offset+47),f"IoU {row['iou']:.3f} / "+('ACCEPT' if row['score']>=threshold else 'REJECT'),font=font(16),fill=color)
        table='<table><tr><th>Saved prediction</th><th>Score</th><th>DEV cutoff</th><th>IoU</th><th>Decision</th></tr>'+''.join(
            '<tr><td>'+label+'</td><td>'+f"{row['score']:.6f}"+'</td><td>'+f'{threshold:.6f}'+'</td><td>'+f"{row['iou']:.6f}"+'</td><td>'+('ACCEPT' if row['score']>=threshold else 'REJECT')+'</td></tr>'
            for label,row,threshold in [('Primary',p,cutoff),('Bag ablation',b,bag_cutoff)])+'</table>'
        cards.append('<section><h2>'+html.escape(case['title'])+'</h2><p>TEST #'+str(case['index'])+' · '+case['family']+' · target '+('present' if case['present'] else 'absent')+'</p><div class="views"><figure><figcaption>One reference, 96 × 96</figcaption><img class="reference" src="'+case['images']['reference']+'"></figure><figure><figcaption>Query, 192 × 192 — recorded boxes overlaid</figcaption><img class="query" src="'+case['images']['annotated']+'"></figure></div>'+table+'<p class="small">'+html.escape(case['selection_rule'])+'. The green box marks the visible target only; absent scenes have no green box.</p></section>')
    draw.text((24,1213),'Synthetic structural programs only. Green = visible target box; orange/blue = saved model proposals. No real-photo result is implied.',font=font(16),fill='#a8bed3')
    montage=assets/'synthetic_diagnosis_montage.png';canvas.save(montage)
    html_doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HyperNet synthetic TEST failure illustrations</title><style>body{font:17px/1.5 system-ui;color:#e9f2ff;background:#0d1521;max-width:1160px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:24px}.lead,p{color:#b9ccdf}.notice{border-left:4px solid #ffac55;padding:8px 16px;background:#172335}.legend{display:flex;gap:28px;flex-wrap:wrap}.views{display:grid;grid-template-columns:240px 1fr;gap:24px}figure{margin:0}figcaption{margin:8px 0}img{image-rendering:pixelated;max-width:100%}.reference{width:192px}.query{width:576px}section{border-top:1px solid #42556e;margin-top:32px;padding-top:10px}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{text-align:left;padding:9px;border-bottom:1px solid #42556e}.small{font-size:14px}a{color:#91c8ff}@media(max-width:650px){.views{grid-template-columns:1fr}table{font-size:13px}th,td{padding:6px}}</style><h1>Where HyperNet succeeds and fails on synthetic TEST scenes</h1><p class="notice">Post-hoc illustrations, not accuracy estimates. These four examples are the first saved TEST episode satisfying each written case definition. They were not chosen as representative samples. No model was rerun, retrained, or tuned for this page.</p><p>Each model saw one reference. Both use the same query and their own development-frozen presence cutoff. All top boxes are displayed, even when rejected by that cutoff. Localization requires IoU ≥ 0.5 as well as acceptance.</p><div class="legend"><span style="color:#4ee3a1">Green: visible ground-truth target</span><span style="color:#ffac55">Orange: primary model</span><span style="color:#79b8ff">Blue: bag ablation</span></div>'''+''.join(cards)+'''<p class="small">Procedurally generated 2D images, seed 1777, canonical TEST split. Program IDs, ground-truth boxes and five input-array hashes were verified against the saved diagnosis inputs. These examples do not estimate real-photo accuracy.</p><p><a href="assets/synthetic_diagnosis_montage.png">Full montage</a> · <a href="assets/synthetic_diagnosis_selection.json">Selections and provenance</a> · <a href="index.html">Experiment dashboard</a></p></html>'''
    (ROOT/'synthetic_diagnosis.html').write_text(html_doc,encoding='utf-8')
    assert all(sha(ROOT/path)==digest for path,digest in sources.items())
    result={'format':'hypernet.synthetic_test_posthoc_illustrations.v17.1','created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Four deterministic post-hoc illustrations, not estimates of model accuracy. No inference or tuning.',
            'plan':plan,'source_sha256':sources,'script_sha256':sha(__file__),'source_files_unchanged':True,
            'canonical_episodes_rerendered':len(selected),'new_episode_indices_generated':0,'gpu_used':False,
            'primary_cutoff':cutoff,'bag_cutoff':bag_cutoff,'input_stream_sha256':primary['splits']['test']['input_stream_sha256'],
            'selections':[{key:value for key,value in case.items() if not key.startswith('_')} for case in selected],
            'montage_sha256':sha(montage),'seconds':time.perf_counter()-started}
    (assets/'synthetic_diagnosis_selection.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','indices':[case['index'] for case in selected],'seconds':result['seconds']}))


if __name__=='__main__':main()
