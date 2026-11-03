"""Generate a real procedural preview and bounded QA; never a training counter."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time,sys
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hypernet.scene_curriculum import episode,FAMILIES,CONFIG,config_hash,program_split,bbox_from_mask


def digest(value):return hashlib.sha256(value).hexdigest()
def image_hash(array):return digest(array.tobytes())


def main(qa_per_split):
    assets=ROOT/'assets';assets.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    canvas=Image.new('RGB',(640,64+len(FAMILIES)*242),(15,24,36));draw=ImageDraw.Draw(canvas)
    draw.text((16,12),'Actual 2D generator preview: one reference, positive scene, absent scene',font=font,fill='white')
    draw.text((16,36),'Generated examples only. No model predictions or training-consumption claim.',font=small,fill=(170,190,210))
    selected={}
    for i in range(12):
        row=episode(i,positive_probability=1)
        if row['family'] not in selected:selected[row['family']]=(i,row)
    cards=[];preview_records=[]
    for j,family in enumerate(FAMILIES):
        index,positive=selected[family];negative=episode(index,positive_probability=0)
        prefix='program_'+str(j).zfill(2)
        paths={}
        for name,rgb in [('reference',positive['support']),('positive',positive['query']),('absent',negative['query'])]:
            filename=prefix+'_'+name+'.png';Image.fromarray(rgb).save(assets/filename);paths[name]='assets/'+filename
        y=64+j*242
        draw.text((16,y),family+' | structural program '+positive['program_id'][:12],font=font,fill=(226,235,244))
        for k,(name,rgb) in enumerate([('Reference',positive['support']),('Positive scene',positive['query']),('Absent / changed relationship',negative['query'])]):
            img=Image.fromarray(rgb)
            if name=='Reference':img=img.resize((192,192),Image.Resampling.NEAREST)
            if name=='Positive scene':
                painter=ImageDraw.Draw(img);x1,y1,x2,y2=positive['bbox'].tolist();painter.rectangle([x1,y1,x2-1,y2-1],outline=(60,255,180),width=1)
            canvas.paste(img,(16+k*208,y+27));draw.text((16+k*208,y+222),name,font=small,fill=(180,205,225))
        cards.append('<section><h2>'+family+'</h2><p>One whole-program type; independent support and target geometry. The absent scene changes one structural attribute in the target slot and keeps all outside-slot pixels identical.</p><div class="views">'+''.join('<figure><figcaption>'+name+'</figcaption><img src="'+paths[key]+'"></figure>' for key,name in [('reference','One reference'),('positive','Positive query'),('absent','Absent query')])+'</div><p>Program '+positive['program_id'][:16]+' · edit: '+negative['metadata']['absence_replacement_edit']+' · positive visibility '+str(round(positive['visibility']*100,1))+'%</p></section>')
        preview_records.append({'family':family,'episode_index':index,'program_id':positive['program_id'],'images':paths,
                                'support_rgb_sha256':image_hash(positive['support']),'positive_rgb_sha256':image_hash(positive['query']),
                                'negative_rgb_sha256':image_hash(negative['query']),'bbox':positive['bbox'].tolist(),
                                'positive_visibility':positive['visibility'],'support_frame':positive['metadata']['support_frame']})
    canvas.save(assets/'curriculum_preview.png')
    html='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HyperNet v17 · 2D curriculum</title><style>body{font:16px/1.5 system-ui;background:#0f1824;color:#e4edf6;max-width:1300px;margin:auto;padding:24px}section{margin:28px 0;border-top:1px solid #40556b;padding-top:12px}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}figure{margin:0}img{width:100%;image-rendering:pixelated}p{color:#adc3d7}a{color:#91d6ee}@media(max-width:700px){.views{grid-template-columns:1fr}}</style><h1>Six mathematical 2D construction families</h1><p>Actual generated preview, not a training run or recognition result. Shapes use continuous curves/fields, independently varied geometry, local texture, clutter, and both contrast polarities. The support is cropped with varied margins and aspect-preserving letterboxing. No 3D renderer, natural image, teacher or borrowed weight is used.</p><p>Pixel masks and visible boxes supervise synthetic training. No ground-truth query box is given at real inference. The separate composition stress split reserves new pair constructions; components themselves can be familiar. <a href="index.html">Live experiment</a></p>'''+''.join(cards)+'</html>'
    (ROOT/'gallery.html').write_text(html,encoding='utf-8')
    start=time.perf_counter();counts=Counter();families=Counter();programs={};support_hashes=set();query_hashes=set();min_visibility=1.;min_detail=100000;min_rgb=100000;min_pair_rgb=100000;pair_count=0;frame_margins=Counter();polarity=Counter();pixels=0
    for split in ('train','development','test','compositional'):
        programs[split]=set()
        for i in range(qa_per_split):
            r=episode(i,split=split)
            assert program_split(r['program'])==split
            assert np.array_equal(r['target_mask']>0,r['query_parts']>0)
            assert np.array_equal(r['support_mask']>0,r['support_parts']>0)
            assert np.array_equal(r['bbox'],bbox_from_mask(r['target_mask']))
            assert np.all(r['target_mask']<=r['amodal_target_mask'])
            assert r['metadata']['hard_negative_changed_pixels']>=8 and r['metadata']['absence_replacement_changed_pixels']>=8
            assert r['query_foreground_mask'].any() and np.all(r['target_mask']<=r['query_foreground_mask'])
            assert r['metadata']['visible_hard_negative_detail_pixels']>=8
            assert r['metadata']['visible_absence_detail_pixels']>=8
            assert r['metadata']['visible_counterfactual_rgb_difference_pixels_lower_bound']>=8
            min_rgb=min(min_rgb,r['metadata']['visible_counterfactual_rgb_difference_pixels_lower_bound'])
            if i%16==0:
                other=episode(i,split=split,positive_probability=0 if r['present'] else 1)
                changed=np.max(np.abs(r['query'].astype(np.int16)-other['query'].astype(np.int16)),axis=2)>=3
                assert changed.sum()>=r['metadata']['visible_counterfactual_rgb_difference_pixels_lower_bound']>=8
                assert np.array_equal(r['support'],other['support'])
                assert len(r['metadata']['instances'])==len(other['metadata']['instances'])
                slot=next(v for v in r['metadata']['instances'] if v['role'] in ('target','matched_absence_replacement'))
                x,y,w,h=slot['placement_xywh'];outside=np.ones((192,192),bool);outside[y:y+h,x:x+w]=False
                assert np.array_equal(r['query'][outside],other['query'][outside])
                pair_count+=1;min_pair_rgb=min(min_pair_rgb,int(changed.sum()))
                support_hashes.add(image_hash(other['support']));query_hashes.add(image_hash(other['query']))
                pixels+=other['query'].shape[0]*other['query'].shape[1]+other['support'].shape[0]*other['support'].shape[1]
            if r['present']:
                assert r['visibility']>=.65;min_visibility=min(min_visibility,r['visibility'])
                assert sum(x['role']=='target' for x in r['metadata']['instances'])==1
            else:
                assert not r['target_mask'].any() and not r['query_parts'].any() and not r['bbox'].any()
                assert all(x['program_id']!=r['program_id'] for x in r['metadata']['instances'])
            counts[split]+=1;counts[split+'_present']+=int(r['present']);families[split+':'+r['family']]+=1
            programs[split].add(r['program_id']);support_hashes.add(image_hash(r['support']));query_hashes.add(image_hash(r['query']))
            frame_margins[str(r['metadata']['support_frame']['margin_fraction'])]+=1
            min_detail=min(min_detail,r['metadata']['hard_negative_changed_pixels'],r['metadata']['absence_replacement_changed_pixels'])
            mean_fg=r['support'][r['support_mask']>0].mean();mean_bg=r['support'][r['support_mask']==0].mean()
            polarity['brighter' if mean_fg>=mean_bg else 'darker']+=1
            pixels+=r['query'].shape[0]*r['query'].shape[1]+r['support'].shape[0]*r['support'].shape[1]
    for a in programs:
        for b in programs:
            if a!=b:assert not programs[a]&programs[b]
    elapsed=time.perf_counter()-start
    summary={'format':'hypernet.v17.generator_preview_and_bounded_qa','created_utc':datetime.now(timezone.utc).isoformat(),
             'status':'passed','training_has_run':False,'optimizer_consumed_images':0,'natural_images_used':0,
             'config':CONFIG,'config_sha256':config_hash(),
             'source_sha256':digest((ROOT/'hypernet/scene_curriculum.py').read_bytes()),
             'qa_episodes':4*qa_per_split+pair_count,'qa_primary_episodes':4*qa_per_split,
             'qa_additional_counterfactual_episodes':pair_count,'qa_final_rgb_pairs_checked':pair_count,
             'qa_seconds':elapsed,'qa_episodes_per_second':(4*qa_per_split+pair_count)/elapsed,
             'counts_scope':'Family, split, framing and polarity counts below describe primary episodes; unique raster hashes include paired complements.',
             'counts':dict(counts),'family_counts':dict(families),'distinct_whole_programs_by_split':{k:len(v)for k,v in programs.items()},
             'unique_support_rgb_in_qa':len(support_hashes),'unique_query_rgb_in_qa':len(query_hashes),
             'qa_generated_pixels':pixels,'minimum_visible_target_fraction':min_visibility,'minimum_counterfactual_changed_pixels':min_detail,
             'minimum_surviving_counterfactual_rgb_difference_pixels_lower_bound':min_rgb,
             'minimum_measured_final_pair_rgb_difference_pixels':min_pair_rgb,
             'support_margin_counts':dict(frame_margins),'observed_support_contrast_polarity':dict(polarity),
             'preview_actual_positive_and_absent_episodes':12,'preview_programs':preview_records,
             'checks':['binary occupancy agrees with visible part map','visible masks contained in amodal target masks',
                       'tight half-open visible boxes reproduce','absent target labels empty','target visibility floor',
                       'hard-negative detail survives final sampling and occlusion','at least 8 RGB-distinguishing pixels survive occlusion',
                       'measured final RGB presence counterfactuals differ inside the slot and match exactly outside it',
                       'query foreground masks retain all scene objects even when target absent','whole-program hashes disjoint across all four splits'],
             'limits':['Finite bounded grammar; not unlimited real semantics.','Primitive subprograms may recur across whole-program hash splits.',
                       'Part correspondence across instances is not pixel correspondence.','This bounded QA is not an audit of future optimizer consumption.',
                       'Display positive-box outlines are annotations, not input pixels.']}
    preflight_path=assets/'generator_preflight.json'
    if preflight_path.exists():
        preflight=json.loads(preflight_path.read_text(encoding='utf-8-sig'))
        source_matches=preflight['source_sha256']==summary['source_sha256']
        summary['full_preflight_artifact']='assets/generator_preflight.json'
        summary['full_preflight_source_matches']=source_matches
        summary['full_preflight_status']=preflight['status']
        summary['full_preflight_actual_episodes']=preflight['total_actual_episodes_rendered']
        summary['full_preflight_failure_count']=len(preflight['failures'])
        summary['full_preflight_rejected_nuisance_draws']=preflight.get('rejected_ambiguous_nuisance_draws_primary',0)
        summary['full_preflight_maximum_retry_count']=preflight.get('maximum_retry_count',0)
        summary['status']='passed' if preflight['status']=='passed' and source_matches else 'validating'
        summary['superseded_qa_artifacts']=['assets/qa_superseded_57a5d733.json',
            'assets/preflight_failed_50a8d3c0.json','assets/preflight_failed_66b8a01d.json',
            'assets/preflight_aborted_partial_material_revision.json']
    (ROOT/'generation_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('status','qa_episodes','qa_seconds','qa_episodes_per_second','minimum_visible_target_fraction','source_sha256')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--qa-per-split',type=int,default=96);a=p.parse_args();main(a.qa_per_split)
