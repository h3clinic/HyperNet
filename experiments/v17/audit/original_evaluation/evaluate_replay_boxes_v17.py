"""Immutable DEV replay with common clipping; no model inference or score changes."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prepare_coco_v17 import verify_plan,sha,read,write_new,PLAN
from hypernet.detection_metrics import clip_box,iou,calibration,COMMON_BOX_RULE

def replay(method):
    plan=verify_plan(); rows=plan['dev']; targets=plan['targets']
    old=ROOT/'experiments'/f'{method}_dev'; new=ROOT/'experiments'/f'{method}_dev_v2'
    original=read(old/'result.json')
    if sha(old/'arrays.npz')!=original['score_file_sha256'] or sha(old/'predictions.jsonl')!=original['predictions_sha256']:
        raise ValueError('Original evaluation changed')
    arrays=dict(np.load(old/'arrays.npz'));records=[json.loads(line) for line in (old/'predictions.jsonl').read_text().splitlines()]
    if arrays['image_ids'].tolist()!=[r['image_id'] for r in rows] or arrays['category_ids'].tolist()!=[t['category_id'] for t in targets]: raise ValueError('Original pairing differs')
    if len(records)!=len(rows)*len(targets) or new.exists(): raise ValueError('Unexpected old rows or existing replay')
    new.mkdir();changed=0
    with (new/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for i,row in enumerate(rows):
            for j,target in enumerate(targets):
                record=records[i*len(targets)+j]
                if record['image_id']!=row['image_id'] or record['category_id']!=target['category_id'] or record['score']!=arrays['scores'][i,j]: raise ValueError('Original records not paired')
                raw=record['box'];box=clip_box(raw,row['width'],row['height']);changed+=int(raw!=box)
                record.update(raw_box=raw,box=box,iou=iou(box,row['boxes_by_target'][str(target['category_id'])]))
                arrays['boxes'][i,j]=box;arrays['overlaps'][i,j]=record['iou']
                f.write(json.dumps(record,allow_nan=False)+'\n')
    np.savez_compressed(new/'arrays.npz',**arrays)
    calibrated=calibration(arrays['scores'],arrays['overlaps'],rows,targets)
    previous=read(old/'calibration.json')
    if any(calibrated[k]['cutoff']!=previous[k]['cutoff'] for k in calibrated): raise ValueError('Unexpected cutoff change')
    write_new(new/'calibration.json',calibrated)
    result={**original,'score_file_sha256':sha(new/'arrays.npz'),'predictions_sha256':sha(new/'predictions.jsonl'),
            'summary':{k:v['summary'] for k,v in calibrated.items()},'common_box_rule':COMMON_BOX_RULE,
            'replay':{'inference_rerun':False,'scores_unchanged':True,'boxes_clipped':changed,
                      'source_files':{str((old/name).resolve()):sha(old/name) for name in ('arrays.npz','predictions.jsonl','result.json','calibration.json')},
                      'script_sha256':sha(__file__)},'metrics_sha256':sha(ROOT/'hypernet/detection_metrics.py')}
    write_new(new/'result.json',result)
    print(json.dumps({'method':method,'boxes_clipped':changed,'localized_macro_recall':result['summary']['primary_5pct']['macro_localized_recall'],
                      'localized_hits':result['summary']['primary_5pct']['localized_hits'],'cutoff_unchanged':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',required=True,choices=['owlvit_base_patch32','raw_template'])
    replay(p.parse_args().method)
