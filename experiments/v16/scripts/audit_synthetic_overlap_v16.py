"""Saved-raster hash overlap census; no rendering or learned-model calls."""
from pathlib import Path
import hashlib,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def run(result_path,output):
    if Path(output).exists():raise FileExistsError(output)
    result=read(result_path);run=ROOT/'training/primary_4096';completion=read(run/'completion.json')
    rgb=set();masks=set();bound={str(Path(result_path).resolve()):sha(result_path),str(run/'completion.json'):sha(run/'completion.json')}
    for item in completion['shards']:
        p=run/item['path'];assert sha(p)==item['sha256'];bound[str(p)]=item['sha256']
        with np.load(p,allow_pickle=False)as data:
            rgb.update(row.tobytes().hex()for row in data['rgb_sha256'].reshape(-1,32))
            masks.update(row.tobytes().hex()for row in data['mask_sha256'].reshape(-1,32))
    assert len(rgb)==completion['unique_rgb_views'] and len(masks)==completion['unique_raster_masks']
    records={}
    for field,values in [('reference_rgb',result['input_rgb_hashes']['reference']),('query_rgb',result['input_rgb_hashes']['query']),
                         ('negative_rgb',result['input_rgb_hashes']['negative']),('reference_mask',result['reference_mask_sha256']),
                         ('query_mask',result['query_mask_sha256'])]:
        train=masks if field.endswith('_mask')else rgb
        assert all(len(value)==64 and not(set(value)-set('0123456789abcdef'))for value in values)
        records[field]={'rows':len(values),'unique_rasters':len(set(values)),
                        'rows_exactly_matching_training':sum(value in train for value in values),
                        'unique_rasters_matching_training':len(set(values)&train)}
    assert records['reference_rgb']['rows']==records['reference_mask']['rows']==512
    assert records['query_rgb']['rows']==records['query_mask']['rows']==records['negative_rgb']['rows']==1024
    cross={name:len(set(result['input_rgb_hashes']['reference'])&set(result['input_rgb_hashes'][name]))for name in ['query','negative']}
    mask_cross=len(set(result['reference_mask_sha256'])&set(result['query_mask_sha256']))
    for path,expected in bound.items():assert sha(path)==expected
    out={'format':'hypernet.v16.synthetic_raster_overlap_audit.v1','passed':True,'source_sha256':sha(__file__),
         'training_distinct_rgb':len(rgb),'training_distinct_masks':len(masks),'diagnostic':records,
         'reference_query_exact_rgb_overlap':cross['query'],'reference_negative_exact_rgb_overlap':cross['negative'],
         'reference_query_exact_mask_overlap':mask_cross,'bound_files':bound,'image_or_model_calls':False,
         'interpretation':'Exact SHA256 raster overlap is measured, not perceptual equivalence. Heldout graph signatures remain within the same fixed mathematical grammar. Negative diagnostic signatures may have appeared in training. Identical input streams across training arms are audited separately.'}
    with Path(output).open('x',encoding='utf-8')as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'output':str(output),'sha256':sha(output),'overlap':records}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,default=ROOT/'experiments/synthetic_final_diagnosis/results.json');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.result,args.output)
