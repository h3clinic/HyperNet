"""Fixed analytic descriptor scores on the exact old42crop/12self controls.

Privileged old-DEV diagnosis only. No natural fitting, fresh TEST loader, model
changes, generic-photo detection claim, or score-formula search is performed.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse, hashlib, json, sys, time, unittest
ROOT=Path(__file__).resolve().parents[2]
V19=ROOT/'outputs/hypernet_v19'; WORK=ROOT/'work/v19_diagnosis'
sys.path.insert(0,str(V19));sys.path.insert(0,str(V19/'evaluation'))
import numpy as np
import torch
from hypernet_v19.matcher import FrozenDenseMatcher,prepare_image
from diagnose_crop_controls import crop,intersects,measures,CASE_SHA
from data_protocol import load_old,load_reference

DEV_MANIFEST=ROOT/'outputs/hypernet_v18/data/real_dev_only.json'
DEV_MANIFEST_SHA='c773ac5592d0f1dcfe4a61f623077f7e58f22aa4a7a5401587b417cab80ca9cf'
FORMULAS={
 'bidirectional_max_cosine':'0.5*(mean_i max_j C_ij + mean_j max_i C_ij)',
 'foreground_mutual_geometry':'sum_i s_i*a_i*(1-r_i)/sum_i s_i; a_i=sum_j M_ij*q_j',
 'definitions':{'C':'Final descriptor cosine matrix',
    'M':'Existing row-softmax times column-softmax including the model frozen temperature and dustbin',
    's_q':'Existing predicted support/query foreground probabilities; no oracle mask',
    'r':'Existing per-reference-token normalized pair-distance geometry_residual',
    'geometry_penalty_coefficient':1.0,'zero_denominator_guard':1e-8},
 'selection':'Both formulas specified before inference; no natural fitting or hyperparameter sweep'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write_new(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def analytic_scores(diagnostics,support_foreground,query_foreground):
    corr=diagnostics['correlation'];mutual=diagnostics['mutual'];residual=diagnostics['geometry_residual']
    maximum=.5*(corr.amax(-1).mean(-1)+corr.amax(-2).mean(-1))
    s=support_foreground.expand(-1,corr.shape[1],-1)
    a=(mutual*query_foreground[...,None,:]).sum(-1)
    geometry=(s*a*(1-residual)).sum(-1)/s.sum(-1).clamp_min(1e-8)
    return {'bidirectional_max_cosine':maximum,'foreground_mutual_geometry':geometry}


def load_cases():
    if sha(DEV_MANIFEST)!=DEV_MANIFEST_SHA:raise ValueError('Pinned old-DEV-only manifest changed')
    plan=read(DEV_MANIFEST)
    if 'test'in plan:raise ValueError('This diagnostic requires a DEV-only manifest')
    targets=plan['targets'];rows=plan['dev'][:48]
    refmap={row['category_id']:load_reference(row)for row in plan['references']}
    refs=[refmap[target['category_id']]for target in targets]
    patches=[];cases=[]
    for row in rows:
        rgb=load_old(row)
        for j,target in enumerate(targets):
            key=str(target['category_id']);truth=row['boxes_by_target'][key]
            if not row['valid_by_target'][key]or not truth:continue
            gt=max(truth,key=lambda b:(b[2]-b[0])*(b[3]-b[1]));patch,box=crop(rgb,gt)
            eligible=[k for k,t in enumerate(targets)if k!=j and row['valid_by_target'][str(t['category_id'])]
                      and all(not intersects(box,b)for b in row['boxes_by_target'][str(t['category_id'])])]
            cases.append({'image_id':row['image_id'],'target_index':j,'category_id':target['category_id'],
                          'box':box,'negative_indices':eligible});patches.append(patch)
    actual=hashlib.sha256(json.dumps(cases,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if len(cases)!=42 or actual!=CASE_SHA:raise ValueError('Oracle cases/eligible references differ from existing42-case audit')
    return targets,refs,cases,patches


@torch.inference_mode()
def score_images(matcher,banks,images):
    support={key:torch.cat([bank['bank'][key]for bank in banks],0)for key in banks[0]['bank']}
    result={mode+'_'+score:[]for mode in('pipeline','symmetric')
            for score in('learned_head','bidirectional_max_cosine','foreground_mutual_geometry')}
    for rgb in images:
        canvas,meta=prepare_image(rgb,max_side=matcher.max_side)
        tensor=torch.from_numpy(canvas.copy()).permute(2,0,1)[None].to(matcher.device,dtype=torch.float32)/255
        pyramid=matcher.model.encode_query(tensor)
        box=torch.tensor([[[0.,0.,float(meta['resized_width']),float(meta['resized_height'])]]],device=matcher.device)
        pipeline=matcher.model.sample_query_regions(pyramid,box)
        symmetric=matcher.enroll(rgb)['bank']
        for mode,region in(('pipeline',pipeline),('symmetric',symmetric)):
            region={key:value.expand(len(banks),*value.shape[1:])for key,value in region.items()}
            learned,diagnostics=matcher.model.score_regions(support,region,diagnostics=True)
            scores={'learned_head':learned,**analytic_scores(diagnostics,support['foreground'],region['foreground'])}
            for name,value in scores.items():result[mode+'_'+name].append(value[:,0].float().cpu().numpy())
    return {key:np.stack(value)for key,value in result.items()}


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    destination=args.output.resolve()
    if not destination.is_relative_to(WORK.resolve())or destination==WORK.resolve():raise ValueError('Output must be a new subdirectory of work/v19_diagnosis')
    if destination.exists():raise FileExistsError('Preserve completed/existing analytic diagnosis')
    checkpoints={'trained':args.trained_checkpoint.resolve(),'random':args.random_checkpoint.resolve()}
    sources=[Path(__file__),V19/'hypernet_v19/model.py',V19/'hypernet_v19/matcher.py',V19/'hypernet_v19/proposals.py',
             V19/'evaluation/diagnose_crop_controls.py',V19/'evaluation/diagnose_old_dev.py',V19/'evaluation/data_protocol.py']
    bindings={str(path.relative_to(ROOT)).replace('\\','/'):sha(path)for path in sources}
    # This function opens only the pinned DEV-only manifest and hash-verified
    # original reference/old-DEV JPEGs. No v19 mixed-cohort plan is opened.
    targets,refs,cases,patches=load_cases()
    self_cases=[{'target_index':j,'negative_indices':[k for k in range(len(targets))if k!=j]}for j in range(len(targets))]
    admission={'format':'hypernet.v19.analytic_crop_diagnosis.v1','scope':'Privileged oldDEV crops and exact-reference-self controls, not deployed detection or fresh validation',
      'cases_sha256':CASE_SHA,'old_dev_manifest_sha256':DEV_MANIFEST_SHA,'targets':targets,'cases':cases,
      'formulas':FORMULAS,'encoding_modes':{'pipeline':'Original reference bank versus full oracle crop using standard query pyramid',
        'symmetric':'Both sides use the same predicted-extent support encoder; representation diagnostic only'},
      'crop_rgb_sha256':[hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()for rgb in patches],
      'reference_rgb_sha256':[hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()for rgb in refs],
      'max_side':args.max_side,'device':args.device,'source_sha256':bindings,
      'checkpoint_sha256':{name:sha(path)for name,path in checkpoints.items()},'natural_weight_updates':0,
      'fresh_test_pixels_or_records_opened':0,'natural_fitted_parameters':0}
    destination.mkdir(parents=True);write_new(destination/'admission.json',admission)
    overall={};started=time.monotonic()
    for name,path in checkpoints.items():
        matcher=FrozenDenseMatcher(path,device=args.device,max_side=args.max_side)
        updates=matcher.saved['training']['optimizer_steps']
        if(name=='trained'and updates<1)or(name=='random'and updates!=0):raise ValueError('Wrong trained/random checkpoint role')
        banks=[matcher.enroll(ref)for ref in refs];start=time.monotonic()
        crop_scores=score_images(matcher,banks,patches);self_scores=score_images(matcher,banks,refs)
        if not all(np.isfinite(value).all()for value in[*crop_scores.values(),*self_scores.values()]):raise ValueError('Nonfinite analytic score')
        matcher.assert_unchanged()
        if any(sha(ROOT/source)!=expected for source,expected in bindings.items()):raise ValueError('Diagnostic source changed')
        method_dir=destination/name;method_dir.mkdir()
        np.savez_compressed(method_dir/'scores.npz',**{'crop_'+k:v for k,v in crop_scores.items()},**{'self_'+k:v for k,v in self_scores.items()})
        result={'provenance':matcher.provenance(),'oracle_crops':{key:measures(value,cases)for key,value in crop_scores.items()},
          'reference_self':{key:measures(value,self_cases)for key,value in self_scores.items()},'seconds':time.monotonic()-start,
          'scores_sha256':sha(method_dir/'scores.npz'),'cases_sha256':CASE_SHA,'formulas':FORMULAS,
          'learned_head_is_control_not_refitted':True,'natural_fitted_parameters':0,'fresh_test_pixels_or_records_opened':0}
        write_new(method_dir/'result.json',result);overall[name]=result
        print(json.dumps({'method':name,'seconds':result['seconds'],
            'oracle_correct_top1':{key:value['correct_top1']for key,value in result['oracle_crops'].items()},
            'self_correct_top1':{key:value['correct_top1']for key,value in result['reference_self'].items()}},indent=2),flush=True)
        del matcher,banks
        if args.device=='cuda':torch.cuda.empty_cache()
    write_new(destination/'comparison.json',{'admission_sha256':sha(destination/'admission.json'),'seconds':time.monotonic()-started,
        'methods':overall,'no_model_selection_claim':True,'no_full_photo_or_fresh_test_result':True})


def self_test():
    class Tests(unittest.TestCase):
        def test_exact_formula_and_geometry_penalty(self):
            corr=torch.eye(4)[None,None];fg=torch.ones(1,1,4)
            diag={'correlation':corr,'mutual':corr,'geometry_residual':torch.zeros(1,1,4)}
            score=analytic_scores(diag,fg,fg)
            self.assertEqual(float(score['bidirectional_max_cosine']),1.)
            self.assertEqual(float(score['foreground_mutual_geometry']),1.)
            diag['geometry_residual'].fill_(.25)
            self.assertEqual(float(analytic_scores(diag,fg,fg)['foreground_mutual_geometry']),.75)
        def test_bidirectional_max_is_symmetric(self):
            corr=torch.tensor([[[[.1,.8,.2],[.3,.4,.9]]]])
            def call(c):return analytic_scores({'correlation':c,'mutual':torch.ones_like(c),
                'geometry_residual':torch.zeros(*c.shape[:-1])},torch.ones(*c.shape[:-1]),
                torch.ones(*c.shape[:-2],c.shape[-1]))['bidirectional_max_cosine']
            self.assertTrue(torch.equal(call(corr),call(corr.transpose(-1,-2))))
        def test_foreground_weights_and_empty_guard(self):
            corr=torch.eye(2)[None,None];diag={'correlation':corr,'mutual':corr,'geometry_residual':torch.zeros(1,1,2)}
            self.assertEqual(float(analytic_scores(diag,torch.zeros(1,1,2),torch.ones(1,1,2))['foreground_mutual_geometry']),0.)
            value=analytic_scores(diag,torch.ones(1,1,2),torch.tensor([[[1.,0.]]]))['foreground_mutual_geometry']
            self.assertEqual(float(value),.5)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful()else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--trained-checkpoint',type=Path)
    parser.add_argument('--random-checkpoint',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');parser.add_argument('--max-side',choices=(384,640),type=int,default=384)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:sys.exit(self_test())
    if not args.trained_checkpoint or not args.random_checkpoint or not args.output:parser.error('Both checkpoints and a new output directory are required')
    run(args)
