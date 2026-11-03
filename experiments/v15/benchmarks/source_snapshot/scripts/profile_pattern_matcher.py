"""Timing/equivalence on an already enrolled reference, not evaluation queries."""
from pathlib import Path
import os,sys,time,json,hashlib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from hypernet.detail_features import FrozenDetails
from hypernet.pattern_matcher import PatternMatcher,RECIPES,raw_pixel_vector
from hypernet.gravity_spy_images import GravitySpyCohort
from pattern_matcher_scalar_reference import PatternMatcher as Scalar


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cohort=GravitySpyCohort(ROOT/'data/gravity_spy_v15_final')
    extractor=FrozenDetails(ROOT/'checkpoints/details_run01/families_0262144.pt',batch_size=64,device='cuda')
    before=extractor.provenance();vector=PatternMatcher(extractor);scalar=Scalar(extractor)
    refs=[cohort.reference(label)for label in cohort.target_labels];banks=[vector.enroll(rgb)for rgb in refs]
    event_id=cohort.plan['reference_by_label'][cohort.target_labels[0]]
    encoded=extractor.encode(refs[0]);equivalence={}
    for kind in('learned','analytic'):
        expected=scalar.score_many(banks,encoded,recipes=RECIPES,feature_kind=kind)
        actual=vector.score_many(banks,encoded,recipes=RECIPES,feature_kind=kind)
        differences=[];evidence_exact=True
        for a,b in zip(actual,expected):
            for recipe in RECIPES:
                differences.append(abs(a[recipe]['score']-b[recipe]['score']))
                evidence_exact&=a[recipe]['evidence']==b[recipe]['evidence']
        assert max(differences)<2e-6 and evidence_exact
        equivalence[kind]={'max_score_error':max(differences),'all_discrete_evidence_and_translation_exact':evidence_exact}
    timings=[]
    for iteration in range(8):
        start=time.perf_counter();rgb=cohort.load_rgb(event_id);decoded=time.perf_counter()
        features=extractor.encode(rgb);encoded_at=time.perf_counter()
        vector.score_many(banks,features,recipes=RECIPES,feature_kind='learned');matched=time.perf_counter()
        vector.score_many(banks,features,recipes=RECIPES,feature_kind='analytic');analytic=time.perf_counter()
        raw_pixel_vector(rgb);raw=time.perf_counter()
        timings.append({'decode_hash':decoded-start,'encode':encoded_at-decoded,'learned_matching':matched-encoded_at,
            'analytic_matching':analytic-matched,'raw_vector':raw-analytic,'total':raw-start})
    assert extractor.provenance()==before
    result={'scope':'Performance only: first enrolled reference reused as query. Zero CALIB or TEST query reads; no task-accuracy inference.',
        'reference_event_id':event_id,'equivalence':equivalence,'timings':timings,
        'median_after_first':{key:float(np.median([r[key]for r in timings[1:]]))for key in timings[0]},
        'checkpoint':before,'matcher_sha256':hashlib.sha256((ROOT/'hypernet/pattern_matcher.py').read_bytes()).hexdigest(),
        'scalar_source_sha256':hashlib.sha256((ROOT/'tests/pattern_matcher_scalar_reference.py').read_bytes()).hexdigest(),
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'model_unchanged':True}
    (ROOT/'experiments/pattern_matcher_vectorization_profile.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items()if k not in('timings','checkpoint')},indent=2))


if __name__=='__main__':main()
