"""Score a query using one reference and HyperNet's own mathematical weights.

Pattern mode is whole-image pattern retrieval, not a semantic bounding-box
detector. Anomaly mode treats the reference as normal and measures deviations.
Scores are not probabilities. A supplied threshold must be calibrated for the
intended domain; this command never fits on query images or downloads weights.
"""
from pathlib import Path
import argparse,json,math,os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image,ImageOps
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hypernet.detail_features import FrozenDetails
from hypernet.pattern_matcher import PatternMatcher,RECIPES
from hypernet.grid_anomaly import ReferenceBank


def load(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--query',type=Path,required=True)
    parser.add_argument('--mode',choices=['pattern','anomaly'],default='pattern')
    parser.add_argument('--recipe',choices=list(RECIPES),default='translation_006')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--threshold',type=float)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.threshold is not None and not math.isfinite(args.threshold):parser.error('Threshold must be finite')
    if args.output and args.output.exists():parser.error('Output already exists; choose a new file')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    features=FrozenDetails(args.checkpoint,device=args.device,batch_size=32)
    before=features.provenance();reference=load(args.reference);query=load(args.query)
    if args.mode=='pattern':
        matcher=PatternMatcher(features,recipe=args.recipe)
        bank=matcher.enroll(reference);prediction=matcher.predict(bank,query)
        scope='Whole-image pattern similarity; no bounding-box or semantic-category accuracy claim.'
    else:
        bank=ReferenceBank(features.encode(reference,sampling_stride=2),device=args.device)
        score=bank.score(features.encode(query,sampling_stride=2))
        prediction={key:value for key,value in score.items() if key not in ('patch_scores','xy')}
        scope='One normal reference anomaly score; larger means less similar to the reference texture.'
    after=features.provenance()
    if before!=after:raise RuntimeError('Frozen model changed during inference')
    result={'mode':args.mode,'scope':scope,'score':prediction['score'],'prediction':prediction,
            'threshold':args.threshold,'alarm':None if args.threshold is None else prediction['score']>=args.threshold,
            'score_is_probability':False,'reference_images':1,'real_weight_updates':0,
            'provenance':before,'model_state_unchanged':True,
            'threshold_note':'No alarm decision without a supplied domain-calibrated threshold.'}
    rendered=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:stream.write(rendered+'\n')
    print(rendered)


if __name__=='__main__':main()
