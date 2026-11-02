"""Enroll one local reference and compare a full query; model weights stay fixed."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def load_rgb(path):
    with Image.open(path) as image:
        if image.width*image.height>16_000_000 or min(image.size)<8:raise ValueError('Use at least 8 pixels per side and at most 16 million pixels')
        if getattr(image,'n_frames',1)!=1:raise ValueError('Use a still photograph')
        return np.array(ImageOps.exif_transpose(image).convert('RGB'),dtype=np.uint8)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--query',type=Path,required=True)
    parser.add_argument('--box',type=float,nargs=4,metavar=('X1','Y1','X2','Y2'),help='Reference box in original pixels; default whole reference')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/detector_micro_aware.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--operating-mode',choices=['transferred','real_dev_calibrated'],default='transferred')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output file; previous comparisons are preserved')
    if not 1<=args.threads<=8:raise ValueError('Use 1–8 CPU threads')
    import torch
    from hypernet.real_photo import RealPhotoDetector,TRANSFERRED_CUTOFFS
    from hypernet.detail_pipeline import state_hash
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    model=RealPhotoDetector(args.checkpoint);before=state_hash(model.model)
    reference,query=load_rgb(args.reference),load_rgb(args.query)
    enrolled=model.enroll(reference,args.box);predictions=model.predict(enrolled,query)
    thresholds=TRANSFERRED_CUTOFFS
    if args.operating_mode=='real_dev_calibrated':
        benchmark=json.loads((ROOT/'benchmarks/v0.13/aggregate_metrics.json').read_text())
        thresholds=benchmark['modes']['real_dev_calibrated']['thresholds']
    for alias,row in predictions.items():row.update(threshold=thresholds[alias],accepted=row['score']>=thresholds[alias])
    after=state_hash(model.model)
    if before!=after or any(p.grad is not None for p in model.model.parameters()):raise RuntimeError('Read-only inference contract violated')
    result={'format':'hypernet.local_photo_comparison.v13','predictions':predictions,'operating_mode':args.operating_mode,
        'additional_real_calibration_photos':96 if args.operating_mode=='real_dev_calibrated' else 0,
        'reference_images':1,'weight_updates':0,'model_state_before':before,'model_state_after':after,
        'provenance':model.provenance(),'reference_preprocessing':enrolled['metadata'],
        'input_file_sha256':{kind:hashlib.sha256(path.read_bytes()).hexdigest() for kind,path in [('reference',args.reference),('query',args.query)]},
        'scope':'Manual comparison, not benchmark evidence. v13 failed real-photo category localization; scores are not probabilities.',
        'input_photos_written_or_uploaded':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'output':str(args.output),'weight_updates':0,'methods':{a:{k:r[k] for k in ['score','threshold','accepted','bbox_xyxy']} for a,r in predictions.items()}},indent=2))
if __name__=='__main__':main()
