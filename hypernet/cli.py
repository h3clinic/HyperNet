"""Command line entry point. Run python -m hypernet --help."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

def configure_runtime(threads=4,seed=17):
    import numpy as np
    np.random.seed(seed)
    try:
        import cv2
        cv2.setNumThreads(threads);cv2.setRNGSeed(seed)
    except ImportError:pass
    if importlib.util.find_spec("torch"):
        import torch
        torch.set_num_threads(threads);torch.manual_seed(seed)

def main(argv=None):
    parser=argparse.ArgumentParser(description="HyperNet single-reference recognition workbench")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor",help="Show available libraries; no weight downloads")
    p=sub.add_parser("prepare",help="Build a labeled synthetic same-source stress test")
    p.add_argument("--sources",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--objects",default=24,type=int);p.add_argument("--size",default=256,type=int);p.add_argument("--seed",default=17,type=int)
    p=sub.add_parser("import-sequences",help="Import local Oxford/HPatches multi-view sequences")
    p.add_argument("--sources",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--size",default=384,type=int);p.add_argument("--seed",default=17,type=int)
    p=sub.add_parser("validate",help="Validate a manifest and image/split hashes")
    p.add_argument("manifest",type=Path)
    p=sub.add_parser("export-fiftyone",help="Export calibrated pairs for optional visual error analysis")
    p.add_argument("--predictions",required=True,type=Path);p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path);p.add_argument("--model",required=True)
    p=sub.add_parser("evaluate",help="Score all models, freeze dev thresholds, report test results")
    p.add_argument("--manifest",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--models",default="orb,sift,partgraph")
    p.add_argument("--weights",type=Path,help="Explicit ResNet18 state_dict checkpoint")
    p.add_argument("--loftr-checkpoint",type=Path)
    p.add_argument("--allow-download",action="store_true",help="Allow official pretrained checkpoint downloads")
    p.add_argument("--checkpoints",type=Path,default=Path("checkpoints"))
    p.add_argument("--tiny-steps",type=int,default=60);p.add_argument("--target-fpr",type=float,default=.05)
    p.add_argument("--bootstrap",type=int,default=500);p.add_argument("--seed",type=int,default=17)
    p.add_argument("--threads",type=int,default=4);p.add_argument("--device",default="cpu")
    p=sub.add_parser("match",help="Inspect one reference/query pair without accuracy claims")
    p.add_argument("--reference",required=True,type=Path);p.add_argument("--query",required=True,type=Path)
    p.add_argument("--model",default="partgraph");p.add_argument("--output",required=True,type=Path)
    p.add_argument("--weights",type=Path);p.add_argument("--loftr-checkpoint",type=Path)
    p.add_argument("--allow-download",action="store_true");p.add_argument("--checkpoints",type=Path,default=Path("checkpoints"))
    p.add_argument("--tiny-steps",type=int,default=60);p.add_argument("--device",default="cpu")
    p=sub.add_parser("match-shape",help="Inspect one category reference/query pair with scratch-only shape methods")
    p.add_argument("--reference",required=True,type=Path);p.add_argument("--query",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--model",default="shape_light_v4",choices=("analytic_patches","tiny_v3","shape_bypass_v3","shape_learned_v4","shape_guarded_v4","shape_light_v4","shape_score_v4"))
    p.add_argument("--threshold",type=float,help="Optional cutoff from separate development data; never estimated from this query")
    p.add_argument("--checkpoints",type=Path,default=Path("checkpoints/shape"))
    p.add_argument("--tiny-steps",type=int,default=120);p.add_argument("--seed",type=int,default=73)
    p.add_argument("--threads",type=int,default=2);p.add_argument("--device",default="cpu")
    p=sub.add_parser("train-reference",help="Train a small CNN using patches of exactly one image")
    p.add_argument("--reference",required=True,type=Path);p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--steps",type=int,default=120);p.add_argument("--seed",type=int,default=17)
    p.add_argument("--threads",type=int,default=4);p.add_argument("--device",default="cpu")
    p=sub.add_parser("prepare-categories",help="Build one-shot category episodes from labeled folders")
    p.add_argument("--sources",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--categories",type=int,default=24);p.add_argument("--dev-categories",type=int)
    p.add_argument("--episodes",type=int,default=3);p.add_argument("--positive-queries",type=int,default=10)
    p.add_argument("--negative-queries",type=int,default=10);p.add_argument("--size",type=int,default=256)
    p.add_argument("--seed",type=int,default=53)
    p=sub.add_parser("validate-categories",help="Validate category identities, labels and support/query separation")
    p.add_argument("manifest",type=Path)
    p=sub.add_parser("plan-coco-categories",help="Select annotated COCO category episodes before fetching images")
    p.add_argument("--annotations",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--categories",type=int,default=24);p.add_argument("--dev-categories",type=int)
    p.add_argument("--episodes",type=int,default=3);p.add_argument("--positive-queries",type=int,default=10)
    p.add_argument("--negative-queries",type=int,default=10);p.add_argument("--seed",type=int,default=53)
    p=sub.add_parser("evaluate-categories",help="Evaluate category presence on separate development/test categories")
    p.add_argument("--manifest",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--models",default="analytic_patches,tiny_v3,shape_bypass_v3,shape_learned_v4,shape_guarded_v4,shape_light_v4,shape_score_v4")
    p.add_argument("--weights",type=Path);p.add_argument("--allow-download",action="store_true")
    p.add_argument("--scratch-only",action="store_true",help="Reject external weights and pretrained category methods before construction")
    p.add_argument("--checkpoints",type=Path,default=Path("checkpoints/category"))
    p.add_argument("--tiny-steps",type=int,default=120);p.add_argument("--target-fpr",type=float,default=.05)
    p.add_argument("--bootstrap",type=int,default=500);p.add_argument("--seed",type=int,default=53)
    p.add_argument("--threads",type=int,default=4);p.add_argument("--device",default="cpu")
    args=parser.parse_args(argv)
    try:
        if args.command=="doctor":
            import importlib.metadata
            result={"python":sys.version,"executable":sys.executable,"libraries":{}}
            for dist in ["numpy","Pillow","opencv-python-headless","torch","torchvision","kornia","pytest"]:
                try:result["libraries"][dist]=importlib.metadata.version(dist)
                except importlib.metadata.PackageNotFoundError:result["libraries"][dist]=None
            print(json.dumps(result,indent=2));return 0
        configure_runtime(getattr(args,"threads",4),getattr(args,"seed",17))
        from hypernet.data import load_rgb,load_manifest
        if args.command=="prepare":
            from hypernet.data import build_synthetic_dataset
            print(build_synthetic_dataset(args.sources,args.output,seed=args.seed,max_objects=args.objects,size=args.size))
        elif args.command=="import-sequences":
            from hypernet.benchmarks import build_sequence_dataset
            print(build_sequence_dataset(args.sources,args.output,size=args.size,seed=args.seed))
        elif args.command=="validate":
            print(json.dumps(load_manifest(args.manifest)["validation"],indent=2))
        elif args.command=="export-fiftyone":
            from hypernet.exports import export_fiftyone_json
            print(export_fiftyone_json(args.predictions,args.manifest,args.output,args.model))
        elif args.command=="prepare-categories":
            from hypernet.category_data import build_category_dataset
            print(build_category_dataset(args.sources,args.output,seed=args.seed,max_categories=args.categories,
                dev_categories=args.dev_categories,episodes_per_category=args.episodes,
                positive_queries=args.positive_queries,negative_queries=args.negative_queries,size=args.size))
        elif args.command=="plan-coco-categories":
            from hypernet.category_coco import plan_coco_category_dataset
            from hypernet.runner import write_json
            if args.output.exists():raise FileExistsError("Choose a new category plan path")
            plan=plan_coco_category_dataset(args.annotations,seed=args.seed,max_categories=args.categories,
                dev_categories=args.dev_categories,episodes_per_category=args.episodes,
                positive_queries=args.positive_queries,negative_queries=args.negative_queries,
                min_support_area=4096)
            args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,plan)
            print(args.output.resolve())
        elif args.command=="validate-categories":
            from hypernet.category_data import load_category_manifest
            print(json.dumps(load_category_manifest(args.manifest)["validation"],indent=2))
        elif args.command=="evaluate-categories":
            from hypernet.category_registry import make_category_models
            from hypernet.category_runner import run_category_evaluation
            from hypernet.category_reporting import make_category_report
            from hypernet.category_data import load_category_manifest
            requested=[s.strip() for s in args.models.split(",")]
            if args.scratch_only and (args.weights is not None or args.allow_download or
                                     any(s.startswith("category_") for s in requested)):
                raise ValueError("Scratch-only evaluation rejects pretrained methods, weights, and downloads")
            models=make_category_models(requested,weights_path=args.weights,
                allow_download=args.allow_download,checkpoint_dir=args.checkpoints,tiny_steps=args.tiny_steps,
                seed=args.seed,device=args.device)
            if args.scratch_only and any(m.provenance().get("external_training_images") != 0 for m in models.values()):
                raise ValueError("Scratch-only evaluation requires zero external training images for every method")
            summary=run_category_evaluation(args.manifest,models,args.output,target_fpr=args.target_fpr,
                bootstrap_samples=args.bootstrap,seed=args.seed)
            with (args.output/"predictions.jsonl").open(encoding="utf-8") as f:
                rows=[json.loads(line) for line in f if line.strip()]
            make_category_report(summary,rows,load_category_manifest(args.manifest),args.output)
            print(f"Category report: {(args.output/'report.html').resolve()}")
        elif args.command=="match-shape":
            import math
            from hypernet.category_registry import make_category_models
            from hypernet.runner import write_json
            if args.output.exists():raise FileExistsError("Choose a new output path")
            if args.threshold is not None and not math.isfinite(args.threshold):
                raise ValueError("Use a finite development cutoff")
            model=make_category_models([args.model],checkpoint_dir=args.checkpoints,
                tiny_steps=args.tiny_steps,seed=args.seed,device=args.device)[args.model]
            provenance=model.provenance()
            if provenance.get("external_training_images")!=0 or provenance.get("external_pretraining",False) not in (False,None):
                raise ValueError("Shape matching requires zero external training")
            prediction=model.predict(load_rgb(args.reference),load_rgb(args.query))
            if not math.isfinite(prediction.score):raise ValueError("Nonfinite model score")
            args.output.parent.mkdir(parents=True,exist_ok=True)
            write_json(args.output,{"prediction":prediction.to_dict(),"provenance":model.provenance(),
                "threshold":args.threshold,"accepted":None if args.threshold is None else prediction.score>=args.threshold,
                "note":"One reference trains the CNN; this query does not. Score is not probability. A cutoff must come from separate development data; one pair does not establish accuracy."})
            print(args.output.resolve())
        elif args.command=="train-reference":
            from hypernet.training import train_reference_cnn
            if args.checkpoint.exists():raise FileExistsError("Choose a new checkpoint path")
            metadata=train_reference_cnn(load_rgb(args.reference),args.checkpoint,steps=args.steps,seed=args.seed,device=args.device)
            print(json.dumps(metadata,indent=2))
        else:
            from hypernet.registry import make_models
            from hypernet.runner import run_evaluation,write_json
            names=args.models.split(",") if args.command=="evaluate" else [args.model]
            models=make_models([s.strip() for s in names],weights_path=args.weights,
                allow_download=args.allow_download,checkpoint_dir=args.checkpoints,tiny_steps=args.tiny_steps,
                seed=getattr(args,"seed",17),device=args.device,loftr_checkpoint=args.loftr_checkpoint)
            if args.command=="evaluate":
                run_evaluation(args.manifest,models,args.output,target_fpr=args.target_fpr,
                               bootstrap_samples=args.bootstrap,seed=args.seed)
                print(f"Report: {(args.output/'report.html').resolve()}")
            else:
                model=models[args.model];prediction=model.predict(load_rgb(args.reference),load_rgb(args.query))
                args.output.parent.mkdir(parents=True,exist_ok=True)
                if args.output.exists():raise FileExistsError("Choose a new output path")
                write_json(args.output,{"prediction":prediction.to_dict(),"provenance":model.provenance(),
                                        "note":"Score is not a probability; acceptance requires development calibration."})
                print(args.output.resolve())
        return 0
    except (ValueError,FileNotFoundError,FileExistsError,ImportError) as exc:
        parser.exit(2,f"Error: {exc}\n")

if __name__=="__main__":raise SystemExit(main())
