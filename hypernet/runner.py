"""Evaluation orchestrator with per-pair evidence and development-only calibration."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
import numpy as np
from hypernet.data import load_manifest, load_rgb
from hypernet.metrics import calibrate_threshold, evaluate_scores, box_iou

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")

def homography_corner_error(predicted, truth, shape):
    """Mean corner transfer discrepancy in query pixels; not camera-pose error."""
    if predicted is None:
        return None
    h,w=shape[:2]
    points=np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]],float)
    projected=[]
    for H in (predicted,truth):
        H=np.asarray(H,float)
        if H.shape!=(3,3) or not np.isfinite(H).all() or np.max(np.abs(H))==0:
            return None
        H=H/np.max(np.abs(H))
        a=points@H.T
        if np.any(np.abs(a[:,2])<1e-12):
            return None
        projected.append(a[:,:2]/a[:,2:])
    error=float(np.linalg.norm(projected[0]-projected[1],axis=1).mean())
    return error if np.isfinite(error) else None

def run_evaluation(manifest_path, models, output_dir, target_fpr=.05, bootstrap_samples=500, seed=17):
    manifest_path=Path(manifest_path).resolve()
    output=Path(output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/"summary.json").exists() or (output/"predictions.jsonl").exists():
        raise FileExistsError("Use a fresh output folder to preserve previous experiment evidence")
    manifest=load_manifest(manifest_path)
    root=Path(manifest["root"])
    pairs=manifest["pairs"]
    if not any(p["split"]=="dev" for p in pairs) or not any(p["split"]=="test" for p in pairs):
        raise ValueError("Both development and test identities are required")
    images={}
    def image(relative):
        if relative not in images:images[relative]=load_rgb(root/relative)
        return images[relative]
    versions={}
    for package in ["numpy","Pillow","opencv-python-headless","torch","torchvision","kornia","hypernet-starter"]:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    summary={"completed":False,"created_utc":datetime.now(timezone.utc).isoformat(),"manifest":str(manifest_path),
             "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
             "protocol":manifest.get("protocol"),"dataset_notes":manifest.get("notes",manifest.get("description")),
             "dataset_limitations":manifest.get("limitations",[]),"identity_definition":manifest.get("identity_definition"),
             "validation":manifest["validation"],"versions":versions,
             "platform":platform.platform(),"seed":seed,"target_dev_fpr":target_fpr,"models":{},
             "interpretation":["One reference image per identity does not imply zero external pretraining.",
                               "Synthetic same-source tests do not establish generalization to independent object views.",
                               "Test FPR is measured; a development FPR target is not a test guarantee.",
                               "Pair latencies include reused feature caches, exclude file IO/model loading and explicit adaptation; not a hardware-normalized benchmark."]}
    all_rows=[]
    with (output/"predictions.jsonl").open("w",encoding="utf-8") as stream:
        for alias,model in models.items():
            print(f"MODEL {alias}: {len(pairs)} pairs",flush=True)
            started=time.perf_counter()
            if hasattr(model,"prepare"):
                for reference in sorted({p["reference"] for p in pairs}):
                    model.prepare(image(reference))
            adaptation_seconds=time.perf_counter()-started
            rows=[]
            for index,pair in enumerate(pairs):
                ref,query=image(pair["reference"]),image(pair["query"])
                begin=time.perf_counter()
                result=model.predict(ref,query)
                elapsed=time.perf_counter()-begin
                if not np.isfinite(result.score):
                    raise ValueError(f"{alias}: nonfinite score on {pair['id']}")
                row={**pair,"model":alias,"score":float(result.score),"latency_ms":elapsed*1000,
                     "prediction":result.to_dict()}
                if pair.get("H_ref_to_query") is not None:
                    row["corner_error_px"]=homography_corner_error(result.homography,pair["H_ref_to_query"],ref.shape)
                if pair.get("target_bbox") is not None:
                    row["bbox_iou"]=box_iou(result.bbox_xyxy,pair["target_bbox"]) if result.bbox_xyxy is not None else 0.
                row=json_safe(row)
                stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush()
                rows.append(row)
                if (index+1)%50==0:print(f"  {index+1}/{len(pairs)}",flush=True)
            dev=[r for r in rows if r["split"]=="dev"]
            if getattr(model,"calibration_mode",None)=="route_budget":
                from hypernet.calibration import calibrate_route_thresholds,apply_route_thresholds
                calibration=calibrate_route_thresholds(dev,target_fpr)
                rows=apply_route_thresholds(rows,calibration)
            else:
                calibration=calibrate_threshold(dev,target_fpr)
            test=[r for r in rows if r["split"]=="test"]
            threshold=calibration["threshold"]
            for row in rows:row["accepted"]=row["score"]>=threshold
            metrics=evaluate_scores(test,threshold,bootstrap_samples=bootstrap_samples,seed=seed)
            gt_geometry=[r for r in test if r["present"] and "H_ref_to_query" in r]
            geometry={"positive_pairs_with_ground_truth":len(gt_geometry)}
            for cutoff in [3,5,10]:
                geometry[f"fraction_corner_error_le_{cutoff}px"]=(sum(r.get("corner_error_px") is not None and r["corner_error_px"]<=cutoff for r in gt_geometry)/len(gt_geometry) if gt_geometry else None)
            delays=[r["latency_ms"] for r in test]
            summary["models"][alias]={"provenance":model.provenance(),"calibration":calibration,
                "test":metrics,"geometry":geometry,"adaptation_seconds":adaptation_seconds,
                "latency_ms":{"median":float(np.median(delays)),"p95":float(np.quantile(delays,.95))}}
            all_rows.extend(rows)
            write_json(output/"summary.json",summary)
            print(f"  test AUROC={metrics['overall']['auroc']}, TPR={metrics['overall']['tpr']}, FPR={metrics['overall']['fpr']}",flush=True)
    fields=["model","id","object_id","query_object_id","split","condition","present","score","raw_score","score_definition","accepted","latency_ms","corner_error_px","bbox_iou"]
    with (output/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(all_rows)
    # Streaming rows above are recoverable intermediate evidence. Commit calibrated
    # decisions to the final JSONL only after every model has finished successfully.
    temporary=output/"predictions.final.jsonl"
    with temporary.open("w",encoding="utf-8") as f:
        for row in all_rows:f.write(json.dumps(json_safe(row),allow_nan=False)+"\n")
    temporary.replace(output/"predictions.jsonl")
    summary["completed"]=True
    write_json(output/"summary.json",summary)
    from hypernet.reporting import make_report
    make_report(summary,all_rows,root,output)
    return summary
