"""Development-only budgeted calibration for heterogeneous fallback routes."""
from copy import deepcopy
import numpy as np
from .metrics import calibrate_threshold, binary_metrics

ROUTES=("primary","alignment_fallback")

def calibrate_route_thresholds(dev_rows,target_fpr=.05):
    rows=list(dev_rows)
    if not rows or any(r.get("split")!="dev" for r in rows):
        raise ValueError("Route calibration requires development rows only")
    if not 0<=target_fpr<1:
        raise ValueError("target_fpr must be in [0,1)")
    for r in rows:
        route=r["prediction"]["evidence"].get("route")
        if route not in {*ROUTES,"both_rejected"}:
            raise ValueError("Missing/unknown operational cascade route")
        if not np.isfinite(r["score"]) or r["score"]<0:
            raise ValueError("Route scores must be finite and nonnegative")
    branches={}
    for route in ROUTES:
        # All negatives remain in the denominator. Other routes have zero score
        # and are not eligible for this branch, so the two error budgets add.
        branch_rows=[{**r,"score":r["score"] if r["prediction"]["evidence"]["route"]==route else 0.} for r in rows]
        branches[route]=calibrate_threshold(branch_rows,target_fpr/len(ROUTES))
    calibrated=apply_route_thresholds(rows,{"branch_thresholds":branches})
    m=binary_metrics([r["present"] for r in calibrated],[r["score"] for r in calibrated],0.)
    if m["fpr"] is not None and m["fpr"]>target_fpr+1e-12:
        raise AssertionError("Development error budget was exceeded")
    return {"threshold":0.,"rule":"route-specific score margin >= 0","mode":"route_budget",
            "branch_thresholds":branches,"target_fpr":target_fpr,"n_dev":len(rows),
            "n_dev_negative":sum(not r["present"] for r in rows),
            "empirical_dev_fpr":m["fpr"],"empirical_dev_tpr":m["tpr"],
            "calibration_object_ids":sorted({str(r["object_id"]) for r in rows}),
            "score_definition":"Raw selected-route score minus its development cutoff; rejected routes score -1. Raw scores preserved separately.",
            "budget_allocation":"Half the total empirical development FPR budget per operational route; all development negatives in each denominator.",
            "interpretation":"Development empirical control only; no population/test FPR guarantee. Margin scores are not calibrated probabilities."}

def apply_route_thresholds(rows,calibration):
    result=[]
    for original in rows:
        row=deepcopy(original)
        raw=float(row["score"])
        route=row["prediction"]["evidence"].get("route")
        if route in ROUTES:
            score=raw-calibration["branch_thresholds"][route]["threshold"]
        elif route=="both_rejected":
            score=-1.
        else:
            raise ValueError("Unknown operational route")
        row["raw_score"]=raw
        row["score"]=float(score)
        row["score_definition"]="route_cutoff_margin"
        result.append(row)
    return result

