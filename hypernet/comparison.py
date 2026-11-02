"""Merge completed runs on an identical manifest without rerunning models."""
from pathlib import Path
import csv
import copy
import json
from hypernet.runner import write_json
from hypernet.reporting import make_report

def combine_runs(run_dirs,output_dir,replace_models=False):
    output=Path(output_dir).resolve()
    if output.exists():raise FileExistsError("Combined output must be a new directory")
    runs=[Path(p).resolve() for p in run_dirs]
    if not runs:raise ValueError("At least one run required")
    summaries=[json.loads((p/'summary.json').read_text(encoding='utf-8')) for p in runs]
    if any(s.get('completed') is not True for s in summaries):raise ValueError("Only completed runs may be combined")
    identity={(s['manifest_sha256'],s['target_dev_fpr']) for s in summaries}
    if len(identity)!=1:raise ValueError("Runs must use the identical manifest and calibration target")
    merged=copy.deepcopy(summaries[0]);merged['models']={};merged['source_runs']=[str(p) for p in runs]
    merged['superseded_models']=[]
    rows=[]
    for path,summary in zip(runs,summaries):
        overlap=set(merged['models'])&set(summary['models'])
        if overlap and not replace_models:raise ValueError(f"Duplicate model aliases: {sorted(overlap)}")
        if overlap:
            merged['superseded_models'].append({'models':sorted(overlap),'replacement_run':str(path),
                'reason':'Caller explicitly selected a later corrected run; original run folders are preserved.'})
            rows=[r for r in rows if r['model'] not in overlap]
        merged['models'].update(summary['models'])
        for line in (path/'predictions.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if not isinstance(row.get('accepted'),bool):raise ValueError("Uncalibrated prediction record")
            rows.append(row)
    output.mkdir(parents=True)
    write_json(output/'summary.json',merged)
    with (output/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    fields=['model','id','object_id','query_object_id','split','condition','present','score','accepted','latency_ms','corner_error_px','bbox_iou']
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    make_report(merged,rows,Path(merged['manifest']).parent,output)
    return output/'report.html'
