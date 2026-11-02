"""Self-contained experiment report and compact visual failure evidence."""
from pathlib import Path
import html
import json
import numpy as np
from PIL import Image, ImageDraw
from hypernet.data import load_rgb

def _format(value, percent=True):
    if value is None:return "—"
    return f"{100*value:.1f}%" if percent else f"{value:.3f}"

def _preview(row,root,path):
    reference=Image.fromarray(load_rgb(root/row['reference']))
    query=Image.fromarray(load_rgb(root/row['query']))
    thumbw,thumbh=300,250
    canvas=Image.new('RGB',(640,320),'#f4f7f8');draw=ImageDraw.Draw(canvas)
    scales=[]
    for im,x in [(reference,10),(query,330)]:
        original=im.size;im.thumbnail((thumbw,thumbh))
        canvas.paste(im,(x,45));scales.append((im.width/original[0],im.height/original[1]))
    draw.text((10,10),'REFERENCE',fill='black')
    score_label='margin' if row.get('score_definition')=='route_cutoff_margin' else 'score'
    draw.text((330,10),f"QUERY | {'present' if row['present'] else 'absent'} | {score_label} {row['score']:.3f}",fill='black')
    prediction=row['prediction'];p=prediction.get('points_reference',[]);q=prediction.get('points_query',[])
    mask=prediction.get('inliers',[])
    kept=[i for i in range(min(len(p),len(q))) if not mask or (i<len(mask) and mask[i])][:35]
    for i in kept:
        a=(10+p[i][0]*scales[0][0],45+p[i][1]*scales[0][1])
        b=(330+q[i][0]*scales[1][0],45+q[i][1]*scales[1][1])
        if all(np.isfinite(v) for v in [*a,*b]):draw.line([a,b],fill='#26a875',width=1)
    draw.text((10,298),f"{row['model']} | {row['condition']} | {row['id'][:60]}",fill='black')
    canvas.save(path)

def make_report(summary,rows,root,output):
    root,output=Path(root),Path(output)
    esc=html.escape
    table=[];conditions=[];budgets=[];details=[]
    for name,model in summary['models'].items():
        m=model['test']['overall'];cal=model['calibration'];latency=model['latency_ms']
        route_calibration=cal.get('mode')=='route_budget'
        # Separate branch cutoffs support binary decisions, not comparable ranking units.
        auc='—*' if route_calibration else _format(m['auroc'],False)
        ap='—*' if route_calibration else _format(m['average_precision'],False)
        table.append(f"<tr><th>{esc(name)}</th><td>{auc}</td><td>{ap}</td><td>{_format(m['tpr'])}</td><td>{_format(m['fpr'])}</td><td>{_format(m['balanced_accuracy'])}</td><td>{latency['median']:.1f}</td></tr>")
        for condition,values in model['test']['by_condition'].items():
            condition_auc='—*' if route_calibration else _format(values['auroc'],False)
            conditions.append(f"<tr><td>{esc(name)}</td><td>{esc(condition)}</td><td>{_format(values['tpr'])}</td><td>{_format(values['fpr'])}</td><td>{condition_auc}</td></tr>")
        ci=model['test']['object_cluster_ci']
        details.append(f"<details><summary>{esc(name)}: calibration, confidence intervals, geometry and data budget</summary><pre>{esc(json.dumps({'calibration':cal,'object_cluster_ci':ci,'geometry':model['geometry'],'provenance':model['provenance']},indent=2))}</pre></details>")
    selected=[]
    for name,model in summary['models'].items():
        threshold=model['calibration']['threshold']
        test=[r for r in rows if r['model']==name and r['split']=='test']
        false_neg=sorted([r for r in test if r['present'] and r['score']<threshold],key=lambda r:r['score'])
        false_pos=sorted([r for r in test if not r['present'] and r['score']>=threshold],key=lambda r:-r['score'])
        for label,chosen in [('Missed target',false_neg[:2]),('False acceptance',false_pos[:2])]:
            for row in chosen:selected.append((label,row))
    examples=output/'examples';examples.mkdir(exist_ok=True)
    galleries=[]
    for i,(label,row) in enumerate(selected):
        filename=f'failure_{i:03}.jpg';_preview(row,root,examples/filename)
        galleries.append(f"<figure><img src='examples/{filename}' alt='{esc(label)}: {esc(row['model'])}' loading='lazy'><figcaption>{esc(label)} · {esc(row['model'])} · {esc(row['condition'])}</figcaption></figure>")
    dataset_notes=summary.get('dataset_notes') or []
    if isinstance(dataset_notes,str):dataset_notes=[dataset_notes]
    if any(m['calibration'].get('mode')=='route_budget' for m in summary['models'].values()):
        dataset_notes=list(dataset_notes)+['* Cascade scores are margins above separate route cutoffs. Their pooled AUROC/AP depend on arbitrary branch scales, so they are omitted here. JSON retains those diagnostic calculations; use TPR/FPR for cascade comparisons. These margins are not probabilities.']
    notes=''.join(f'<li>{esc(str(x))}</li>' for x in summary['interpretation']+summary.get('dataset_limitations',[])+dataset_notes)
    content=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>HyperNet starter evaluation</title><style>body{{background:#f5f7f8;color:#1c3038;font:16px/1.6 system-ui;margin:0}}main{{max-width:1200px;padding:36px 24px;margin:auto}}h1{{font-size:40px;line-height:1.15}}h2{{margin-top:32px}}.card,details{{background:white;border:1px solid #d1dce0;border-radius:9px;padding:18px;margin:16px 0}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}th,td{{padding:11px;text-align:left;border-bottom:1px solid #d6e0e4}}a{{color:#006963}}.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:15px}}figure{{margin:0;background:white;padding:8px}}img{{width:100%;height:auto}}figcaption{{font-size:13px}}.muted{{color:#526972}}summary{{cursor:pointer}}</style></head><body><main><p class="muted">HYPERNET / STARTER EXPERIMENT / {esc(summary['created_utc'][:10])}</p><h1>Measured first draft</h1><p>Protocol: <b>{esc(str(summary['protocol']))}</b>. All methods receive one reference per identity. Thresholds use development identities only; the table below contains held-out test predictions.</p><p><a href="summary.json">Full metrics + provenance</a> · <a href="predictions.csv">Prediction table</a> · <a href="predictions.jsonl">Raw correspondence evidence</a></p><div class="card"><b>How to read this run</b><ul>{notes}</ul><p>AUROC and binary AP summarize pair ranking. TPR is accepted present targets; FPR is accepted absent targets. The development FPR target was {_format(summary['target_dev_fpr'])}. These metrics are not COCO detection mAP.</p></div><h2>Held-out results</h2><div class="scroll"><table><thead><tr><th>Method</th><th>AUROC</th><th>Binary AP</th><th>TPR</th><th>FPR</th><th>Balanced accuracy</th><th>Median ms</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><h2>Conditions</h2><div class="scroll"><table><thead><tr><th>Method</th><th>Condition</th><th>TPR</th><th>FPR</th><th>AUROC</th></tr></thead><tbody>{''.join(conditions)}</tbody></table></div><h2>Failure examples</h2><p>Up to two lowest-scoring misses and two highest-scoring false acceptances per method. Lines visualize at most 35 retained correspondences; they are not human-verified annotations.</p><div class="gallery">{''.join(galleries) or '<p>No mistakes at the selected thresholds in this small test.</p>'}</div><h2>Audit details</h2>{''.join(details)}<details><summary>Environment and dataset validation</summary><pre>{esc(json.dumps({'versions':summary['versions'],'validation':summary['validation'],'manifest_sha256':summary['manifest_sha256'],'seed':summary['seed']},indent=2))}</pre></details><p class="muted">An experimental integration, not a reproduction of every cited paper and not a claim of state-of-the-art recognition.</p></main></body></html>'''
    (output/'report.html').write_text(content,encoding='utf-8')
