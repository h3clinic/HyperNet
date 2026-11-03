"""Render completed results and deterministic illustrative cases; never infer."""
from pathlib import Path
import hashlib
import html
import json
import sys
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'evaluation'))
from data_protocol import load_plan, load_test, load_reference, read, sha, verify_protocol, METHODS
from vendor_v17.detection_metrics import summarize, truth_arrays

NAMES = {'v19': 'HyperNet v19', 'v17': 'HyperNet v17', 'raw_template': 'Raw template',
         'owlvit_base_patch32': 'Official image-guided OWL-ViT B/32'}


def checked_result_arrays(method, result, arrays, calibration, rows, targets, plan_sha, protocol_sha, protocol):
    """Reject mixed cohorts, target orders or DEV cutoffs before rendering."""
    if result.get('method') != method or result.get('split') != 'test':
        raise ValueError('Report needs the completed method TEST result')
    if result['data_plan_sha256'] != plan_sha or result['protocol_sha256'] != protocol_sha:
        raise ValueError('Result belongs to another cohort or frozen protocol')
    if arrays['image_ids'].tolist() != [r['image_id'] for r in rows]:
        raise ValueError('Result photo roster differs')
    if arrays['category_ids'].tolist() != [t['category_id'] for t in targets]:
        raise ValueError('Result target ordering differs')
    shape = (len(rows), len(targets))
    if arrays['scores'].shape != shape or arrays['overlaps'].shape != shape or arrays['boxes'].shape != (*shape, 4):
        raise ValueError('Incomplete result arrays')
    if not all(np.isfinite(arrays[k]).all() for k in ('scores', 'overlaps', 'boxes')):
        raise ValueError('Nonfinite report prediction')
    point = calibration['primary_5pct']; frozen = protocol['methods'][method]
    if point['nominal_dev_negative_fpr'] != .05 or point['cutoff'] != frozen['cutoffs']['primary_5pct']:
        raise ValueError('Report calibration differs from the frozen DEV cutoff')
    if result['configuration'] != frozen['configuration']:
        raise ValueError('Report model/preprocessing differs from the frozen method')
    positive, valid = truth_arrays(rows, targets)
    measured = summarize(arrays['scores'], arrays['overlaps'], positive, valid, point['cutoff'], targets)
    if measured != result['summary']['primary_5pct']:
        raise ValueError('Report summary differs from the measured arrays and DEV cutoff')
    return measured, point['cutoff']


def main():
    base = ROOT/'evaluation/results'; acquisition = read(ROOT/'evaluation/data/acquisition.json')
    if acquisition.get('completed') is not True: raise ValueError('Report requires completed acquisition')
    protocol = verify_protocol(acquisition['protocol_path'], acquisition['protocol_sha256'])
    paired = read(base/'paired_test.json'); plan = load_plan()
    plan_sha = sha(ROOT/'evaluation/data/plan.json')
    if acquisition['data_plan_sha256'] != plan_sha or paired['data_plan_sha256'] != plan_sha or paired.get('split') != 'test':
        raise ValueError('Acquisition/paired comparison belongs to another cohort')
    for path, expected in paired['input_bindings'].items():
        if sha(Path(path)) != expected: raise ValueError('Paired comparison input changed')
    eligible = acquisition['eligible_image_ids']
    if len(set(eligible)) != len(eligible) or not set(eligible) <= {r['image_id'] for r in plan['test']}:
        raise ValueError('Invalid eligible TEST roster')
    rows = [r for r in plan['test'] if r['image_id'] in set(acquisition['eligible_image_ids'])]
    targets = plan['targets']; summaries = {}; arrays = {}; cutoffs = {}; bindings = {}
    if paired['photos'] != len(rows) or paired['nominal_pairs'] != len(rows)*len(targets):
        raise ValueError('Paired comparison count differs from the report cohort')
    for method in METHODS:
        result_path = base/f'{method}_test/result.json'; calibration_path = base/f'{method}_dev/calibration.json'
        result = read(result_path); path = base/f'{method}_test/arrays.npz'
        if result['score_file_sha256'] != sha(path): raise ValueError('Result arrays changed')
        a = dict(np.load(path, allow_pickle=False))
        summaries[method], cutoffs[method] = checked_result_arrays(method, result, a, read(calibration_path), rows, targets,
                                                                  plan_sha, acquisition['protocol_sha256'], protocol)
        arrays[method] = a
        for input_path in (path, result_path, calibration_path):
            bindings[str(input_path.relative_to(ROOT)).replace('\\', '/')] = sha(input_path)
    positive = np.array([[bool(row['boxes_by_target'][str(t['category_id'])]) for t in targets] for row in rows])
    valid = np.array([[row['valid_by_target'][str(t['category_id'])] for t in targets] for row in rows])
    accepted = {m: a['scores'] >= cutoffs[m] for m, a in arrays.items()}
    hits = {m: accepted[m] & (a['overlaps'] >= .5) & positive & valid for m, a in arrays.items()}
    gate = paired['operating_points']['primary_5pct']['v19_minus_owlvit_base_patch32']['observed_win_gate']
    status = ('V19 passes the observed recall/false-alarm comparison on this test; inspect uncertainty and scope.' if gate else
              'V19 has not beaten OWL-ViT on the predeclared recall and false-alarm comparison.')
    report = {'status_text': status, 'photos': len(rows), 'positive_pairs': int((positive & valid).sum()),
              'negative_pairs': int((~positive & valid).sum()), 'methods': [], 'report_url': 'comparison.html',
              'paired_comparison': paired, 'input_bindings': bindings, 'data_plan_sha256': plan_sha,
              'protocol_sha256': acquisition['protocol_sha256'], 'paired_comparison_sha256': sha(base/'paired_test.json')}
    table = []
    for m in METHODS:
        s = summaries[m]
        item = {'name': NAMES[m], 'recall': f"{100*s['macro_localized_recall']:.2f}%",
                'false_alarms': f"{10000*s['macro_false_positive_rate']:.1f}",
                'hits': f"{s['localized_hits']} / {s['positive_pairs']}", 'metrics': s}
        report['methods'].append(item)
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                     (item['name'], item['recall'], item['false_alarms'], item['hits'],
                      f"{s['false_presence']} / {s['negative_pairs']}"))+'</tr>')
    groups = [('V19 hit; OWL miss', hits['v19'] & ~hits['owlvit_base_patch32']),
              ('OWL hit; V19 miss', hits['owlvit_base_patch32'] & ~hits['v19']),
              ('Both localized correctly', hits['v19'] & hits['owlvit_base_patch32']),
              ('V19 false alarm', accepted['v19'] & ~positive & valid),
              ('Both missed a present target', positive & valid & ~hits['v19'] & ~hits['owlvit_base_patch32'])]
    asset = ROOT/'assets/photo_cases'; asset.mkdir(parents=True, exist_ok=True)
    refs = {r['category_id']: r for r in plan['references']}
    sections = []; selections = []
    for group_name, mask in groups:
        locations = np.argwhere(mask); cards = []
        for i, j in locations[:2]:
            i, j = int(i), int(j); row = rows[i]; target = targets[j]; cid = target['category_id']
            query = Image.fromarray(load_test(row, acquisition)); reference = Image.fromarray(load_reference(refs[cid]))
            query.thumbnail((840, 550)); scale_x = query.width/row['width']; scale_y = query.height/row['height']
            draw = ImageDraw.Draw(query)
            for box in row['boxes_by_target'][str(cid)]:
                draw.rectangle((np.asarray(box)*[scale_x, scale_y, scale_x, scale_y]).tolist(), outline='#75ffb7', width=3)
            for method, color in [('owlvit_base_patch32', '#54c8ff'), ('v19', '#ffad54')]:
                box = arrays[method]['boxes'][i, j]*[scale_x, scale_y, scale_x, scale_y]
                draw.rectangle(box.tolist(), outline=color, width=3 if accepted[method][i, j] else 1)
            reference.thumbnail((150, 150))
            canvas = Image.new('RGB', (query.width+170, max(query.height, 190)), '#10151a')
            canvas.paste(reference, ((150-reference.width)//2+10, 28)); canvas.paste(query, (170, 0))
            ImageDraw.Draw(canvas).text((10, 7), 'ONE REFERENCE', fill='#e8edf2')
            filename = f'{row["image_id"]}_{cid}.jpg'; canvas.save(asset/filename, quality=92)
            notes = []
            for method in ('v19', 'owlvit_base_patch32'):
                notes.append(f"{NAMES[method]}: score {arrays[method]['scores'][i,j]:.4f}, IoU {arrays[method]['overlaps'][i,j]:.3f}, "+
                             ('accepted' if accepted[method][i, j] else 'rejected'))
            cards.append(f'<article><h3>{html.escape(target["name"])} · photo {row["image_id"]}</h3>'
                         f'<img src="assets/photo_cases/{filename}" alt="Reference and annotated query">'
                         +'<p>'+'<br>'.join(html.escape(n) for n in notes)+'</p></article>')
            selections.append({'group': group_name, 'image_id': row['image_id'], 'category_id': cid,
                               'pair_index': [i, j], 'asset': 'assets/photo_cases/'+filename,
                               'asset_sha256': sha(asset/filename), 'notes': notes})
        sections.append(f'<h2>{html.escape(group_name)} · {len(locations)} cases</h2>'+(''.join(cards) or '<p>No cases in this group.</p>'))
    comparison = paired['operating_points']['primary_5pct']['v19_minus_owlvit_base_patch32']['macro_localized_recall']
    difference = 'V19 − OWL macro recall: '+f"{100*comparison['estimate']:.2f} percentage points; 95% paired photo-bootstrap interval ["+\
                 f"{100*comparison['ci95'][0]:.2f}, {100*comparison['ci95'][1]:.2f}]."
    fp_comparison = paired['operating_points']['primary_5pct']['v19_minus_owlvit_base_patch32']['macro_false_positive_rate']
    difference += ' V19 − OWL macro false-positive rate: '+f"{100*fp_comparison['estimate']:.2f} percentage points; 95% interval ["+\
                  f"{100*fp_comparison['ci95'][0]:.2f}, {100*fp_comparison['ci95'][1]:.2f}]."
    interval_caution = paired.get('zero_hit_caution', '')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HyperNet v19 — measured comparison</title>
<style>body{{background:#10151a;color:#e8edf2;font:16px/1.55 system-ui;margin:0}}main{{max-width:1120px;margin:auto;padding:32px 24px}}a{{color:#82d8ca}}p{{color:#b9c6d0}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #34434e;text-align:left}}article{{background:#172028;border:1px solid #34434e;padding:18px;border-radius:12px;margin:16px 0}}img{{max-width:100%;height:auto}}h1{{font-size:38px;line-height:1.15}}h2{{margin-top:32px}}.scroll{{overflow:auto}}</style>
<main><a href="index.html#results">Back to experiment</a><h1>Measured fresh-photo comparison</h1><p><strong>{html.escape(status)}</strong></p>
<p>{len(rows)} photos, {int((positive & valid).sum())} positive target/photo cases, {int((~positive & valid).sum())} negative cases. One fixed reference per category; no real-image weight updates or borrowed HyperNet weights.</p>
<div class="scroll"><table><tr><th>Method</th><th>Macro localized recall</th><th>Macro false alarms / 10,000 negatives</th><th>Hits</th><th>False alarms</th></tr>{''.join(table)}</table></div><p>{html.escape(difference)}</p><p>{html.escape(interval_caution)}</p>
<p>Correct localization requires IoU ≥ 0.5. The primary cutoff for each method was fitted to a nominal 5% false-alarm rate on 288 old development photos, then frozen; the table reports the actual fresh-test false alarms. This is a custom metric, not COCO AP. Natural development labels were used for engineering and calibration. Freshness is relative to HyperNet; COCO may occur in OWL pretraining. Fixed references and category choices limit generalization of this comparison.</p>
<h2>Illustrative cases</h2><p>First two cases in each outcome group, in frozen photo/category order, selected after scoring. These illustrate errors and successes; the full table above uses every eligible case. Green: ground truth. Orange: V19. Blue: OWL. Thin predicted boxes were rejected by their cutoff. Photo derivatives stay local and are excluded from the public code package.</p>{''.join(sections)}</main></html>'''
    (ROOT/'comparison.html').write_text(page, encoding='utf-8')
    (ROOT/'report_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (ROOT/'assets/photo_cases/selection.json').write_text(json.dumps({'rule': 'First two in frozen row-major order per outcome, post hoc',
        'input_bindings': bindings, 'cases': selections}, indent=2)+'\n')
    print(json.dumps({'photos': len(rows), 'status': status, 'selected_cases': len(selections)}))


if __name__ == '__main__': main()
