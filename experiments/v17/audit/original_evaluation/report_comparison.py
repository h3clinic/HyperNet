"""Render measured comparison tables and a deterministic local failure gallery.

This script never trains, selects thresholds, fetches photos, or changes scores.
Real-photo derivatives stay local; publication can include aggregate results and
this code without redistributing those derivatives.
"""
from pathlib import Path
import argparse, html, json, sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_coco_v17 import read, sha, verify, verify_plan, DATA, PLAN, load_reference, load_dev_reference, load_test
from hypernet.detection_metrics import truth_arrays

NAMES = {
    'v17': 'HyperNet v17 / spatial matching',
    'v17_no_geometry': 'HyperNet / bag-of-parts control',
    'random': 'Untrained HyperNet',
    'raw_template': 'Raw pixel template matching',
    'owlvit_base_patch32': 'Official OWL-ViT B/32',
}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(args):
    plan = verify_plan(); rows = plan[args.split]; acquisition = None
    if args.split == 'test':
        acquisition = read(DATA / 'acquisition.json')
        if not acquisition['completed']: raise ValueError('Completed measured cohort required')
        keep = set(acquisition['eligible_image_ids'])
        rows = [r for r in rows if r['image_id'] in keep]
    targets = plan['targets']; present, valid = truth_arrays(rows, targets)
    results, arrays, cutoffs = {}, {}, {}
    for method in NAMES:
        directory = ROOT / 'experiments' / f'{method}_{args.split}_v2'
        results[method] = read(directory / 'result.json')
        if (results[method]['method'] != method or results[method]['split'] != args.split
                or results[method]['data_plan_sha256'] != sha(PLAN)):
            raise ValueError('Measured result belongs to a different method, split, or plan')
        if sha(directory / 'arrays.npz') != results[method]['score_file_sha256']:
            raise ValueError('Measured array hash changed')
        arrays[method] = dict(np.load(directory / 'arrays.npz'))
        if arrays[method]['image_ids'].tolist() != [r['image_id'] for r in rows]:
            raise ValueError('Photo pairing differs')
        if arrays[method]['category_ids'].tolist() != [t['category_id'] for t in targets]:
            raise ValueError('Target pairing differs')
        cutoffs[method] = read(ROOT / 'experiments' / f'{method}_dev_v2/calibration.json')['primary_5pct']['cutoff']
        if results[method]['summary']['primary_5pct']['cutoff'] != cutoffs[method]:
            raise ValueError('Displayed result uses a different calibration cutoff')
    paired_path = ROOT / 'experiments' / f'paired_comparison_{args.split}_v2.json'
    paired = read(paired_path)
    if paired['split'] != args.split or paired['data_plan_sha256'] != sha(PLAN):
        raise ValueError('Paired confidence intervals belong to a different split or plan')
    verify(paired['input_bindings'])
    for method in NAMES:
        for path in (ROOT/'experiments'/f'{method}_{args.split}_v2/arrays.npz',
                     ROOT/'experiments'/f'{method}_dev_v2/calibration.json'):
            if paired['input_bindings'].get(str(path.resolve())) != sha(path):
                raise ValueError('Paired intervals do not bind the displayed arrays/calibration')
    summaries = {k: v['summary']['primary_5pct'] for k, v in results.items()}
    for method, measured in summaries.items():
        for metric in ('macro_localized_recall', 'macro_false_positive_rate'):
            estimate = paired['operating_points']['primary_5pct']['methods'][method][metric]['estimate']
            if not np.isclose(estimate, measured[metric], rtol=0, atol=1e-12):
                raise ValueError('Paired interval estimate differs from the displayed result')
    primary, owl = summaries['v17'], summaries['owlvit_base_patch32']
    win = primary['macro_localized_recall'] > owl['macro_localized_recall'] and primary['macro_false_positive_rate'] <= owl['macro_false_positive_rate']
    conclusion = ('HyperNet meets the descriptive observed recall/false-alarm comparison on this cohort; confidence intervals and narrow scope still apply.'
                  if win else 'HyperNet has not beaten OWL-ViT on the predeclared recall and false-alarm comparison.')
    summary = {
        'status': 'complete', 'split': args.split,
        'summary': conclusion + (' These are fresh TEST results.' if args.split == 'test' else ' These are development results, not a fresh test.'),
        'data_plan_sha256': sha(PLAN), 'paired_comparison_sha256': sha(paired_path),
        'photos': len(rows), 'target_photo_pairs': int(present.size),
        'positive_pairs': int((present & valid).sum()), 'negative_pairs': int((~present & valid).sum()),
        'observed_win_gate': bool(win),
        'rows': [{'name': NAMES[k], 'localized_recall': f"{100*s['macro_localized_recall']:.2f}%",
                  'false_alarms': f"{10000*s['macro_false_positive_rate']:.1f} / 10,000 negative pairs",
                  'balanced_accuracy': f"{100*s['macro_presence_balanced_accuracy']:.2f}%"}
                 for k, s in summaries.items()],
        'metrics': summaries, 'paired_intervals': paired['operating_points'],
        'limitations': plan['limitations'] + [
            'Rates are macro averages over 12 fixed categories; the reference choice is not varied.',
            'The matched control changes the aggregation operator as well as removing relative part positions.',
            'Own synthetic pretraining and one real reference do not mean one total training image: labeled DEV images calibrate thresholds.',
            'No new algorithmic novelty or universal OWL-family superiority is established by this experiment.'
        ],
    }
    write(ROOT / ('results_summary.json' if args.split == 'test' else 'development_summary.json'), summary)
    gallery(args.split, rows, targets, plan, acquisition, arrays, cutoffs, present, valid)
    print(json.dumps({'summary': summary['summary'], 'rows': summary['rows']}))


def gallery(split, rows, targets, plan, acquisition, arrays, cutoffs, present, valid):
    decisions = {m: a['scores'] >= cutoffs[m] for m, a in arrays.items()}
    correct = {m: decisions[m] & (a['overlaps'] >= .5) & present & valid for m, a in arrays.items()}
    conditions = {
        'OWL localizes; HyperNet misses': present & valid & correct['owlvit_base_patch32'] & ~correct['v17'],
        'HyperNet localizes; OWL misses': present & valid & correct['v17'] & ~correct['owlvit_base_patch32'],
        'Both localize': correct['v17'] & correct['owlvit_base_patch32'],
        'Both miss localization': present & valid & ~correct['v17'] & ~correct['owlvit_base_patch32'],
        'HyperNet false alarm': ~present & valid & decisions['v17'],
        'OWL false alarm': ~present & valid & decisions['owlvit_base_patch32'],
    }
    # Two first image-id/category-order cases per outcome: deterministic,
    # explicitly illustrative selection, never an estimate of prevalence.
    selected = []
    for label, condition in conditions.items():
        for i, j in np.argwhere(condition)[:2]: selected.append((label, int(i), int(j)))
    destination = ROOT / 'assets' / f'{split}_comparison_examples'
    destination.mkdir(parents=True, exist_ok=True)
    references = {r['category_id']: r for r in plan['references']}
    cards, records = [], []
    for number, (label, i, j) in enumerate(selected):
        row, target = rows[i], targets[j]
        source = load_dev_reference(row) if split == 'dev' else load_test(row, acquisition)
        ref = load_reference(references[target['category_id']])
        canvas = Image.fromarray(source); scale = min(1., 720 / max(canvas.size))
        canvas = canvas.resize((round(canvas.width*scale), round(canvas.height*scale)))
        draw = ImageDraw.Draw(canvas)
        for box in row['boxes_by_target'][str(target['category_id'])]:
            draw.rectangle(tuple(float(v)*scale for v in box), outline='#61ff8d', width=2)
        descriptions = []
        for method, color, display in [('v17', '#ffae47', 'HyperNet'), ('owlvit_base_patch32', '#59ceff', 'OWL')]:
            accepted = bool(decisions[method][i, j]); box = arrays[method]['boxes'][i, j]
            if accepted:
                draw.rectangle(tuple(float(v)*scale for v in box), outline=color, width=3)
                draw.text((max(0, float(box[0])*scale), max(0, float(box[1])*scale-13)), display, fill=color, stroke_width=1, stroke_fill='black')
            descriptions.append(f"{display}: {'accepted' if accepted else 'rejected'}; IoU {arrays[method]['overlaps'][i,j]:.3f}")
        query_name, ref_name = f'{number:02d}_query.jpg', f'{number:02d}_reference.jpg'
        canvas.save(destination/query_name, quality=92)
        reference = Image.fromarray(ref); reference.thumbnail((160, 160)); reference.save(destination/ref_name, quality=95)
        rel = destination.relative_to(ROOT).as_posix()
        source_info = row.get('source', {})
        license_url = source_info.get('license_url', '')
        lic = f'<a href="{html.escape(license_url, quote=True)}">Source license</a>' if license_url else 'Source attribution is recorded in the cohort plan.'
        title = f"{label} · {target.get('category_name', target.get('name', target['category_id']))} · COCO {row['image_id']}"
        cards.append(f'<section><h2>{html.escape(title)}</h2><div class="images"><figure><img src="{rel}/{ref_name}" alt="Single reference"><figcaption>Single reference</figcaption></figure><figure><img src="{rel}/{query_name}" alt="Compared predictions"><figcaption>Green: truth. Orange: accepted HyperNet box. Blue: accepted OWL box.</figcaption></figure></div><p>{html.escape(" · ".join(descriptions))}</p><p>{lic}</p></section>')
        records.append({'selection': label, 'image_id': row['image_id'], 'category_id': target['category_id'], 'source': source_info})
    write(destination/'selection.json', {'split': split, 'rule': 'First two photo/category-order cases per outcome; posthoc illustrations, not prevalence estimates', 'examples': records})
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HyperNet v17 comparison examples</title><style>body{font:16px/1.5 system-ui;background:#101b28;color:#eef4fa;max-width:1100px;margin:auto;padding:24px}a{color:#86d8eb}section{border-top:1px solid #506075;margin-top:24px;padding-top:16px}.images{display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap}figure{margin:0}img{max-width:100%;height:auto}h2{font-size:20px}figcaption{font-size:13px}</style><a href="index.html">Experiment report</a><h1>Measured reference-to-photo examples</h1><p>Split: '+html.escape(split)+'. First two cases per outcome in fixed photo/category order. These are illustrations selected after scoring, not a representative accuracy sample. Rejected boxes are omitted. All aggregate metrics use the full cohort.</p>'+''.join(cards)+'</html>'
    (ROOT/f'{split}_comparison_gallery.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev', 'test'), default='test')
    main(parser.parse_args())
