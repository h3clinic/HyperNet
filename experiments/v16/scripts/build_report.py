"""Build user-facing summaries from completed saved artifacts only."""
from pathlib import Path
import argparse, json, hashlib
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
NAMES = {'v16': 'v16 with relationships', 'v16_no_relations': 'Same training, without relation losses',
         'random': 'Random weights, same architecture', 'v15': 'Previous v15', 'raw48': 'Raw pixels (48×48)'}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(test, analysis=None, synthetic=None, message=None):
    r = read(test)
    if not r['completed'] or r['stage'] != 'test': raise ValueError('Completed TEST result required')
    paired = read(analysis) if analysis else None
    if paired and paired['inputs'].get(str((Path(test).parent / 'scores.npz').resolve())) != r['scores_sha256']:
        raise ValueError('Paired analysis belongs to different TEST scores')
    rows = []
    for alias, metrics in r['metrics'].items():
        ci = paired['bootstrap']['methods'][alias] if paired and paired['bootstrap']['available'] else None
        rows.append({'alias': alias, 'name': NAMES[alias], 'primary': alias == 'v16',
                     'macro': metrics['macro'], 'union': metrics['union'], 'per_target': metrics['per_target'],
                     'intervals': ci})
    training = {}
    for alias, folder in [('v16', 'primary_4096'), ('v16_no_relations', 'no_relation_aux_4096')]:
        path = ROOT / 'training' / folder / 'completion.json'; c = read(path)
        if c['status'] != 'complete': raise ValueError('Both training arms must be complete')
        training[alias] = {k: c[k] for k in ('optimizer_steps', 'training_views', 'unique_structural_signatures', 'unique_rgb_views', 'unique_raster_masks', 'seconds', 'input_stream_sha256')}
        training[alias]['completion_sha256'] = sha(path)
    q = read(ROOT / 'data/gravity_fresh_v16/blind_quality_audit.json')
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(),
              'summary': message or 'The frozen fresh-image comparison is complete. See all methods, error counts and uncertainty below.',
              'rows': rows, 'training': training, 'events': r['events'], 'time_groups': r['time_groups'],
              'targets': r['target_labels'], 'selected_recipe': r['selected_recipe'],
              'quality': {'requested': q['requested'], 'eligible': q['eligible'], 'excluded': len(q['excluded'])},
              'reference_images_per_target': 1, 'additional_real_calibration_events': r['additional_real_calibration_events'],
              'borrowed_pretrained_weights': False, 'natural_weight_training_images': 0,
              'paired_contrasts': paired['bootstrap']['contrasts'] if paired and paired['bootstrap']['available'] else None,
              'synthetic_diagnosis': read(synthetic) if synthetic else None,
              'provenance': {'test_results_sha256': sha(test), 'test_scores_sha256': r['scores_sha256'],
                             'paired_analysis_sha256': sha(analysis) if analysis else None,
                             'protocol_sha256': r['protocol_sha256'], 'admission_sha256': r['admission_sha256']},
              'limitations': ['Labels are volunteer consensus, not expert-confirmed physical truth or classifier predictions.',
                             'The cohort contains selected, triggered O1/O2 events. It is not an unbiased continuous detector archive.',
                             'False alarms per 10,000 are normalized rates with finite actual negative denominators.',
                             'Paired bootstrap holds references and calibration fixed; it does not capture those choices or establish physical independence.',
                             'Training types are embedded mathematical graph structures, not all possible natural categories.',
                             'The two trained arms reused exactly the same rendered data; do not add their unique-image counts.']}
    out = ROOT / 'results_summary.json'; out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': 'http://127.0.0.1:8797/', 'summary_sha256': sha(out), 'methods': len(rows), 'test_events': r['events']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test', type=Path, required=True); p.add_argument('--analysis', type=Path)
    p.add_argument('--synthetic', type=Path); p.add_argument('--message-file', type=Path)
    a = p.parse_args(); build(a.test, a.analysis, a.synthetic, a.message_file.read_text(encoding='utf-8').strip() if a.message_file else None)
