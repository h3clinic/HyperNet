"""Bind the completed blind acquisition audit to the already frozen scorer."""
from pathlib import Path
import argparse, json, hashlib
from datetime import datetime, timezone


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def admit(protocol_path, output):
    protocol_path = Path(protocol_path).resolve(); output = Path(output)
    if output.exists(): raise FileExistsError('Do not overwrite an earlier admission')
    p = read(protocol_path); root = Path(p['plan_path']).parent
    for name, expected in p['bindings'].items():
        if sha(name) != expected: raise ValueError('Protocol-bound artifact changed: ' + name)
    qpath = root / 'blind_quality_audit.json'; q = read(qpath)
    acquired = read(root / 'acquisition_test.json')
    gate = read(root / 'acquisition_admission.json'); plan = read(root / 'plan.json')
    if q['fresh_plan_sha256'] != sha(root / 'plan.json') or q['protocol_sha256'] != sha(protocol_path):
        raise ValueError('Blind quality audit refers to a different plan/protocol')
    if q['acquisition_sha256'] != sha(root / 'acquisition_test.json') or acquired['failures']:
        raise ValueError('Incomplete or changed acquisition')
    if acquired['event_ids'] != sorted(r['event_id'] for r in plan['events']) or acquired['requested'] != len(plan['events']):
        raise ValueError('Acquisition does not cover the entire frozen cohort')
    if gate['protocol_sha256'] != sha(protocol_path) or gate['fresh_plan_sha256'] != sha(root / 'plan.json'):
        raise ValueError('Actual acquisition admission differs')
    if q['model_scores_used'] or q['manual_image_curation'] or q['replacement_sampling']:
        raise ValueError('Only blind, predeclared quality exclusions are admitted')
    expected_ids = {r['event_id'] for r in plan['events']}
    kept = q['eligible_event_ids']; removed = [r['event_id'] for r in q['excluded']]
    if len(set(kept)) != len(kept) or set(kept) & set(removed) or set(kept) | set(removed) != expected_ids:
        raise ValueError('Quality audit does not partition the exact fixed cohort')
    if set(q['image_record_sha256']) != expected_ids:
        raise ValueError('Quality audit omitted acquired sidecars')
    bindings = {str(path.resolve()): sha(path) for path in
                (qpath, root / 'acquisition_test.json', root / 'acquisition_admission.json')}
    for event_id, expected in q['image_record_sha256'].items():
        path = root / 'image_records' / (event_id + '.json')
        if sha(path) != expected: raise ValueError('Quality-bound image record changed')
        bindings[str(path.resolve())] = expected
    result = {'format': 'hypernet.v16.fresh_scoring_admission', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'test_scoring_authorized': True, 'protocol_path': str(protocol_path), 'protocol_sha256': sha(protocol_path),
              'fresh_plan_sha256': sha(root / 'plan.json'), 'blind_quality_audit_sha256': sha(qpath),
              'effective_event_ids': kept, 'image_record_sha256': q['image_record_sha256'], 'bindings': bindings,
              'requested': len(expected_ids), 'eligible': len(kept), 'excluded': len(removed),
              'model_scores_used_for_admission': False, 'replacement_sampling': False}
    with output.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'admission': str(output), 'sha256': sha(output), 'eligible': len(kept), 'excluded': len(removed)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); admit(args.protocol, args.output)
