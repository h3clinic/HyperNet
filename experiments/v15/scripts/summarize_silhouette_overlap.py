"""Posthoc grouping of saved correspondence predictions; no model calls."""
from pathlib import Path
import argparse, hashlib, json, statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    errors = [value for row in rows for value in row.get('endpoint_error_pixels', [])]
    anchors = sum(row['anchors'] for row in rows)
    if len(errors) != anchors:
        raise ValueError('Saved error and anchor counts differ')
    return {
        'families': len(rows),
        'usable_families': sum(row['anchors'] > 0 for row in rows),
        'families_without_anchors': sum(row['anchors'] == 0 for row in rows),
        'families_with_zero_target_foreground': sum(row['target_foreground_pixels'] == 0 for row in rows),
        'anchors': anchors,
        'pck2_count': sum(value <= 2 for value in errors),
        'pck4_count': sum(value <= 4 for value in errors),
        'pck_at_2px': sum(value <= 2 for value in errors) / anchors if anchors else None,
        'pck_at_4px': sum(value <= 4 for value in errors) / anchors if anchors else None,
        'mean_endpoint_error_px': statistics.mean(errors) if errors else None,
        'median_endpoint_error_px': statistics.median(errors) if errors else None,
        'family_ids': [row['family_id'] for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlap', type=Path, required=True)
    parser.add_argument('--results', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier diagnostic evidence')
    overlap = json.loads(args.overlap.read_text())
    seen = set(overlap['canonical_mask_overlap_family_ids'])
    report = {'completed': True, 'posthoc': True, 'source_sha256': sha(__file__),
              'overlap_audit_sha256': sha(args.overlap), 'model_calls': 0,
              'gradient_updates': 0, 'renderer_calls': 0, 'results': [],
              'interpretation': 'Exact canonical silhouette-hash overlap grouping of existing saved predictions. '
              'This posthoc analysis neither selects the final checkpoint nor changes the main evaluation. '
              'Unseen hashes do not establish unrelated geometry; function modes are shared.'}
    for path in args.results:
        saved = json.loads(path.read_text())
        if not saved['completed'] or saved['search_mode'] != 'full':
            raise ValueError('Completed all-pixel retrieval results required')
        rows = saved['rows']
        before = summarize(rows)
        if before['anchors'] != saved['anchors'] or before['pck_at_2px'] != saved['pck_at_2px']:
            raise ValueError('Saved overall summary does not match rows')
        report['results'].append({
            'checkpoint_name': saved['checkpoint_name'],
            'checkpoint_sha256': saved['checkpoint_sha256'],
            'input_stream_sha256': saved['input_stream_sha256'],
            'results_sha256': sha(path),
            'overall': before,
            'silhouette_seen': summarize([row for row in rows if row['family_id'] in seen]),
            'silhouette_unseen': summarize([row for row in rows if row['family_id'] not in seen]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output),
                      'final': {name: {k: v for k, v in report['results'][-1][name].items() if k != 'family_ids'}
                                for name in ('overall', 'silhouette_seen', 'silhouette_unseen')}}))


if __name__ == '__main__':
    main()
