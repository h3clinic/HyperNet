"""Build the fresh O3b cohort for v0.5: roster, then images, then arrays.

    python scripts/prepare_gravity_spy_o3.py --stage roster --tables <folder> --output <folder>
    python scripts/prepare_gravity_spy_o3.py --stage fetch  --output <folder>
    python scripts/prepare_gravity_spy_o3.py --stage arrays --output <folder>

The roster is written and hashed before any image is fetched. No v0.5 score
exists until arrays are built, and no TEST score until the v0.5 protocol is
frozen separately.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet import gravity_spy_o3 as o3

CONFIDENCE = 0.95
PER_CLASS_TEST = 150
PER_CLASS_CALIB = 50
SEED = 20261103


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('roster', 'fetch', 'arrays'), required=True)
    parser.add_argument('--tables', type=Path); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--v01', type=Path, default=ROOT / 'benchmarks/gravity_spy_v0.1/protocol.json')
    parser.add_argument('--v11-metadata', type=Path, default=Path('C:/Users/brita/Documents/Codex/2026-11-01/wan/gravity_spy_data/trainingset_v1d1_metadata.csv'))
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    if args.stage == 'roster':
        if (out / 'roster.json').exists():
            raise FileExistsError('Roster already frozen')
        v01 = json.loads(args.v01.read_text()); classes = v01['target_classes']
        table = o3.load_tables(args.tables)
        v11_ids = set(pd.read_csv(args.v11_metadata)['gravityspy_id'])
        overlap = int(table['gravityspy_id'].isin(v11_ids).sum())
        counts_all = table['ml_label'].value_counts().to_dict()
        counts_conf = table[table['ml_confidence'] >= CONFIDENCE]['ml_label'].value_counts().to_dict()
        test, calib = o3.sample_cohort(table, classes, confidence=CONFIDENCE, per_class=PER_CLASS_TEST,
                                       calib_per_class=PER_CLASS_CALIB, seed=SEED, exclude_ids=v11_ids)
        keep = ['gravityspy_id', 'ifo', 'event_time', 'peak_frequency', 'snr', 'ml_label', 'ml_confidence', 'target', 'group'] + list(o3.URL_COLUMNS.values())
        test[keep].to_csv(out / 'test_roster.csv', index=False); calib[keep].to_csv(out / 'calib_roster.csv', index=False)
        roster = {'format': 'hypernet.gravity_spy_o3_roster.v0.5', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'source': o3.ZENODO, 'tables': list(o3.TABLE_MD5), 'table_md5': o3.TABLE_MD5,
                  'label_source': 'Gravity Spy CNN ml_label with ml_confidence >= %.2f; not volunteer or expert labels' % CONFIDENCE,
                  'time_grouping_seconds': o3.GROUP_SECONDS, 'one_event_per_group': True, 'seed': SEED,
                  'per_class_caps': {'test': PER_CLASS_TEST, 'calib': PER_CLASS_CALIB},
                  'target_classes': classes, 'distractor_label': 'other (ml_label outside the 20 target classes, including O3-only classes)',
                  'v11_id_overlap_in_tables': overlap, 'table_rows': int(len(table)),
                  'label_counts_all': counts_all, 'label_counts_confident': counts_conf,
                  'test_counts': test['target'].value_counts().to_dict(), 'calib_counts': calib['target'].value_counts().to_dict(),
                  'test_events': int(len(test)), 'calib_events': int(len(calib)),
                  'test_roster_sha256': hashlib.sha256((out / 'test_roster.csv').read_bytes()).hexdigest(),
                  'calib_roster_sha256': hashlib.sha256((out / 'calib_roster.csv').read_bytes()).hexdigest(),
                  'images_fetched': False}
        (out / 'roster.json').write_text(json.dumps(roster, indent=2))
        print(json.dumps({k: roster[k] for k in ('table_rows', 'v11_id_overlap_in_tables', 'test_events', 'calib_events', 'test_counts', 'calib_counts')}, indent=1))
        return

    roster = json.loads((out / 'roster.json').read_text())
    if args.stage == 'fetch':
        for split in ('calib', 'test'):
            rows = pd.read_csv(out / f'{split}_roster.csv')
            records, failures = o3.fetch_images(rows, out / 'images' / split)
            (out / f'{split}_images.json').write_text(json.dumps({'sha256': records, 'failures': failures}, indent=1))
            print(f'{split}: {len(records)} files, {len(failures)} failures')
        return

    if args.stage == 'arrays':
        from PIL import Image
        for split in ('calib', 'test'):
            rows = pd.read_csv(out / f'{split}_roster.csv'); folder = out / 'images' / split
            arrays = {d: [] for d in o3.URL_COLUMNS}; keep = []; dropped = []
            for _, row in rows.iterrows():
                paths = {d: folder / f"{row['gravityspy_id']}_{d}.png" for d in o3.URL_COLUMNS}
                missing = [d for d, p in paths.items() if not p.exists()]
                if missing:
                    dropped.append({'gravityspy_id': row['gravityspy_id'], 'target': row['target'], 'reason': 'missing_png', 'durations': missing}); continue
                try:
                    for p in paths.values():
                        Image.open(p).verify()
                    imgs = {d: o3.png_to_array(p) for d, p in paths.items()}
                except Exception as e:
                    dropped.append({'gravityspy_id': row['gravityspy_id'], 'target': row['target'], 'reason': f'decode_error: {e}'}); continue
                for d in o3.URL_COLUMNS:
                    arrays[d].append(imgs[d])
                keep.append(row['gravityspy_id'])
            blank = [g for g, a in zip(keep, arrays['1.0']) if float(a.max() - a.min()) <= 1e-6]
            np.savez_compressed(out / f'{split}_arrays.npz', gravityspy_id=np.array(keep), **{f'd{d}': np.stack(arrays[d]) for d in o3.URL_COLUMNS})
            kept = rows[rows['gravityspy_id'].isin(keep)]['target'].value_counts().to_dict()
            zero = [c for c in roster['target_classes'] if roster[f'{split}_counts'].get(c, 0) > 0 and kept.get(c, 0) == 0]
            manifest = {'split': split, 'roster_events': int(len(rows)), 'complete_events': len(keep), 'dropped': dropped,
                        'blank_on_1s_kept_in_npz': blank, 'kept_counts': kept, 'resample': 'linear',
                        'npz_sha256': hashlib.sha256((out / f'{split}_arrays.npz').read_bytes()).hexdigest()}
            (out / f'{split}_arrays.json').write_text(json.dumps(manifest, indent=1))
            print(f'{split}: {len(keep)} complete of {len(rows)}, {len(dropped)} dropped, {len(blank)} blank on 1.0 s')
            if zero:
                raise RuntimeError(f'{split}: classes with zero complete events: {zero}')
        return


if __name__ == '__main__':
    main()
