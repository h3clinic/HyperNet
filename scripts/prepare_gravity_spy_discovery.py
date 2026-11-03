"""Candidate pool for the rare-glitch discovery run over O3b.

Pool rule (declared here, hashed into pool.json before any image is fetched):
every O3b event whose Gravity Spy CNN confidence is below 0.7, or whose CNN
probability for any of the five rare target types is at least 0.3 while its
label is a different class. These are the events the official labeler was
unsure about or half-saw a rare type in. One duration (0.5 s) per event.

    python scripts/prepare_gravity_spy_discovery.py --stage roster --tables <folder> --output <folder>
    python scripts/prepare_gravity_spy_discovery.py --stage fetch  --output <folder>
    python scripts/prepare_gravity_spy_discovery.py --stage arrays --output <folder>
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet import gravity_spy_o3 as o3

RARE = ('Paired_Doves', '1400Ripples', 'Chirp', 'Helix', 'Wandering_Line')
LOW_CONFIDENCE = 0.7
RARE_PROBABILITY = 0.3
DURATION = '0.5'
CHUNK = 4000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('roster', 'fetch', 'arrays'), required=True)
    parser.add_argument('--tables', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    if args.stage == 'roster':
        if (out / 'pool.json').exists():
            raise FileExistsError('Pool already frozen')
        table = o3.load_tables(args.tables)
        low = (table['ml_confidence'] < LOW_CONFIDENCE).to_numpy()
        second = np.zeros(len(table), bool)
        for r in RARE:
            second |= ((table['ml_label'] != r) & (table[r] >= RARE_PROBABILITY)).to_numpy()
        pool = table[low | second].copy()
        pool['why'] = np.where(low[low | second] & second[low | second], 'both', np.where(low[low | second], 'low_confidence', 'rare_probability'))
        pool['group'] = o3.time_groups(pool)
        keep = ['gravityspy_id', 'ifo', 'event_time', 'peak_frequency', 'snr', 'ml_label', 'ml_confidence', 'why', 'group'] + list(RARE) + [o3.URL_COLUMNS[DURATION]]
        pool[keep].to_csv(out / 'pool_roster.csv', index=False)
        record = {'format': 'hypernet.gravity_spy_discovery_pool.v0.1', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'source': o3.ZENODO, 'table_md5': o3.TABLE_MD5, 'table_rows': int(len(table)),
                  'rule': f'ml_confidence < {LOW_CONFIDENCE} OR (ml_label != r AND P(r) >= {RARE_PROBABILITY}) for r in rare types',
                  'rare_types': list(RARE), 'duration': DURATION, 'events': int(len(pool)),
                  'why_counts': pool['why'].value_counts().to_dict(), 'ml_label_counts': pool['ml_label'].value_counts().to_dict(),
                  'ifo_counts': pool['ifo'].value_counts().to_dict(), 'time_groups_4s': int(pool['group'].nunique()),
                  'roster_sha256': hashlib.sha256((out / 'pool_roster.csv').read_bytes()).hexdigest(), 'images_fetched': False}
        (out / 'pool.json').write_text(json.dumps(record, indent=2))
        print(json.dumps({k: record[k] for k in ('events', 'why_counts', 'ifo_counts', 'time_groups_4s')}, indent=1))
        return

    rows = pd.read_csv(out / 'pool_roster.csv')
    if args.stage == 'fetch':
        records, failures = o3.fetch_images(rows, out / 'images', durations=(DURATION,), pause=.2)
        (out / 'images.json').write_text(json.dumps({'count': len(records), 'failures': failures, 'sha256': records}, indent=1))
        print(f'{len(records)} files, {len(failures)} failures')
        return

    if args.stage == 'arrays':
        from PIL import Image
        ids, dropped, chunk, part = [], [], [], 0
        manifest = {'chunks': [], 'dropped': dropped}
        def flush():
            nonlocal chunk, part
            if not chunk:
                return
            path = out / f'arrays_{part:03d}.npz'
            np.savez_compressed(path, gravityspy_id=np.array([g for g, _ in chunk]), image=np.stack([a for _, a in chunk]))
            manifest['chunks'].append({'file': path.name, 'events': len(chunk), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            chunk = []; part += 1
        for i, row in rows.iterrows():
            p = out / 'images' / f"{row['gravityspy_id']}_{DURATION}.png"
            if not p.exists():
                dropped.append({'gravityspy_id': row['gravityspy_id'], 'reason': 'missing_png'}); continue
            try:
                Image.open(p).verify(); a = o3.png_to_array(p)
            except Exception as e:
                dropped.append({'gravityspy_id': row['gravityspy_id'], 'reason': f'decode_error: {e}'}); continue
            if float(a.max() - a.min()) <= 1e-6:
                dropped.append({'gravityspy_id': row['gravityspy_id'], 'reason': 'blank'}); continue
            # uint8 is exactly what the scoring pipeline consumes (to_uint8_rgb quantizes to 8 bits).
            chunk.append((row['gravityspy_id'], np.clip(np.rint(a * 255), 0, 255).astype(np.uint8))); ids.append(row['gravityspy_id'])
            if len(chunk) == CHUNK:
                flush()
            if (i + 1) % 5000 == 0:
                print(f'converted {i + 1}/{len(rows)}', flush=True)
        flush()
        manifest.update({'events': len(ids), 'dropped_count': len(dropped), 'blank_rule': 'arrays with max minus min <= 1e-6 are dropped (reason "blank")',
                         'blank_count': sum(d['reason'] == 'blank' for d in dropped), 'dtype': 'uint8 0-255 of the 140x170 gray array', 'duration': DURATION})
        (out / 'arrays.json').write_text(json.dumps(manifest, indent=1))
        print(f'{len(ids)} events in {part} chunks, {len(dropped)} dropped')
        return


if __name__ == '__main__':
    main()
