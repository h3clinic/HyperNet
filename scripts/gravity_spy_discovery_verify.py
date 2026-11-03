"""Check discovery rankings against volunteer labels (Zenodo 5911227).

Truth is volunteer-only. In the volunteer file, `tracks[0]` is the 22-class
Gravity Spy CNN prior that ran the Zooniverse workflow and each later row is
one registered volunteer's confusion-weighted vote (checked on 3,000 sampled
events: row 0 equals the ML probability columns in 100%, rows equal
Nclassifications + 1 in 99.9%, volunteer-only argmax equals the published
consensus in 100% of retired events). A volunteer "pick" is the argmax of that
volunteer's row. Rows with no volunteer (prior only) count as uncovered.

Two different CNNs label the pool: the O3b tables (Zenodo 5649212; a later
model with Fast_Scattering and Blip_Low_Frequency) and the volunteer file (the
22-class model the volunteers saw). They agree on only 41% of covered pool
events. "new" therefore requires BOTH labels to differ from the type, and the
share whose O3b label is Fast_Scattering or Blip_Low_Frequency (classes the
volunteers had no button for) is reported as a vocabulary gap.

Truth tiers for an event with n volunteers and p picks of the type:
  majority   p >= 2 and p > n/2            (primary)
  confirmed  p >= 2                        (descriptive)
  supported  p >= 1 and p >= ceil(n/2)     (descriptive; with n <= 2 this is one volunteer)

Evaluated subset per type: pool events whose O3b ml_label != type, for the
arms and the CNN P(type) comparator alike; the full pool is secondary. Each
ranking's own covered count sets a hypergeometric null. Counts are given per
event and per 4 s time group. Top-K events with no volunteer data are exported
as unclassified candidates and count as zero finds.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_gravity_spy_discovery import RARE
from gravity_spy_benchmark import sha256

CLASSES = ['1080Lines', '1400Ripples', 'Air_Compressor', 'Blip', 'Chirp', 'Extremely_Loud', 'Helix', 'Koi_Fish', 'Light_Modulation',
           'Low_Frequency_Burst', 'Low_Frequency_Lines', 'No_Glitch', 'None_of_the_Above', 'Paired_Doves', 'Power_Line', 'Repeating_Blips',
           'Scattered_Light', 'Scratchy', 'Tomte', 'Violin_Mode', 'Wandering_Line', 'Whistle']
VOLUNTEER_ZENODO = '5911227'
VOLUNTEER_MD5 = '0f120a57897c7d00c7a5d6813c2a0f17'
KS = (20, 50, 200)
TIERS = ('majority', 'confirmed', 'supported')
VOCAB_GAP = ('Fast_Scattering', 'Blip_Low_Frequency')


def declaration():
    return {'volunteer_zenodo': VOLUNTEER_ZENODO, 'volunteer_md5': VOLUNTEER_MD5, 'classes': CLASSES, 'ks': list(KS),
            'truth_tiers': {'majority': 'p >= 2 and p > n/2 (primary)', 'confirmed': 'p >= 2', 'supported': 'p >= 1 and p >= ceil(n/2); single volunteer when n <= 2'},
            'new_rule': 'tier holds AND O3b-table ml_label != type AND volunteer-file ml_label != type; vocabulary-gap share (O3b label in Fast_Scattering/Blip_Low_Frequency) reported',
            'covered': 'event has at least one volunteer row; prior-only rows are uncovered',
            'evaluated_subset': 'per type, pool events with O3b ml_label != type; full pool secondary',
            'null': 'hypergeometric on each ranking: N = covered eligible pool events, M = pool-wide new_<tier> count, n = covered events in the top-K, x = observed',
            'tie_rule': 'descending score, stable sort, ties by roster order; CNN comparator sorted the same way on P(type)',
            'groups': 'counts also given per 4 s time group (pool_roster.csv group column)',
            'uncovered': 'top-K events without volunteer data are exported as candidates and count as zero finds'}


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def volunteer_picks(hdf5, ids):
    """gravityspy_id -> (n volunteers, picks per class, volunteer-file ml_label); prior-only rows skipped."""
    import pandas as pd
    d = pd.read_hdf(hdf5, key='image_db').drop_duplicates('gravityspy_id').set_index('gravityspy_id')
    d = d[d.index.isin(set(ids))]
    out = {}
    for gid, row in d.iterrows():
        tr = np.asarray(row['tracks'], float)
        if tr.shape[0] <= 1:
            continue
        out[str(gid)] = (int(tr.shape[0] - 1), np.bincount(tr[1:].argmax(1), minlength=len(CLASSES)), str(row['ml_label']))
    return out


def judge(picks, n, rare):
    p = int(picks[CLASSES.index(rare)])
    return {'majority': p >= 2 and p > n / 2, 'confirmed': p >= 2, 'supported': p >= 1 and p >= int(np.ceil(n / 2))}


def tally(ids, vol, rare, pool_ml, pool_group):
    out = {'k': len(ids), 'covered': 0, 'uncovered_ids': [], 'covered_groups': set()}
    for t in TIERS:
        out[t] = 0; out[f'new_{t}'] = 0; out[f'new_{t}_groups'] = set(); out[f'vocab_gap_{t}'] = 0
    out['new_majority_ids'] = []
    for gid in ids:
        gid = str(gid)
        if gid not in vol:
            out['uncovered_ids'].append(gid); continue
        n, picks, vml = vol[gid]; out['covered'] += 1; out['covered_groups'].add(pool_group[gid])
        j = judge(picks, n, rare); other = pool_ml[gid] != rare and vml != rare; gap = pool_ml[gid] in VOCAB_GAP
        for t in TIERS:
            out[t] += j[t]
            if j[t] and other:
                out[f'new_{t}'] += 1; out[f'new_{t}_groups'].add(pool_group[gid]); out[f'vocab_gap_{t}'] += gap
        if j['majority'] and other:
            out['new_majority_ids'].append({'gravityspy_id': gid, 'n': n, 'picks': int(picks[CLASSES.index(rare)]), 'o3b_label': pool_ml[gid], 'volunteer_file_label': vml})
    out['covered_groups'] = len(out['covered_groups'])
    for t in TIERS:
        out[f'new_{t}_groups'] = len(out[f'new_{t}_groups'])
    out['uncovered'] = len(out['uncovered_ids'])
    return out


def with_null(t, N, M_by_tier):
    from scipy.stats import hypergeom
    t['null'] = {}
    for tier in TIERS:
        M, n, x = M_by_tier[tier], t['covered'], t[f'new_{tier}']
        t['null'][tier] = {'expected': round(n * M / N, 3) if N else None, 'p_ge_observed': float(hypergeom.sf(x - 1, N, M, n)) if N and n else None,
                           'q95': int(hypergeom.ppf(.95, N, M, n)) if N and n else None}
    return t


def pool_ceilings(picks_csv, roster_csv):
    """Pool-wide new_<tier> counts under the frozen rules, from the pre-freeze picks CSV (both labels required)."""
    import pandas as pd
    v = pd.read_csv(picks_csv); roster = pd.read_csv(roster_csv).set_index('gravityspy_id')
    out = {}
    for rare in RARE:
        p = v[rare].to_numpy(); n = v['n'].to_numpy(); ml_file = v['ml'].to_numpy(); ml_o3b = roster.loc[v['gid'], 'ml_label'].to_numpy()
        other = (ml_o3b != rare) & (ml_file != rare)
        tiers = {'majority': (p >= 2) & (p > n / 2), 'confirmed': p >= 2, 'supported': (p >= 1) & (p >= np.ceil(n / 2))}
        out[rare] = {f'new_{t}': int((m & other & (n >= 1)).sum()) for t, m in tiers.items()}
        out[rare]['covered_with_volunteers'] = int((n >= 1).sum())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', type=Path, required=True); parser.add_argument('--volunteer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd
    if md5(args.volunteer) != VOLUNTEER_MD5:
        raise ValueError('Volunteer file checksum differs from Zenodo')
    if (args.output / 'verification.json').exists():
        raise FileExistsError('Verification already written')
    protocol = json.loads((args.output / 'protocol.json').read_text()); rankings = json.loads((args.output / 'rankings.json').read_text())
    if rankings['protocol_sha256'] != sha256(args.output / 'protocol.json'):
        raise RuntimeError('rankings were not produced under this protocol')
    if protocol['inputs']['verify_script_sha256'] != sha256(Path(__file__)) or protocol['verification'] != declaration():
        raise RuntimeError('verifier changed since the freeze')
    if protocol['inputs']['pool_roster.csv'] != sha256(args.pool / 'pool_roster.csv'):
        raise RuntimeError('pool roster changed since the freeze')
    scores = np.load(args.output / 'scores.npz'); ids_all = np.array([str(g) for g in scores['gravityspy_id']])
    pool = pd.read_csv(args.pool / 'pool_roster.csv').set_index('gravityspy_id')
    pool_ml = pool['ml_label'].to_dict(); pool_group = pool['group'].to_dict()
    vol = volunteer_picks(args.volunteer, set(ids_all))
    primary = protocol['primary']; report = {'format': 'hypernet.gravity_spy_discovery_verification.v0.2', 'protocol_sha256': sha256(args.output / 'protocol.json'),
                                             'verify_script_sha256': sha256(Path(__file__)), 'rankings_sha256': sha256(args.output / 'rankings.json'),
                                             'events_scored': int(len(ids_all)), 'events_with_volunteers': len(vol),
                                             'volunteers_per_covered_event': {str(k): int(v) for k, v in zip(*np.unique([vol[g][0] for g in vol], return_counts=True))},
                                             'per_type': {}, 'primary_cells': {}}
    candidates = []
    for rare in RARE:
        elig = np.array([pool_ml[g] != rare for g in ids_all]); elig_ids = ids_all[elig]
        covered_elig = [g for g in elig_ids if g in vol]
        ceiling = tally(covered_elig, vol, rare, pool_ml, pool_group); M = {t: ceiling[f'new_{t}'] for t in TIERS}; N = len(covered_elig)
        entry = {'eligible_events': int(elig.sum()), 'covered_eligible': N, 'pool_new': {t: M[t] for t in TIERS}, 'pool_new_groups': {t: ceiling[f'new_{t}_groups'] for t in TIERS},
                 'arms': {}, 'cnn_probability': {}, 'full_pool_secondary': {}}
        cnn_order = np.argsort(-pool.loc[elig_ids, rare].to_numpy(), kind='stable')
        for k in KS:
            for arm in rankings['rankings']:
                s = scores[f'{arm}|{rare}'][elig]; top = elig_ids[np.argsort(-s, kind='stable')[:k]]
                entry['arms'].setdefault(arm, {})[str(k)] = with_null(tally(top, vol, rare, pool_ml, pool_group), N, M)
            entry['cnn_probability'][str(k)] = with_null(tally(elig_ids[cnn_order[:k]], vol, rare, pool_ml, pool_group), N, M)
            # full-pool secondary: the frozen rankings as written (they include CNN-labeled events of the type)
            entry['full_pool_secondary'][str(k)] = {arm: {t: tally([h['gravityspy_id'] for h in rankings['rankings'][arm][rare][:k]], vol, rare, pool_ml, pool_group)[f'new_{t}'] for t in TIERS}
                                                    for arm in rankings['rankings']}
        # candidates: primary arm, primary K, eligible subset
        s = scores[f'{primary["arm"]}|{rare}'][elig]; top = elig_ids[np.argsort(-s, kind='stable')[:primary['k']]]
        for rank, gid in enumerate(top, 1):
            status = 'no volunteer data' if gid not in vol else ('volunteer majority' if judge(vol[gid][1], vol[gid][0], rare)['majority'] else 'volunteer, not majority')
            r = pool.loc[gid]
            candidates.append({'type': rare, 'rank': rank, 'gravityspy_id': gid, 'score': float(s[np.argsort(-s, kind='stable')[rank - 1]]), 'group': int(r['group']),
                               'ifo': r['ifo'], 'event_time': float(r['event_time']), 'o3b_ml_label': r['ml_label'], 'o3b_ml_confidence': float(r['ml_confidence']),
                               'o3b_p_type': float(r[rare]), 'why': r['why'], 'status': status, 'volunteers': vol[gid][0] if gid in vol else 0,
                               'picks_of_type': int(vol[gid][1][CLASSES.index(rare)]) if gid in vol else 0, 'url': r[pool.columns[-1]]})
        report['per_type'][rare] = entry
        if rare in primary['types']:
            cell = entry['arms'][primary['arm']][str(primary['k'])]; comp = entry['cnn_probability'][str(primary['k'])]
            p = cell['null']['majority']['p_ge_observed']
            report['primary_cells'][rare] = {'hybrid_new_majority': cell['new_majority'], 'hybrid_new_majority_groups': cell['new_majority_groups'], 'covered': cell['covered'],
                                             'expected': cell['null']['majority']['expected'], 'p': p, 'cnn_new_majority_groups': comp['new_majority_groups'], 'cnn_new_majority': comp['new_majority'],
                                             'passes': bool(p is not None and p < primary['alpha'] and cell['new_majority_groups'] > comp['new_majority_groups'])}
    passes = sum(c['passes'] for c in report['primary_cells'].values())
    report['primary_verdict'] = 'POSITIVE' if passes >= 2 else 'not confirmed'; report['primary_types_passing'] = passes
    for e in report['per_type'].values():
        for arm in e['arms'].values():
            for t in arm.values():
                t.pop('uncovered_ids', None); t.pop('new_majority_ids', None)
        for t in e['cnn_probability'].values():
            t.pop('uncovered_ids', None); t.pop('new_majority_ids', None)
    (args.output / 'verification.json').write_text(json.dumps(report, indent=1))
    pd.DataFrame(candidates).to_csv(args.output / 'candidates.csv', index=False)
    print(f"PRIMARY VERDICT: {report['primary_verdict']} ({passes} of {len(primary['types'])} types pass)")
    for rare, c in report['primary_cells'].items():
        print(f"  {rare:14s} hybrid@200 new_majority {c['hybrid_new_majority']} ({c['hybrid_new_majority_groups']} groups; covered {c['covered']}; expected {c['expected']}; p={c['p']:.4f}) vs CNN P(type) {c['cnn_new_majority']} ({c['cnn_new_majority_groups']} groups) -> {'pass' if c['passes'] else 'fail'}")
    print('\nSecondary (descriptive), new counts per tier maj/conf/sup within the eligible subset:')
    print(f"{'type':15s}{'K':>4s} | {'hybrid':>12s} | {'random':>12s} | {'trained':>12s} | {'CNN P(type)':>12s} | pool ceiling")
    for rare, e in report['per_type'].items():
        for k in KS:
            s = str(k); f = lambda t: f"{t['new_majority']}/{t['new_confirmed']}/{t['new_supported']}"
            print(f"{rare:15s}{k:4d} | {f(e['arms']['hybrid'][s]):>12s} | {f(e['arms']['random'][s]):>12s} | {f(e['arms']['trained'][s]):>12s} | {f(e['cnn_probability'][s]):>12s} | {e['pool_new']['majority']}/{e['pool_new']['confirmed']}/{e['pool_new']['supported']}")


if __name__ == '__main__':
    main()
