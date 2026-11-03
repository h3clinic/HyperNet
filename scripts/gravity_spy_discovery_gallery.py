"""Gallery of discovery candidates: the first N of each type's primary-arm list, never a selection.

Each tile shows the pool event's 0.5 s spectrogram (the image the search scored),
its rank and score, the O3b CNN label and confidence, and the volunteer status.
References are shown first. Images are read from the local fetch folder; the
page is local only (the PNGs are the publisher's and are not redistributed).

    python scripts/gravity_spy_discovery_gallery.py --pool <folder> --data <v1.1> --output <benchmark folder> --per-type 20
"""
from __future__ import annotations
from pathlib import Path
import argparse, base64, html, io, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from hypernet import gravity_spy as gs
from hypernet import gravity_spy_o3 as o3
from prepare_gravity_spy_discovery import RARE, DURATION


def tile_png(array):
    from PIL import Image
    im = Image.fromarray(np.clip(np.rint(array * 255), 0, 255).astype(np.uint8)).resize((255, 210), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, format='PNG'); return base64.b64encode(buf.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', type=Path, required=True); parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--per-type', type=int, default=20)
    args = parser.parse_args()
    import pandas as pd
    protocol = json.loads((args.output / 'protocol.json').read_text()); report = json.loads((args.output / 'verification.json').read_text())
    cand = pd.read_csv(args.output / 'candidates.csv')
    meta = pd.read_csv(args.data / 'trainingset_v1d1_metadata.csv').set_index('gravityspy_id')
    import h5py
    f = h5py.File(args.data / 'trainingsetv1d1.h5', 'r')
    sections = []
    for rare in RARE:
        refs = ''.join(f'<figure><img src="data:image/png;base64,{tile_png(f[meta.loc[g, "label"]][meta.loc[g, "sample_type"]][g][f"{DURATION}.png"][()][0])}"><figcaption>reference {i + 1}<br><span>{g}</span></figcaption></figure>'
                       for i, g in enumerate(protocol['reference_gravityspy_ids'][rare]))
        rows = cand[cand['type'] == rare].sort_values('rank').head(args.per_type)
        tiles = []
        for _, r in rows.iterrows():
            p = args.pool / 'images' / f"{r['gravityspy_id']}_{DURATION}.png"
            img = tile_png(o3.png_to_array(p)) if p.exists() else ''
            cls = {'volunteer majority': 'hit', 'volunteer, not majority': 'weak', 'no volunteer data': 'none'}[r['status']]
            tiles.append(f'<figure class="{cls}"><img src="data:image/png;base64,{img}"><figcaption>#{int(r["rank"])} z={r["score"]:.2f}<br>CNN: {html.escape(str(r["o3b_ml_label"]))} ({r["o3b_ml_confidence"]:.2f}), P({rare})={r["o3b_p_type"]:.2f}<br>{html.escape(r["status"])}{" (" + str(int(r["picks_of_type"])) + "/" + str(int(r["volunteers"])) + " picks)" if r["volunteers"] else ""}<br><span>{r["gravityspy_id"]} {r["ifo"]} {r["why"]}</span></figcaption></figure>')
        cell = report['primary_cells'].get(rare)
        head = (f'<p class="cell">Primary cell: hybrid@200 new_majority <b>{cell["hybrid_new_majority"]}</b> ({cell["hybrid_new_majority_groups"]} groups), covered {cell["covered"]}, expected {cell["expected"]}, p={cell["p"]:.4f}; CNN P(type) comparator {cell["cnn_new_majority"]} ({cell["cnn_new_majority_groups"]} groups): <b>{"pass" if cell["passes"] else "fail"}</b></p>'
                if cell else '<p class="cell">Descriptive type (not in the primary rule).</p>')
        pt = report['per_type'][rare]
        sections.append(f'<h2>{rare}</h2>{head}<p>Pool ceiling within the eligible subset: new majority / confirmed / supported = {pt["pool_new"]["majority"]} / {pt["pool_new"]["confirmed"]} / {pt["pool_new"]["supported"]} among {pt["covered_eligible"]} covered events.</p><h3>References (v1.1 train)</h3><div class="row">{refs}</div><h3>First {args.per_type} of the hybrid top-200 (eligible subset), in rank order</h3><div class="row">{"".join(tiles)}</div>')
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Gravity Spy discovery candidates</title>
<style>body{{font-family:system-ui;background:#111;color:#ddd;margin:20px;max-width:1500px}} .row{{display:flex;flex-wrap:wrap;gap:8px}} figure{{margin:0;width:255px;border:3px solid #333;background:#000}} figure.hit{{border-color:#3c3}} figure.weak{{border-color:#cc3}} figure.none{{border-color:#555}} figcaption{{font-size:12px;padding:4px;color:#ccc}} figcaption span{{color:#889;font-size:11px}} h2{{margin-top:40px}} p{{max-width:1100px;color:#bbb}} .cell{{color:#eee}} .legend span{{display:inline-block;padding:2px 8px;margin-right:8px;border:3px solid}}</style></head><body>
<h1>Gravity Spy one-reference discovery run: candidates</h1>
<p><b>Verdict under the frozen rule: {report["primary_verdict"]}</b> ({report["primary_types_passing"]} of {len(protocol["primary"]["types"])} primary types pass).</p>
<p>Tiles are the first {args.per_type} entries of each type's frozen primary-arm list within the eligible subset (events the O3b CNN did not call that type), in rank order; nothing is selected after the fact. {report["events_scored"]} pool events were scored; {report["events_with_volunteers"]} have at least one volunteer classification.</p>
<p class="legend"><span style="border-color:#3c3">volunteer majority for the type</span><span style="border-color:#cc3">volunteers, not a majority</span><span style="border-color:#555">no volunteer has classified this event</span></p>
{"".join(sections)}
<p>Volunteer votes are confusion-weighted picks by a median of two people on the same image; they are supporting evidence, not ground truth. Grey-bordered events are unclassified candidates and count as zero finds in every metric.</p>
</body></html>'''
    (args.output / 'gallery.html').write_text(page, encoding='utf-8'); print('wrote', args.output / 'gallery.html')


if __name__ == '__main__':
    main()
