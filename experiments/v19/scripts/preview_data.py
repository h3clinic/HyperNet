"""A small labeled gallery of actual deterministic procedural outputs."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet_v19.curriculum import episode


def main():
    choices = [('train', 0), ('train', 1), ('train', 2), ('dev', 0), ('dev', 1), ('compositional', 0)]
    canvas = Image.new('RGB', (824, 900), '#10151a')
    draw = ImageDraw.Draw(canvas)
    records = []
    for j, (split, index) in enumerate(choices):
        row = episode(index, split)
        x = (j % 2)*412; y = (j//2)*300
        draw.text((x+8, y+5), f'{split} {index} | target '+('present' if row['positive'] else 'absent'), fill='#dce8ee')
        draw.text((x+8, y+24), 'Reference', fill='#a9bdc9')
        draw.text((x+148, y+24), 'Independent scene', fill='#a9bdc9')
        canvas.paste(Image.fromarray(row['support']), (x+8, y+44))
        query = Image.fromarray(row['query'])
        if row['positive']:
            ImageDraw.Draw(query).rectangle(row['target_box'].tolist(), outline='#77ffc9', width=1)
        canvas.paste(query, (x+148, y+44))
        records.append({'split': split, 'index': index, 'program_id': row['program_id'],
                        'positive': row['positive'], 'image_sha256': row['metadata']['image_sha256'],
                        'source': 'Generated inspection example; not evidence of optimizer consumption'})
    asset = ROOT/'assets/curriculum_preview.png'; asset.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(asset)
    meta = {'source_sha256': hashlib.sha256((ROOT/'hypernet_v19/curriculum.py').read_bytes()).hexdigest(),
            'preview_sha256': hashlib.sha256(asset.read_bytes()).hexdigest(), 'examples': records,
            'annotations': 'Green boxes are procedural ground truth for inspection, never RGB training inputs'}
    (ROOT/'assets/curriculum_preview.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({'preview': str(asset), 'source_sha256': meta['source_sha256']}))


if __name__ == '__main__': main()
