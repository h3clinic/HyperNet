"""First-six diagnostic retrieval illustration from saved scores, with RGB replay."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from hypernet.variation_curriculum import render

RESULTS=ROOT/'experiments/synthetic_final_diagnosis'
ROWS=list(range(6))
ARMS=['random','primary_final_4096','no_relation_aux_final_4096','raw_gray_cosine']


def main():
    torch.set_num_threads(4)
    result=json.loads((RESULTS/'results.json').read_text())
    saved={name:dict(np.load(RESULTS/f'{name}_scores.npz'))for name in ARMS}
    source=saved[ARMS[0]]
    reference_indices=sorted(set([int(source['truth'][i])for i in ROWS]+
                                [int(saved[arm]['prediction'][i])for arm in ARMS for i in ROWS]))
    ref_batch=render(source['reference_codes'][reference_indices],source['reference_seeds'][reference_indices],96,'cuda')['rgb'].cpu().numpy()
    refs={i:image for i,image in zip(reference_indices,ref_batch)}
    query_batch=render(source['query_codes'][ROWS],source['query_seeds'][ROWS],96,'cuda')['rgb'].cpu().numpy()
    verified=[]
    for kind,index,image in ([('reference',i,refs[i])for i in reference_indices]+
                             [('query',i,image)for i,image in zip(ROWS,query_batch)]):
        actual=hashlib.sha256(image.tobytes()).hexdigest()
        if actual!=result['input_rgb_hashes'][kind][index]:raise RuntimeError('Renderer replay differs from scored input')
        verified.append({'kind':kind,'index':index,'rgb_sha256':actual})
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(font_path),17)if font_path.exists()else ImageFont.load_default()
    small=ImageFont.truetype(str(font_path),14)if font_path.exists()else ImageFont.load_default()
    cell=204;top=104;row_height=234
    canvas=Image.new('RGB',((len(ARMS)+2)*cell+12,top+len(ROWS)*row_height+46),'#111827')
    draw=ImageDraw.Draw(canvas)
    draw.text((12,10),'Heldout mathematical structures: first six queries, fixed before results',font=font,fill='white')
    draw.text((12,36),'Each reference and query is a distinct deformed/material instance. No semantic category claim.',font=small,fill='#d1d5db')
    headers=['Query instance','True one-image reference','Random CNN retrieved','Primary CNN retrieved','No-relation CNN retrieved','Raw gray cosine retrieved']
    for col,title in enumerate(headers):draw.text((8+col*cell,72),title,font=small,fill='white')
    selections=[]
    for row,i in enumerate(ROWS):
        truth=int(source['truth'][i]);chosen=[int(saved[arm]['prediction'][i])for arm in ARMS]
        images=[query_batch[row],refs[truth],*[refs[index]for index in chosen]]
        codes=[int(source['query_codes'][i]),int(source['reference_codes'][truth]),
               *[int(source['reference_codes'][index])for index in chosen]]
        for col,(image,code)in enumerate(zip(images,codes)):
            x=8+col*cell;y=top+row*row_height
            canvas.paste(Image.fromarray(image).resize((192,192),Image.Resampling.NEAREST),(x,y))
            correct=col<2 or chosen[col-2]==truth
            color='#9fe7bc'if correct else '#ffafaf'
            draw.text((x,y+197),f'type {code:06x}'+(''if col<2 else (' / correct'if correct else ' / wrong')),font=small,fill=color)
        selections.append({'query_row':i,'true_reference_index':truth,'predicted_reference_indices':dict(zip(ARMS,chosen))})
    draw.text((12,top+len(ROWS)*row_height+8),'Illustration only; complete metrics use all 1,024 queries against 512 references.',font=small,fill='#d1d5db')
    asset=ROOT/'assets/synthetic_retrieval_first_six.png';asset.parent.mkdir(parents=True,exist_ok=True);canvas.save(asset)
    record={'selection':'First six query rows in sorted heldout signature order; declared before diagnostic results',
            'query_rows':ROWS,'selections':selections,'verified_rgb_replays':verified,
            'new_model_inference':False,'results_sha256':hashlib.sha256((RESULTS/'results.json').read_bytes()).hexdigest(),
            'image_sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}
    (RESULTS/'first_six_illustration.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'asset':str(asset),'verified_replays':len(verified),'image_sha256':record['image_sha256']}))


if __name__=='__main__':main()
