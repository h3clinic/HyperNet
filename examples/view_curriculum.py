"""Portable localhost browser for the published 2000-step, seed73 v09 curriculum."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from collections import OrderedDict
from threading import Lock
from io import BytesIO
import argparse,base64,hashlib,json,sys,zipfile
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hypernet.procedural_shapes import procedural_triplets,FAMILIES
from hypernet.procedural_pretraining import transformed_triplets
from hypernet.training_v4 import corrupt_same_geometry

def png(tensor,mask=False):
    a=tensor.detach().cpu().numpy()
    a=a[0] if mask else np.moveaxis(a,0,-1)
    stream=BytesIO();Image.fromarray(np.uint8(np.rint(np.clip(a,0,1)*255))).save(stream,format='PNG')
    return stream.getvalue()

def corruption_states(steps,n,seed):
    """Skip only random draws, preserving exact original corruption RNG states."""
    g=torch.Generator().manual_seed(seed);states=[g.get_state().clone()]
    for _ in range(steps):
        for view in range(2):
            for shape in ((n,1,1,1),(n,1,1,1),(n,3,1,1),(n,1,1,1),(n,1,1,1)):
                torch.rand(shape,generator=g,dtype=torch.float32)
            torch.randn((n,3,32,32),generator=g,dtype=torch.float32)
        states.append(g.get_state().clone())
    # Independent checks against the actual corruption function, not only skip code.
    for index in (0,min(19,steps-1),steps-1):
        g.set_state(states[index]);dummy=torch.zeros(n,3,32,32)
        corrupt_same_geometry(dummy,g);corrupt_same_geometry(dummy,g)
        assert torch.equal(g.get_state(),states[index+1])
    return states

HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All HyperNet training images</title><style>
body{margin:0;background:#eef3f7;color:#142b3b;font:16px/1.5 system-ui}header{background:#142b3b;color:white;padding:24px 4vw}h1{margin:0;font-size:32px}header p{margin:8px 0}.controls{position:sticky;top:0;z-index:3;background:#fff;padding:14px 4vw;border-bottom:1px solid #bdcbd6;display:flex;gap:12px;flex-wrap:wrap;align-items:center}input,select,button,a.button{font:inherit;padding:8px;border:1px solid #8aa3b5;border-radius:6px;background:white;color:#153d58}button{cursor:pointer}button:disabled{opacity:.4}input[type=number]{width:100px}input[type=range]{width:200px}main{max-width:1150px;margin:auto;padding:20px}.status{padding:12px 0}.triplet{background:white;border:1px solid #d6e0e7;border-radius:10px;padding:16px;margin:16px 0}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{margin:0}.card img{width:100%;max-width:280px;aspect-ratio:1;image-rendering:pixelated;cursor:zoom-in;background:#eee}.card figcaption{font-size:13px;margin-top:5px}.meta{color:#4f6677;font-size:13px}dialog{border:0;border-radius:12px;max-width:94vw}dialog img{width:min(512px,80vw);image-rendering:pixelated}dialog::backdrop{background:#0009}.footer{display:flex;gap:16px;padding:24px 0}a{color:#145f99}.error{color:#b11}.notice{background:#e1edf7;padding:16px;border-radius:8px}#loading{font-weight:600}@media(max-width:650px){.images{gap:7px}.triplet{padding:9px}h1{font-size:25px}}
</style><header><h1>All HyperNet procedural training images</h1><p><b>Every one of the 2,000 training batches.</b> No sample filtering. Both synthetic backbones used this same sequence.</p><p>96,000 generated RGB images · 96,000 Gaussian target views · 192,000 actual corrupted CNN input views</p></header>
<div class="controls"><button id="prev">← Previous</button><label>Batch <input id="batch" type="number" min="1" max="2000" value="1"> / 2,000</label><button id="go">Go</button><button id="next">Next →</button><input id="slider" type="range" min="1" max="2000" value="1"><label>View <select id="view"><option value="raw">Raw generated RGB</option><option value="clean" selected>Gaussian target</option><option value="first">Actual CNN input 1</option><option value="second">Actual CNN input 2</option><option value="mask">Procedural foreground mask</option></select></label></div>
<main><div class="notice">Each page contains all <b>16 triplets / 48 images</b> from that training batch. Change the view selector to inspect the raw generation, Gaussian transformation, both exact corrupted training inputs, or their masks. These are 32 × 32 pixels, enlarged; click any image for a larger view. Pages are regenerated from the frozen seeds, so every image is accessible without storing hundreds of thousands of PNG files.</div>
<p><label>Jump to RGB image ID <input id="image-id" type="number" min="1" max="96000" value="1"></label> <button id="image-go">Find</button> <a id="download" href="#">Download this page as PNGs</a></p><div class="status"><span id="loading">Loading batch…</span> <span id="count"></span></div><div id="images"></div><div class="footer"><button id="prev-bottom">← Previous batch</button><button id="next-bottom">Next batch →</button></div><p class="meta">All pixels come from the procedural generator. No natural photo, font, pretrained image model, or downloaded asset generated these images. The foreground masks are known from the drawing programs and are not supplied for real query photos. Exact corruption RNG replay was checked against the training function. Inputs are regenerated as float32; displayed/downloaded PNGs use 8-bit RGB.</p></main><dialog id="zoom"><button id="close">Close</button><p id="zoom-title"></p><img id="zoom-image"></dialog>
<script>
const $=id=>document.getElementById(id);let current=0,ticket=0;const roles=['Generated reference','Same structure / new appearance','Changed structure or texture'];
function clamp(n){return Math.max(0,Math.min(1999,n));}
async function show(step){current=clamp(step);$('batch').value=current+1;$('slider').value=current+1;const v=$('view').value,job=++ticket;$('loading').textContent='Generating exact batch '+(current+1)+'…';$('loading').className='';$('prev').disabled=current===0;$('next').disabled=current===1999;$('download').href='/download/'+current+'?view='+v;history.replaceState(null,'','#batch='+(current+1)+'&view='+v);
try{const response=await fetch('/api/batch/'+current+'?view='+v);if(!response.ok)throw Error(await response.text());const data=await response.json();if(job!==ticket)return;$('images').replaceChildren();data.triplets.forEach((t,i)=>{let section=document.createElement('section');section.className='triplet';let title=document.createElement('h3');title.textContent='Triplet '+(i+1)+' · '+t.family+' · negative change: '+t.mutation;section.append(title);let grid=document.createElement('div');grid.className='images';t.images.forEach((item,k)=>{let card=document.createElement('figure');card.className='card';let image=document.createElement('img');image.src=item.png;image.alt=roles[k]+'; image '+item.id;image.onclick=()=>{$('zoom-image').src=item.png;$('zoom-title').textContent='Image '+item.id+' · '+roles[k]+' · '+v;$('zoom').showModal()};let caption=document.createElement('figcaption');caption.textContent=roles[k]+' · ID '+item.id;card.append(image,caption);grid.append(card)});section.append(grid);$('images').append(section)});$('loading').textContent='Batch '+(current+1)+' ready.';$('count').textContent='48 / 96,000 image IDs in this view; generation seed '+data.seed+'.';}catch(e){if(job===ticket){$('loading').textContent=e.message;$('loading').className='error'}}}
$('go').onclick=()=>show(Number($('batch').value)-1);$('batch').onkeydown=e=>{if(e.key==='Enter')$('go').click()};$('prev').onclick=()=>show(current-1);$('next').onclick=()=>show(current+1);$('prev-bottom').onclick=()=>{show(current-1);window.scrollTo(0,0)};$('next-bottom').onclick=()=>{show(current+1);window.scrollTo(0,0)};$('view').onchange=()=>show(current);$('slider').onchange=()=>show(Number($('slider').value)-1);$('image-go').onclick=()=>show(Math.floor((Number($('image-id').value)-1)/48));$('close').onclick=()=>$('zoom').close();const initial=new URLSearchParams(location.hash.slice(1));if(['raw','clean','first','second','mask'].includes(initial.get('view')))$('view').value=initial.get('view');show((Number(initial.get('batch'))||1)-1);
</script></html>'''

class Curriculum:
    def __init__(self):
        snapshot=json.loads((ROOT/'publication/source_snapshot.json').read_text())
        if not all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in snapshot['core_files'].items()):
            raise ValueError('Package differs from the published source snapshot')
        control=json.loads((ROOT/'publication/procedural_control.json').read_text())
        meta=control['models']['detail_metric']
        if (meta['steps'],meta['triplets_per_step'],meta['seed']) != (2000,16,73):
            raise ValueError('This viewer replays only the published v09 control curriculum')
        self.states=corruption_states(2000,48,75)
        self.cache=OrderedDict();self.lock=Lock()
    def batch(self,step):
        if not 0<=step<2000:raise ValueError('Batch must be between0and1999')
        with self.lock:
            if step in self.cache:self.cache.move_to_end(step);return self.cache[step]
            data=procedural_triplets(16,100073+step);clean,masks=transformed_triplets(data,200073+step)
            g=torch.Generator();g.set_state(self.states[step])
            first=corrupt_same_geometry(clean,g);second=corrupt_same_geometry(clean,g)
            assert torch.equal(g.get_state(),self.states[step+1])
            result={'raw':data['images'],'clean':clean,'first':first,'second':second,'mask':masks,'records':data['records']}
            self.cache[step]=result
            while len(self.cache)>4:self.cache.popitem(last=False)
            return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=0);parser.add_argument('--threads',type=int,default=2);args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    torch.set_num_threads(args.threads);curriculum=Curriculum()
    class Handler(BaseHTTPRequestHandler):
        def send(self,body,content_type='application/json',status=200):
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            u=urlparse(self.path)
            try:
                if u.path in ('/','/gallery'):return self.send(HTML.encode(),'text/html; charset=utf-8')
                prefix='/api/batch/' if u.path.startswith('/api/batch/') else '/download/' if u.path.startswith('/download/') else None
                if prefix is None:return self.send(b'Not found','text/plain',404)
                step=int(u.path[len(prefix):]);view=parse_qs(u.query).get('view',['clean'])[0]
                if view not in ('raw','clean','first','second','mask'):raise ValueError('Unknown view')
                batch=curriculum.batch(step);images=batch[view];encoded=[png(t,view=='mask') for t in images]
                if prefix=='/download/':
                    buf=BytesIO()
                    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
                        for i,b in enumerate(encoded):archive.writestr(f'batch{step+1:04d}_{view}_image{step*48+i+1:06d}.png',b)
                    return self.send(buf.getvalue(),'application/zip')
                rows=[]
                for i,record in enumerate(batch['records']):
                    rows.append({'family':FAMILIES[record['program']['family']],'mutation':record['mutation'],
                        'images':[{'id':step*48+j+1,'png':'data:image/png;base64,'+base64.b64encode(encoded[j]).decode()} for j in (i,i+16,i+32)]})
                return self.send(json.dumps({'batch':step+1,'seed':100073+step,'view':view,'triplets':rows}).encode())
            except Exception as error:return self.send(str(error).encode(),'text/plain',400)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);url=f'http://127.0.0.1:{server.server_port}/'
    print(url,flush=True);server.serve_forever()
if __name__=='__main__':main()
