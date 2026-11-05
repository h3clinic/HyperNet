"""Show the four fixed consumed reference events at all authentic durations."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;O=R/'data';W=R.parents[1]
REFS=[('Low Frequency Burst','632xGf8n9x'),('Low Frequency Lines','Vengh4mj2t'),('Scattered Light','dac0D6ni5R'),('Blip','yxumvoPABn')]
VIEWS=[('d05','0.5 seconds'),('d1','1 second'),('d2','2 seconds'),('d4','4 seconds')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=json.loads((O/'plan.json').read_text());assert set(plan['references'].values())=={v for _,v in REFS}
 path=W/'outputs/hypernet_v41/preprocess.py';assert plan['source_hashes'][str(path)]==sha(path)
 spec=importlib.util.spec_from_file_location('montage_frozen_preprocess',path);prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
 crop_w,crop_h=571,477;pad=18;head=115;rowhead=58;foot=85;width=4*crop_w+5*pad;height=head+4*(rowhead+crop_h+pad)+foot
 canvas=Image.new('RGB',(width,height),'#f2f5fa');draw=ImageDraw.Draw(canvas)
 font_path=Path('C:/Windows/Fonts/segoeui.ttf');bold_path=Path('C:/Windows/Fonts/segoeuib.ttf')
 font=lambda size,bold=False:ImageFont.truetype(str(bold_path if bold else font_path),size)
 draw.text((pad,12),'The same reference event at four genuine time windows',fill='#172b49',font=font(33,True))
 for col,(_,label) in enumerate(VIEWS):draw.text((pad+col*(crop_w+pad),68),label,fill='#172b49',font=font(28,True))
 sources=[]
 for row,(target,event) in enumerate(REFS):
  y=head+row*(rowhead+crop_h+pad);draw.text((pad,y+8),target+'  |  event '+event,fill='#172b49',font=font(27,True))
  for col,(key,label) in enumerate(VIEWS):
   binding_path=O/'source_bindings'/key/(event+'.json');b=json.loads(binding_path.read_text());assert b['event_id']==event and b['sha256']==sha(b['path'])
   with Image.open(b['path']) as im:raw=np.asarray(im.convert('RGB'))
   prep.validate_renderer(raw);crop=prep.native_rgb(raw,'full');assert crop.shape==(crop_h,crop_w,3)
   canvas.paste(Image.fromarray(crop),(pad+col*(crop_w+pad),y+rowhead))
   sources.append(dict(target=target,event_id=event,duration_key=key,duration_seconds=b['duration_seconds'],raw_path=b['path'],raw_sha256=b['sha256'],binding_sha256=sha(binding_path)))
 draw.text((pad,height-foot+10),'One labeled reference event per row; four rendered windows provide more context than one PNG.',fill='#34435b',font=font(25))
 draw.text((pad,height-foot+44),'Native plot pixels; axes and timestamps removed with the fixed crop. No reserved images.',fill='#34435b',font=font(24))
 output=R/'reference_windows.png';canvas.save(output)
 (R/'reference_windows.json').write_text(json.dumps(dict(image_sha256=sha(output),code_sha256=sha(__file__),plan_sha256=sha(O/'plan.json'),preprocess_sha256=sha(path),native_crop=prep.PLOT_CROP,sources=sources,reference_events=4,rendered_windows=16,no_visual_selection=True,no_new_or_reserved_events=True),indent=2))
 print('REFERENCE MONTAGE',str(output))
if __name__=='__main__':main()
