"""Deterministic mathematical patch quartets for reference-conditioned ranking.

The v17 renderer is vendored byte-for-byte and never modified. A positive and a
one-attribute counterfactual share their entire scene, nuisance draw, crop, pad,
and resize. Only the target program changes. The fourth image is a reference
for a different, absent program; its training comparison uses the SAME positive
patch. No image is copied from a natural-image dataset.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from .vendor_v17 import scene_curriculum as sc

FORMAT = 'hypernet.pair_curriculum.v18.1'
V17_SOURCE_SHA256 = 'dedea89ccd079f9864923d933626a55fa855390ed5fe2e3030b4d7bb8fde4b6c'
IMAGE_KEYS = ('support', 'positive', 'near_negative', 'wrong_support')
CONFIG = {'format': FORMAT, 'size': 96, 'query_size': 192, 'default_seed': 1801,
          'frozen_generator_sha256': V17_SOURCE_SHA256, 'max_nuisance_attempts': 32,
          'minimum_final_changed_rgb_pixels': 8, 'changed_channel_threshold': 3,
          'crop_margins': [0., .05, .15, .3], 'jitter_as_fraction_of_margin': .75,
          'crop_rule': 'Same crop enclosing both amodal objects and every changed RGB pixel, plus1pixel guard; identical border RGB padding and aspect-preserving resize; fixed tight-union/full-slot fallbacks',
          'wrong_reference_rule': 'Different macro family, absent from every complete and contained scene program; same reference nuisance/background draw',
          'compositional_wrong_reference_exception': 'All programs are composite; use a different ordered constituent-family pair',
          'positive_pair': ['support', 'positive'], 'near_negative_pair': ['support', 'near_negative'],
          'wrong_family_pair': ['wrong_support', 'positive'],
          'natural_training_images': 0, 'borrowed_pretrained_weights': False}


def config_hash():
    return hashlib.sha256(json.dumps(CONFIG,sort_keys=True,separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def verify_frozen_source():
    actual=hashlib.sha256(Path(sc.__file__).read_bytes()).hexdigest()
    if actual!=V17_SOURCE_SHA256:
        raise RuntimeError('Vendored v17 generator differs from its frozen source')
    return actual


def _program_tree_ids(program):
    result={sc.program_id(program)}
    for child in program.get('parts', []):result.update(_program_tree_ids(child))
    return result


def _scene_programs(row, target, negative, replacement, split, seed):
    """Resolve each recorded instance, including accepted occluders, exactly."""
    index=row['metadata']['index']
    candidates=[target,negative,replacement]
    for j in range(1,6):
        p=sc.sample_program(index*13+j+104729,split,seed)
        candidates.append(negative if sc.contains_program(p,target)else p)
    p=sc.sample_program(index*17+1+982451653,split,seed)
    candidates.append(negative if sc.contains_program(p,target)else p)
    by_id={sc.program_id(p):p for p in candidates}
    return [by_id[r['program_id']]for r in row['metadata']['instances']]


def _different_family(candidate,target,split):
    if split!='compositional':return candidate['family']!=target['family']
    return tuple(p['family']for p in candidate['parts'])!=tuple(p['family']for p in target['parts'])


def _wrong_program(index,split,seed,target,scene):
    for offset in range(4096):
        candidate_index=1000003+index*4099+offset
        candidate=sc.sample_program(candidate_index,split,seed)
        if (_different_family(candidate,target,split)
            and not sc.contains_program(candidate,target)
            and all(not sc.contains_program(p,candidate)for p in scene)):
            return candidate,candidate_index
    raise RuntimeError('Could not find an absent different-family reference program')


def _render_reference(index,split,seed,attempt,program,size):
    # Reproduce the exact reference-stage draw order in the frozen renderer.
    values=[seed,index,sc.SPLIT_NUISANCE_CODES[split],419]
    if attempt:values.append(attempt)
    rng=sc._rng(*values);instance_seed=int(rng.integers(1,2**31));rng.integers(1,2**31)
    instance=sc.render_instance(program,instance_seed,size)
    rgb=sc._background(rng,size);color=sc._color(rng,rgb);sc._paint(rgb,instance,0,0,color)
    framed,_,frame=sc._frame_support(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8),instance['parts'],rng,size)
    return framed,frame


def _fit_pair(positive,negative,box,size):
    x1,y1,x2,y2=map(int,box);a=positive[y1:y2,x1:x2];b=negative[y1:y2,x1:x2]
    if a.shape!=b.shape or min(a.shape[:2])<1:raise ValueError('Invalid shared crop')
    h,w=a.shape[:2];scale=size/max(h,w);rh=max(1,round(h*scale));rw=max(1,round(w*scale))
    mode=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR
    # The crop guard keeps all edited pixels off the border. This is exactly
    # the per-image median-border padding used by frozen v17 prepare_image,
    # while also preventing a counterfactual label cue in letterbox pixels.
    borders=[np.concatenate([x[0],x[-1],x[:,0],x[:,-1]],axis=0)for x in (a,b)]
    if not np.array_equal(borders[0],borders[1]):raise ValueError('Counterfactual crop borders must be identical')
    pad=np.median(borders[0],axis=0).round().astype(np.uint8)
    left=(size-rw)//2;top=(size-rh)//2;images=[]
    for source in (a,b):
        image=np.broadcast_to(pad,(size,size,3)).copy()
        image[top:top+rh,left:left+rw]=cv2.resize(source,(rw,rh),interpolation=mode)
        images.append(image)
    return images,{'crop_xyxy':[int(v)for v in box], 'letterbox_xywh':[left,top,rw,rh],
                   'scale_x':rw/w,'scale_y':rh/h,'pad_rgb':pad.tolist()}


def _crop_options(union_box,slot,seed,index,split,attempt):
    x1,y1,x2,y2=map(int,union_box);w=x2-x1;h=y2-y1
    rng=sc._rng(seed,index,sc.SPLIT_NUISANCE_CODES[split],attempt,1819)
    margin=float(rng.choice(CONFIG['crop_margins']))
    dx=max(0,int(round(w*margin)));dy=max(0,int(round(h*margin)))
    jx=int(round(rng.uniform(-.75,.75)*dx));jy=int(round(rng.uniform(-.75,.75)*dy))
    expanded=[max(0,min(x1,x1-dx+jx)),max(0,min(y1,y1-dy+jy)),
              min(192,max(x2,x2+dx+jx)),min(192,max(y2,y2+dy+jy))]
    boxes=[('jittered_union',expanded),('tight_union',[x1,y1,x2,y2]),('full_slot',slot)]
    result=[];seen=set()
    for name,box in boxes:
        if tuple(box)not in seen:result.append((name,box));seen.add(tuple(box))
    return result,margin


def pair(index,split='train',seed=1801,size=96):
    """Return four RGB uint8 HWC inputs plus explicit, non-model metadata.

    Model pairs are (support,positive), (support,near_negative), and
    (wrong_support,positive). Nuisance rejection never changes the target's
    structural program or silently skips an index. No masks/labels are inputs.
    """
    verify_frozen_source()
    if index<0 or size!=96 or split not in sc.SPLIT_NUISANCE_CODES:
        raise ValueError('Nonnegative index, canonical split, and size96 required')
    index=int(index);seed=int(seed);target=sc.sample_program(index,split,seed)
    first_negative,_=sc.hard_negative(target);replacement,edit=sc._second_negative(target,first_negative)
    target_id=sc.program_id(target);replacement_id=sc.program_id(replacement)
    if not any(sc.program_id(p)==replacement_id for p,_ in sc._single_edits(target)):
        raise RuntimeError('Near negative is not a one-attribute edit')
    rejected=[]
    for attempt in range(CONFIG['max_nuisance_attempts']):
        try:
            a=sc._episode_once(index,split,seed,size,192,1.,.65,'cluttered',nuisance_attempt=attempt)
            b=sc._episode_once(index,split,seed,size,192,0.,.65,'cluttered',nuisance_attempt=attempt)
        except sc.AmbiguousSceneError as exc:
            rejected.append(str(exc));continue
        if a['program_id']!=target_id or b['program_id']!=target_id or not np.array_equal(a['support'],b['support']):
            raise RuntimeError('Paired nuisance draws do not share the same reference program/image')
        pa=next(r for r in a['metadata']['instances']if r['role']=='target')
        pb=next(r for r in b['metadata']['instances']if r['role']=='matched_absence_replacement')
        if pa['placement_xywh']!=pb['placement_xywh']or pa['seed']!=pb['seed']or pb['program_id']!=replacement_id:
            raise RuntimeError('Counterfactual target slot or nuisance draw changed')
        x,y,side,_=pa['placement_xywh'];slot=[x,y,x+side,y+side]
        outside=np.ones((192,192),bool);outside[y:y+side,x:x+side]=False
        if not np.array_equal(a['query'][outside],b['query'][outside]):
            raise RuntimeError('Counterfactual changed pixels outside its target slot')
        target_instance=sc.render_instance(target,pa['seed'],side)
        replacement_instance=sc.render_instance(replacement,pa['seed'],side)
        union=(target_instance['mask']>0)|(replacement_instance['mask']>0)
        union_box=sc.bbox_from_mask(union).astype(int)+np.array([x,y,x,y])
        edited_box=sc.bbox_from_mask(np.any(a['query']!=b['query'],axis=2)).astype(int)
        guarded_union=np.array([min(union_box[0],edited_box[0])-1,min(union_box[1],edited_box[1])-1,
                                max(union_box[2],edited_box[2])+1,max(union_box[3],edited_box[3])+1]).clip(0,192)
        guarded_slot=[max(0,slot[0]-1),max(0,slot[1]-1),min(192,slot[2]+1),min(192,slot[3]+1)]
        crop_options,margin=_crop_options(guarded_union,guarded_slot,seed,index,split,attempt)
        chosen=None
        for crop_name,box in crop_options:
            images,frame=_fit_pair(a['query'],b['query'],box,size)
            changed=np.max(np.abs(images[0].astype(np.int16)-images[1].astype(np.int16)),axis=2)>=3
            if int(changed.sum())>=8:
                chosen=(crop_name,images,frame,int(changed.sum()));break
        if chosen is None:
            rejected.append('Counterfactual has fewer than8 changedRGBpixels after96px framing');continue
        crop_name,images,frame,changed_count=chosen
        scene_a=_scene_programs(a,target,first_negative,replacement,split,seed)
        scene_b=_scene_programs(b,target,first_negative,replacement,split,seed)
        if any(sc.contains_program(p,target)for p in scene_b):
            raise RuntimeError('Near-negative scene accidentally contains the target program')
        wrong,wrong_index=_wrong_program(index,split,seed,target,scene_a+scene_b)
        rebuilt,_=_render_reference(index,split,seed,attempt,target,size)
        if not np.array_equal(rebuilt,a['support']):raise RuntimeError('Reference nuisance reconstruction mismatch')
        wrong_rgb,wrong_frame=_render_reference(index,split,seed,attempt,wrong,size)
        scene_ids=set().union(*(_program_tree_ids(p)for p in scene_a+scene_b))
        output={'support':a['support'],'positive':images[0],'near_negative':images[1],
                'wrong_support':wrong_rgb,'program_id':target_id,'family':target['family']}
        output['metadata']={'format':FORMAT,'config_sha256':config_hash(),'index':index,'split':split,'seed':seed,
            'source_generator_sha256':V17_SOURCE_SHA256,'source_generator_format':sc.FORMAT,
            'program_id':target_id,'program':target,'family':target['family'],
            'near_negative_program_id':replacement_id,'near_negative_program':replacement,'near_negative_edit':edit,
            'wrong_program_id':sc.program_id(wrong),'wrong_program':wrong,'wrong_program_index':wrong_index,
            'wrong_family_rule':'different_constituent_family_pair'if split=='compositional'else'different_macro_family',
            'scene_whole_program_ids':sorted({sc.program_id(p)for p in scene_a+scene_b}),
            'scene_complete_and_contained_program_ids':sorted(scene_ids),
            'nuisance_attempt':attempt,'rejected_nuisance_reasons':rejected,
            'support_instance_seed':a['metadata']['support_instance_seed'],'query_instance_seed':pa['seed'],
            'source_slot_xyxy':slot,'source_amodal_union_xyxy':union_box.tolist(),
            'guarded_crop_union_xyxy':guarded_union.tolist(),
            'shared_crop':frame,'positive_crop_xyxy':frame['crop_xyxy'],'near_negative_crop_xyxy':frame['crop_xyxy'],
            'crop_choice':crop_name,'sampled_margin':margin,'wrong_support_frame':wrong_frame,
            'final_changed_rgb_pixels':changed_count,'positive_target_visibility':a['visibility'],
            'negative_slot_visibility':b['metadata']['slot_visibility'],
            'source_positive_scene_sha256':hashlib.sha256(a['query'].tobytes()).hexdigest(),
            'source_negative_scene_sha256':hashlib.sha256(b['query'].tobytes()).hexdigest(),
            'image_sha256':{key:hashlib.sha256(output[key].tobytes()).hexdigest()for key in IMAGE_KEYS},
            'same_program_positive':True,'same_raster_copy':False,'shared_counterfactual_nuisance':True,
            'synthetic_ground_truth_used_for_training_crop':True,
            'model_input_policy':'Four RGBimages only; metadata/programIDs/boxes/masks are not model inputs',
            'natural_images_used':0,'borrowed_pretrained_weights':False}
        return output
    raise RuntimeError(f'No valid96px pair after{CONFIG["max_nuisance_attempts"]} same-program nuisance draws: index={index},split={split},seed={seed}; {rejected[-1:]}')


def batch(indices,split='train',seed=1801,size=96,device=None):
    """Lazy torch adapter; each tensor is BCHW float32 in[0,1]."""
    import torch
    rows=[pair(int(i),split,seed,size)for i in indices]
    if not rows:raise ValueError('Empty pair batch')
    result={key:torch.from_numpy(np.stack([r[key]for r in rows])).permute(0,3,1,2).float().div_(255)for key in IMAGE_KEYS}
    if device is not None:result={key:value.to(device)for key,value in result.items()}
    result['metadata']=[r['metadata']for r in rows]
    return result
