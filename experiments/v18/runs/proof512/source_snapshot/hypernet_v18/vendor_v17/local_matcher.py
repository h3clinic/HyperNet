"""Frozen one-reference detection. No text, real-image fitting, or GT masks."""
from pathlib import Path
import hashlib
import math
import cv2
import numpy as np
import torch
from .local_relation_model import LocalRelationNet, FORMAT, SCALES, ASPECTS, decoded_boxes, state_hash

MATCHER_FORMAT = 'hypernet.local_matcher.v17.1'


def checked_rgb(value):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError('Expected a nonempty RGB uint8 HWC image')
    return np.ascontiguousarray(value)


def prepare_image(rgb, *, support=False, max_side=384):
    rgb = checked_rgb(rgb); h, w = rgb.shape[:2]
    if int(max_side) < 32: raise ValueError('max_side must be >=32')
    scale = 96/max(h,w) if support else min(1., max_side/max(h,w))
    rh, rw = max(1, round(h*scale)), max(1, round(w*scale))
    resized = cv2.resize(rgb, (rw,rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if support:
        oh = ow = 96; top = (oh-rh)//2; left = (ow-rw)//2
    else:
        oh = max(32, math.ceil(rh/8)*8); ow = max(32, math.ceil(rw/8)*8); top = left = 0
    border = np.concatenate([rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]])
    fill = np.median(border,axis=0).round().astype(np.uint8)
    canvas = np.broadcast_to(fill,(oh,ow,3)).copy(); canvas[top:top+rh,left:left+rw] = resized
    return canvas, {'original_height':h,'original_width':w,'resized_height':rh,'resized_width':rw,
                    'scale_x':rw/w,'scale_y':rh/h,'top':top,'left':left,'canvas_height':oh,'canvas_width':ow}


def box_iou_one(box, others):
    lo=np.maximum(box[:2],others[:,:2]); hi=np.minimum(box[2:],others[:,2:])
    intersection=np.maximum(0,hi-lo).prod(1)
    union=np.maximum(0,box[2:]-box[:2]).prod()+np.maximum(0,others[:,2:]-others[:,:2]).prod(1)-intersection
    return intersection/np.maximum(union,1e-12)


def nms(boxes,scores,threshold=.4,limit=100):
    """Stable greedy NMS; equal scores retain original candidate order."""
    boxes=np.asarray(boxes,np.float64);scores=np.asarray(scores,np.float64)
    if boxes.shape!=(len(scores),4) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError('Finite aligned boxes/scores required')
    order=np.argsort(-scores,kind='stable');chosen=[]
    while len(order) and len(chosen)<limit:
        index=int(order[0]);chosen.append(index)
        order=order[1:][box_iou_one(boxes[index],boxes[order[1:]])<=threshold]
    return chosen


class FrozenLocalMatcher:
    def __init__(self,checkpoint,device='cpu',max_side=384,proposal_limit=100,nms_iou=.4,pre_nms=1000):
        self.path=Path(checkpoint).resolve();self.device=str(device)
        self.checkpoint_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        saved=torch.load(self.path,map_location='cpu',weights_only=True)
        if saved.get('format')!=FORMAT or saved.get('natural_training_images')!=0 or saved.get('borrowed_pretrained_weights')is not False:
            raise ValueError('Expected an own synthetic-only v17 checkpoint')
        self.saved=saved;self.model=LocalRelationNet(geometry=bool(saved['config']['geometry'])).to(device)
        self.model.load_state_dict(saved['state_dict'],strict=True);self.model.eval();self.model.requires_grad_(False)
        self.initial_state=state_hash(self.model)
        if self.initial_state!=saved['state_sha256']:raise ValueError('Checkpoint state hash mismatch')
        self.max_side=int(max_side);self.proposal_limit=int(proposal_limit);self.nms_iou=float(nms_iou);self.pre_nms=int(pre_nms)
        if self.max_side<32 or self.proposal_limit<1 or self.pre_nms<self.proposal_limit or not 0<=self.nms_iou<=1:
            raise ValueError('Invalid frozen inference settings')

    def _tensor(self,image):
        return torch.from_numpy(image.copy()).permute(2,0,1).unsqueeze(0).to(self.device,dtype=torch.float32)/255

    @torch.inference_mode()
    def enroll(self,reference):
        rgb=checked_rgb(reference);prepared,meta=prepare_image(rgb,support=True,max_side=self.max_side)
        encoded=self.model.encode(self._tensor(prepared));bank=self.model.support_bank(encoded)
        return {'format':MATCHER_FORMAT,'state_sha256':self.initial_state,'bank':bank,'preprocessing':meta,
                'reference_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'reference_images':1,
                'learned_mask_only':True,'real_weight_updates':0}

    @torch.inference_mode()
    def encode_query(self,query):
        rgb=checked_rgb(query);prepared,meta=prepare_image(rgb,support=False,max_side=self.max_side)
        encoded=self.model.encode(self._tensor(prepared))
        return {'format':MATCHER_FORMAT,'state_sha256':self.initial_state,'encoded':encoded,'preprocessing':meta,
                'query_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}

    @torch.inference_mode()
    def predict(self,bank,query):
        encoded=query if isinstance(query,dict) else self.encode_query(query)
        for value in (bank,encoded):
            if value.get('format')!=MATCHER_FORMAT or value.get('state_sha256')!=self.initial_state:
                raise ValueError('Bank/query belongs to a different model')
        out=self.model.match_encoded(bank['bank'],encoded['encoded']);meta=encoded['preprocessing']
        scores=out['center_logits'][0].flatten().float().cpu().numpy()
        boxes=decoded_boxes(out)[0].float().cpu().numpy()
        boxes[:,[0,2]]=(boxes[:,[0,2]]-meta['left'])/meta['scale_x']
        boxes[:,[1,3]]=(boxes[:,[1,3]]-meta['top'])/meta['scale_y']
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,meta['original_width'])
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,meta['original_height'])
        valid=np.isfinite(scores)&np.isfinite(boxes).all(1)&((boxes[:,2:]-boxes[:,:2])>0).all(1)
        indices=np.flatnonzero(valid);order=np.argsort(-scores[indices],kind='stable')[:self.pre_nms];indices=indices[order]
        if not len(indices):raise RuntimeError('No finite nonempty detection proposal')
        kept=nms(boxes[indices],scores[indices],self.nms_iou,self.proposal_limit)
        return [{'box':[float(x)for x in boxes[indices[k]]],'score':float(scores[indices[k]])}for k in kept]

    def provenance(self):
        return {'format':MATCHER_FORMAT,'model_format':FORMAT,'checkpoint_sha256':self.checkpoint_sha256,
                'state_sha256':self.initial_state,'optimizer_steps':int(self.saved['optimizer_steps']),
                'training_episodes':int(self.saved['training_episodes']),'training_views':int(self.saved['training_views']),
                'parameters':sum(p.numel()for p in self.model.parameters()),'geometry':self.model.geometry,
                'training_source_sha256':self.saved['config'].get('source_sha256',{}),
                'training_stream_seed':self.saved['config'].get('stream_seed'),
                'training_initialization_seed':self.saved['config'].get('seed'),
                'device':self.device,'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),
                'opencv_version':str(cv2.__version__),'support_size':96,'max_query_side':self.max_side,
                'proposal_limit':self.proposal_limit,'pre_nms_candidates':self.pre_nms,'nms_iou':self.nms_iou,
                'score':'Uncalibrated learned center logit; no confidence prefilter',
                'boxes':'Original-image absolute xyxy pixels, half-open; clipped to image bounds',
                'preprocessing':'Aspect-preserving resize, median-border RGB padding; query never upscaled',
                'anchor_scales':list(SCALES),'anchor_aspects':list(ASPECTS),'descriptor_stride':2,
                'support_coordinates':'Predicted foreground logit>0 extent; detached integer crop; minimum4 descriptor cells per axis; empty mask falls back to full reference',
                'natural_training_images':0,'borrowed_pretrained_weights':False,'real_weight_updates':0,
                'input_supervision':'One supplied reference RGB and query RGB only; no masks, boxes, text or labels'}

    def assert_unchanged(self):
        if state_hash(self.model)!=self.initial_state or hashlib.sha256(self.path.read_bytes()).hexdigest()!=self.checkpoint_sha256:
            raise RuntimeError('Frozen model/checkpoint changed')
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Inference model became trainable')
