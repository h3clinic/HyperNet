"""Pinned official OWL-ViT, image-guided, independent-feature caching only."""
from pathlib import Path
import hashlib, json
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTImageProcessor

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

class OwlBaseline:
    def __init__(self, checkpoint, device='cuda'):
        self.checkpoint=Path(checkpoint).resolve(); self.device=torch.device(device)
        self.acquisition=json.loads((self.checkpoint/'acquisition.json').read_text())
        self.file_hashes={str(self.checkpoint/n):r['sha256'] for n,r in self.acquisition['files'].items()}
        self.assert_unchanged()
        self.processor=OwlViTImageProcessor.from_pretrained(self.checkpoint,local_files_only=True)
        self.model,loading=OwlViTForObjectDetection.from_pretrained(self.checkpoint,local_files_only=True,
            use_safetensors=True,attn_implementation='eager',output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Incomplete official checkpoint: '+str(loading))
        self.model.to(self.device).eval().requires_grad_(False)

    def pixels(self,rgb):
        return self.processor(images=Image.fromarray(rgb),return_tensors='pt')['pixel_values'].to(self.device)

    @torch.inference_mode()
    def enroll(self,rgb):
        fmap=self.model.image_embedder(self.pixels(rgb))[0]
        feats=fmap.reshape(fmap.shape[0],-1,fmap.shape[-1])
        embeds,indices,_=self.model.embed_image_query(feats,fmap)
        if embeds is None: raise ValueError('Official image-query embedding returned no exemplar')
        return {'embeds':embeds,'indices':indices}

    @torch.inference_mode()
    def encode_query(self,rgb):
        fmap=self.model.image_embedder(self.pixels(rgb))[0]
        feats=fmap.reshape(fmap.shape[0],-1,fmap.shape[-1])
        boxes=self.model.box_predictor(feats,fmap)
        return {'feats':feats,'boxes':boxes,'height':rgb.shape[0],'width':rgb.shape[1]}

    @torch.inference_mode()
    def outputs(self,bank,query):
        logits,_=self.model.class_predictor(image_feats=query['feats'],query_embeds=bank['embeds'])
        return logits,query['boxes']

    @torch.inference_mode()
    def predict(self,bank,query):
        if isinstance(query,np.ndarray): query=self.encode_query(query)
        logits,boxes=self.outputs(bank,query)
        if logits.shape[-1]!=1: raise ValueError('Exactly one image exemplar required')
        flat=logits.reshape(-1); top=int(torch.argmax(flat).item())
        score=float(torch.sigmoid(flat[top]).item()); cx,cy,w,h=boxes[0,top].tolist()
        width,height=query['width'],query['height']
        box=[(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height]
        if not np.isfinite([score,*box]).all(): raise ValueError('Nonfinite OWL prediction')
        return [{'score':score,'box':box,'raw_logit':float(flat[top].item()),'query_token':top}]

    @torch.inference_mode()
    def equivalence(self,reference,query):
        bank=self.enroll(reference); enc=self.encode_query(query); logits,boxes=self.outputs(bank,enc)
        direct=self.model.image_guided_detection(pixel_values=self.pixels(query),query_pixel_values=self.pixels(reference))
        repeat=self.model.image_guided_detection(pixel_values=self.pixels(query),query_pixel_values=self.pixels(reference))
        errors={'cached_direct_logits':float((logits-direct.logits).abs().max().item()),
                'cached_direct_boxes':float((boxes-direct.target_pred_boxes).abs().max().item()),
                'direct_repeat_logits':float((direct.logits-repeat.logits).abs().max().item()),
                'direct_repeat_boxes':float((direct.target_pred_boxes-repeat.target_pred_boxes).abs().max().item())}
        checks=[torch.allclose(logits,direct.logits,rtol=1e-5,atol=1e-5),
                torch.allclose(boxes,direct.target_pred_boxes,rtol=1e-5,atol=1e-5),
                torch.equal(direct.logits,repeat.logits),torch.equal(direct.target_pred_boxes,repeat.target_pred_boxes)]
        if not all(checks): raise ValueError('Official/cache or determinism check failed: '+str(errors))
        return {'passed':True,'atol':1e-5,'rtol':1e-5,'max_absolute_errors':errors,'repeat_exact':True}

    def assert_unchanged(self):
        for p,h in self.file_hashes.items():
            if digest(p)!=h: raise ValueError('Pinned OWL checkpoint changed: '+p)

    def provenance(self):
        return {'name':'owlvit_base_patch32','checkpoint_id':self.acquisition['repo_id'],
                'revision':self.acquisition['revision'],'checkpoint_files':self.file_hashes,
                'borrowed_pretrained_weights':True,'image_guided':True,'support_images_per_target':1,
                'text_prompts':False,'weight_updates':0,'score':'sigmoid(max raw image-guided logit), no display normalization',
                'input_size':self.processor.size,'device':str(self.device),'dtype':'float32',
                'torch':torch.__version__,'transformers':transformers.__version__,
                'cache':'Independent image feature extraction; no fitting or test adaptation',
                'pretraining_overlap':'COCO and OpenImages are named in the official model card; fresh to HyperNet does not mean unseen by OWL.'}
