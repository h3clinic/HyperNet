"""Frozen native-resolution features, preserving image pixels through tiling."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from .detail_encoder import DetailEncoder,FORMAT,state_hash


def tile_layout(height,width,core=48,margin=8):
    if min(height,width)<8 or core!=48 or margin!=8:raise ValueError('Expected native64px tiles with48px cores')
    rows=(height+core-1)//core;columns=(width+core-1)//core
    return {'rows':rows,'columns':columns,'pad':((margin,rows*core+margin-height),(margin,columns*core+margin-width),(0,0))}


class FrozenDetails:
    def __init__(self,checkpoint,device='cuda',batch_size=64):
        self.checkpoint=Path(checkpoint).resolve();self.device=device;self.batch_size=batch_size
        saved=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        if saved.get('format')!=FORMAT or saved.get('natural_training_images')!=0 or saved.get('borrowed_pretrained_weights')is not False:
            raise ValueError('Expected this project synthetic-only detail checkpoint')
        self.model=DetailEncoder();self.model.load_state_dict(saved['state_dict'],strict=True)
        if state_hash(self.model)!=saved['state_sha256']:raise ValueError('Checkpoint state mismatch')
        self.model.to(device).eval().requires_grad_(False)
        self.initial_state=saved['state_sha256'];self.steps=saved['optimizer_steps']
        self.families=saved.get('consumed_families',0)
        self.initial_checkpoint_sha=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    def provenance(self):
        return {'format':FORMAT,'state_sha256':state_hash(self.model),'checkpoint_sha256':hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            'optimizer_steps':self.steps,'consumed_synthetic_families':self.families,'borrowed_pretrained_weights':False,
            'natural_weight_training_images':0,'reference_images_per_target':1,'enrollment_weight_updates':0,
            'tile_side':64,'tile_core':48,'overlap_margin':8,'preprocessing':'Native pixels, reflect-padding, no image resize. Same tiling for reference and query.'}

    @torch.inference_mode()
    def encode(self,rgb,sampling_stride=4):
        rgb=np.asarray(rgb)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<8 or rgb.shape[0]*rgb.shape[1]>16_000_000:
            raise ValueError('Expected8px minimum uint8RGB image, maximum16MP')
        if sampling_stride not in(1,2,4,8,12):raise ValueError('Sampling stride must evenly divide48')
        h,w=rgb.shape[:2];layout=tile_layout(h,w);padded=np.pad(rgb,layout['pad'],mode='reflect')
        n=48//sampling_stride;gh=(h+sampling_stride-1)//sampling_stride;gw=(w+sampling_stride-1)//sampling_stride
        descriptors=np.empty((layout['rows']*n,layout['columns']*n,24),np.float32)
        detail=np.empty((layout['rows']*n,layout['columns']*n,4),np.float32)
        positions=[(r,c)for r in range(layout['rows'])for c in range(layout['columns'])]
        for start in range(0,len(positions),self.batch_size):
            batch_positions=positions[start:start+self.batch_size]
            tiles=np.stack([padded[r*48:r*48+64,c*48:c*48+64]for r,c in batch_positions])
            values=torch.from_numpy(tiles).to(self.device).permute(0,3,1,2).contiguous().float()/255
            encoded=self.model(values)
            a=encoded['dense'][:,:,8:56:sampling_stride,8:56:sampling_stride].permute(0,2,3,1).cpu().numpy()
            b=encoded['detail_target'][:,:,8:56:sampling_stride,8:56:sampling_stride].permute(0,2,3,1).cpu().numpy()
            for index,(r,c)in enumerate(batch_positions):
                descriptors[r*n:(r+1)*n,c*n:(c+1)*n]=a[index]
                detail[r*n:(r+1)*n,c*n:(c+1)*n]=b[index]
        descriptors=descriptors[:gh,:gw].copy();detail=detail[:gh,:gw].copy()
        yy,xx=np.meshgrid(np.arange(gh)*sampling_stride,np.arange(gw)*sampling_stride,indexing='ij')
        xy=np.stack([(xx+.5)/w,(yy+.5)/h],axis=-1).astype(np.float32)
        strength=(np.hypot(detail[...,1],detail[...,2])+.5*np.abs(detail[...,3])+.25*np.abs(detail[...,0])).astype(np.float32)
        return {'descriptors':descriptors,'detail':detail,'strength':strength,'xy':xy,
                'source_shape':[h,w],'sampling_stride':sampling_stride,'source_pixels_resized':False,
                'tile_count':len(positions),'detail_scope':'Four fixed analytic channels from the encoder preprocessing, not learned mask outputs.'}
