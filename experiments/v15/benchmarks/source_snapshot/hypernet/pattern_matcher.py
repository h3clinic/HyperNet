"""Fixed-axis, many-point pattern matching with a frozen mathematical CNN.

Labels, query masks and learned cross-event adaptation are absent. Native
gradient strength selects distributed reference anchors. Translation hypotheses
must explain many anchors; one matching patch cannot contribute a full score.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch

RECIPES={'bag':None,'translation_006':.06,'translation_012':.12}
CONFIG={'spatial_cells_per_axis':4,'anchors_per_cell':2,'global_strength_quantile':.75,
        'anchor_separation':.025,'top_candidates_per_anchor':8,'query_nms_radius':.035,
        'maximum_translation_per_axis':.25,'similarity_contrast_denominator_floor':.05}


def raw_pixel_vector(rgb):
    """Declared cheap global control; deliberately compresses spatial detail."""
    import cv2
    values=np.asarray(rgb)
    if values.dtype!=np.uint8 or values.ndim!=3 or values.shape[2]!=3:raise ValueError('Expected uint8 RGB')
    gray=values.astype(np.float32).mean(-1)
    vector=cv2.resize(gray,(48,48),interpolation=cv2.INTER_AREA).reshape(-1)
    vector-=vector.mean()
    return vector/max(float(np.linalg.norm(vector)),1e-8)


def _check(features):
    shape=features['strength'].shape
    if len(shape)!=2 or features['xy'].shape!=(*shape,2):raise ValueError('Aligned spatial features required')
    for key in('descriptors','detail','strength','xy'):
        if not np.isfinite(features[key]).all():raise ValueError('Nonfinite feature input')
    return shape


def select_anchors(features):
    _check(features);strength=np.asarray(features['strength']).reshape(-1)
    xy=np.asarray(features['xy']).reshape(-1,2)
    if not len(strength)or strength.max()<=1e-8:return np.array([],dtype=np.int64),np.array([],dtype=np.int64)
    cutoff=max(float(np.quantile(strength,.75)),1e-8);chosen=[];cells=[]
    cellxy=np.minimum((xy*4).astype(int),3)
    for y in range(4):
        for x in range(4):
            candidates=np.flatnonzero((cellxy[:,0]==x)&(cellxy[:,1]==y)&(strength>=cutoff))
            # Stable flat index breaks strength ties reproducibly.
            candidates=candidates[np.argsort(-strength[candidates],kind='stable')]
            accepted=[]
            for index in candidates:
                if all(np.max(np.abs(xy[index]-xy[prior]))>=.025 for prior in accepted):
                    accepted.append(int(index))
                if len(accepted)==2:break
            chosen.extend(accepted);cells.extend([y*4+x]*len(accepted))
    return np.asarray(chosen,np.int64),np.asarray(cells,np.int64)


@dataclass(frozen=True)
class PatternBank:
    descriptors:np.ndarray
    analytic:np.ndarray
    xy:np.ndarray
    weights:np.ndarray
    cells:np.ndarray
    reference_rgb_sha256:str
    source_shape:tuple


class PatternMatcher:
    def __init__(self,features,recipe='translation_006',device=None):
        if recipe not in RECIPES:raise ValueError('Unknown preregistered recipe')
        self.features=features;self.recipe=recipe;self.device=device or getattr(features,'device','cpu')
        self._bank_cache={};self._xy_cache={}

    def enroll(self,rgb):
        array=np.asarray(rgb);encoded=self.features.encode(array,sampling_stride=4)
        return self.enroll_features(encoded,hashlib.sha256(array.tobytes()).hexdigest())

    def enroll_features(self,features,reference_rgb_sha256='test_features'):
        indices,cells=select_anchors(features)
        strength=features['strength'].reshape(-1)[indices].astype(np.float32)
        weights=np.sqrt(strength)
        if len(weights):weights=np.clip(weights/max(float(weights.mean()),1e-8),.25,4);weights/=weights.sum()
        bank=PatternBank(features['descriptors'].reshape(-1,features['descriptors'].shape[-1])[indices].copy(),
            features['detail'].reshape(-1,4)[indices].copy(),features['xy'].reshape(-1,2)[indices].copy(),
            weights,cells,reference_rgb_sha256,tuple(features['source_shape']))
        for value in(bank.descriptors,bank.analytic,bank.xy,bank.weights,bank.cells):value.flags.writeable=False
        return bank

    @torch.inference_mode()
    def score_many(self,banks,query_features,recipes=None,feature_kind='learned'):
        _check(query_features)
        recipes=tuple(recipes or(self.recipe,))
        if not set(recipes).issubset(RECIPES)or feature_kind not in('learned','analytic'):raise ValueError('Unknown recipe/features')
        banks=list(banks)
        if not banks:return []
        dimension=24 if feature_kind=='learned'else 4
        key='descriptors'if feature_kind=='learned'else'detail'
        counts=[len(bank.xy)for bank in banks]
        if not sum(counts)or query_features['strength'].max()<=1e-8:
            return [{recipe:{'score':0.,'evidence':{'anchor_count':count,'abstain':'no informative reference or query detail'}}for recipe in recipes}for count in counts]
        cache_key=(tuple(id(bank)for bank in banks),feature_kind)
        if cache_key not in self._bank_cache:
            length=max(counts);number=len(banks)
            reference=np.zeros((number,length,dimension),np.float32);coords=np.zeros((number,length,2),np.float32)
            weight=np.zeros((number,length),np.float32);cell_onehot=np.zeros((number,length,16),np.float32)
            valid=np.zeros((number,length),bool)
            for index,(bank,count)in enumerate(zip(banks,counts)):
                reference[index,:count]=getattr(bank,'descriptors'if feature_kind=='learned'else'analytic')
                coords[index,:count]=bank.xy;weight[index,:count]=bank.weights;valid[index,:count]=True
                cell_onehot[index,np.arange(count),bank.cells]=1
            self._bank_cache[cache_key]={
                'reference':torch.nn.functional.normalize(torch.as_tensor(reference,device=self.device),dim=-1,eps=1e-6),
                'coords':torch.as_tensor(coords,device=self.device),'weights':torch.as_tensor(weight,device=self.device),
                'valid':torch.as_tensor(valid,device=self.device),'cells':torch.as_tensor(cell_onehot,device=self.device),
                'length':length,'number':number}
        cached=self._bank_cache[cache_key];length=cached['length'];number=cached['number']
        query=np.asarray(query_features[key]).reshape(-1,dimension)
        a=cached['reference'].reshape(-1,dimension);b=torch.as_tensor(query,device=self.device)
        b=torch.nn.functional.normalize(b,dim=1,eps=1e-6)
        similarities=(a@b.T).clamp(-1,1)
        means=similarities.mean(1,keepdim=True)
        contrast=((similarities-means)/(1-means).clamp_min(.05)).clamp(0,1)
        xy_values=query_features['xy'].reshape(-1,2)
        xy_key=hashlib.sha256(xy_values.tobytes()).hexdigest()
        if xy_key not in self._xy_cache:
            self._xy_cache={xy_key:torch.as_tensor(xy_values,device=self.device)}
        xy=self._xy_cache[xy_key]
        # Pixel coordinates, not an annotation mask, define nonmaximum suppression.
        working=similarities.clone();selected=[];values=[]
        for _ in range(min(8,len(query))):
            winner=working.argmax(1);selected.append(xy[winner]);values.append(contrast.gather(1,winner[:,None]).squeeze(1))
            center=xy[winner]
            excluded=(xy[None,:,0]-center[:,None,0]).abs()<=.035
            excluded&=(xy[None,:,1]-center[:,None,1]).abs()<=.035
            working.masked_fill_(excluded,-2.)
        positions=torch.stack(selected,1).reshape(number,length,-1,2)
        qualities=torch.stack(values,1).reshape(number,length,-1)
        offsets=positions-cached['coords'][:,:,None]
        candidates=offsets.reshape(number,-1,2)
        candidate_valid=(candidates.abs()<=.25).all(-1)&cached['valid'][:,:,None].expand(-1,-1,qualities.shape[-1]).reshape(number,-1)
        candidates=torch.cat((torch.zeros(number,1,2,device=self.device),candidates),1)
        candidate_valid=torch.cat((torch.ones(number,1,dtype=torch.bool,device=self.device),candidate_valid),1)
        distances=(offsets[:,None]-candidates[:,:,None,None]).abs().amax(-1)
        results=[{}for _ in banks];batch_index=torch.arange(number,device=self.device)
        for recipe in recipes:
            tolerance=RECIPES[recipe]
            if tolerance is None:
                per_anchor=qualities[:,:,0];translation=torch.zeros(number,2,device=self.device)
            else:
                consistent=distances<=tolerance
                hypothesis_values=qualities[:,None].expand(-1,candidates.shape[1],-1,-1).masked_fill(~consistent,0)
                anchor_values=hypothesis_values.amax(-1)
                hypothesis_scores=(anchor_values*cached['weights'][:,None]).sum(-1).masked_fill(~candidate_valid,-1)
                best=hypothesis_scores.argmax(-1);per_anchor=anchor_values[batch_index,best]
                translation=candidates[batch_index,best]
            score=(per_anchor*cached['weights']).sum(-1)
            contributing=(per_anchor>.1)&cached['valid']
            covered=(contributing[:,:,None]*cached['cells']).amax(1).sum(-1)
            # One device-to-host synchronization per recipe, rather than per bank/field.
            output=torch.stack((score,contributing.sum(-1),covered,translation[:,0],translation[:,1]),-1).cpu().numpy()
            for index,(bank,count)in enumerate(zip(banks,counts)):
                if count<4:
                    results[index][recipe]={'score':0.,'evidence':{'anchor_count':count,'abstain':'fewer than four distributed anchors'}};continue
                value=output[index];distinct_total=len(np.unique(bank.cells))
                results[index][recipe]={'score':float(value[0]),'evidence':{'anchor_count':count,
                    'contributing_anchors':int(value[1]),'reference_cells':distinct_total,
                    'covered_reference_cells':int(value[2]),'reference_cell_coverage':int(value[2])/max(1,distinct_total),
                    'translation_xy':None if tolerance is None else value[3:5].tolist(),'position_tolerance':tolerance,
                    'aggregation':'weighted mean across every selected anchor; unmatched anchors contribute zero',
                    'feature_kind':feature_kind,'query_pixels_searched':len(query)}}
        return results

    def score(self,bank,query_features,feature_kind='learned'):
        return self.score_many([bank],query_features,feature_kind=feature_kind)[0][self.recipe]

    def predict(self,bank,query_rgb,feature_kind='learned'):
        return self.score(bank,self.features.encode(query_rgb,sampling_stride=4),feature_kind=feature_kind)

    def provenance(self):
        return {'feature_extractor':self.features.provenance(),'matcher':'fixed-axis many-point descriptor matching',
            'config':CONFIG,'recipes':RECIPES,'labels_used_by_scorer':False,'query_annotations_used':False,
            'rotation_or_flip_search':False,'enrollment_weight_updates':0,
            'score_scope':'Query-relative cosine excess averaged over spatially distributed reference anchors; not a probability.'}
