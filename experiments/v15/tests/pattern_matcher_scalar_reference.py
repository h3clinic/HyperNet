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

    def enroll(self,rgb):
        array=np.asarray(rgb);encoded=self.features.encode(array,sampling_stride=4)
        return self.enroll_features(encoded,hashlib.sha256(array.tobytes()).hexdigest())

    def enroll_features(self,features,reference_rgb_sha256='test_features'):
        indices,cells=select_anchors(features)
        strength=features['strength'].reshape(-1)[indices].astype(np.float32)
        weights=np.sqrt(strength)
        if len(weights):weights=np.clip(weights/max(float(weights.mean()),1e-8),.25,4);weights/=weights.sum()
        return PatternBank(features['descriptors'].reshape(-1,features['descriptors'].shape[-1])[indices].copy(),
            features['detail'].reshape(-1,4)[indices].copy(),features['xy'].reshape(-1,2)[indices].copy(),
            weights,cells,reference_rgb_sha256,tuple(features['source_shape']))

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
        reference=np.concatenate([getattr(bank,'descriptors'if feature_kind=='learned'else'analytic')for bank in banks])
        query=np.asarray(query_features[key]).reshape(-1,dimension)
        a=torch.as_tensor(reference,device=self.device);b=torch.as_tensor(query,device=self.device)
        a=torch.nn.functional.normalize(a,dim=1,eps=1e-6);b=torch.nn.functional.normalize(b,dim=1,eps=1e-6)
        similarities=(a@b.T).clamp(-1,1)
        means=similarities.mean(1,keepdim=True)
        contrast=((similarities-means)/(1-means).clamp_min(.05)).clamp(0,1)
        xy=torch.as_tensor(query_features['xy'].reshape(-1,2),device=self.device)
        # Pixel coordinates, not an annotation mask, define nonmaximum suppression.
        working=similarities.clone();selected=[];values=[]
        for _ in range(min(8,len(query))):
            winner=working.argmax(1);selected.append(xy[winner]);values.append(contrast.gather(1,winner[:,None]).squeeze(1))
            center=xy[winner]
            excluded=(xy[None,:,0]-center[:,None,0]).abs()<=.035
            excluded&=(xy[None,:,1]-center[:,None,1]).abs()<=.035
            working.masked_fill_(excluded,-2.)
        positions=torch.stack(selected,1);qualities=torch.stack(values,1)
        results=[];start=0
        for bank,count in zip(banks,counts):
            if count<4:
                results.append({recipe:{'score':0.,'evidence':{'anchor_count':count,'abstain':'fewer than four distributed anchors'}}for recipe in recipes});start+=count;continue
            pos=positions[start:start+count];quality=qualities[start:start+count]
            refxy=torch.as_tensor(bank.xy,device=self.device)
            weights=torch.as_tensor(bank.weights,device=self.device)
            offsets=pos-refxy[:,None]
            candidates=offsets.reshape(-1,2)
            candidates=candidates[(candidates.abs()<=.25).all(1)]
            candidates=torch.cat((torch.zeros(1,2,device=self.device),candidates),0)
            recipe_results={}
            for recipe in recipes:
                tolerance=RECIPES[recipe]
                if tolerance is None:
                    per_anchor=quality[:,0];winner_index=torch.zeros(count,dtype=torch.long,device=self.device);translation=None
                else:
                    consistent=(offsets[None]-candidates[:,None,None]).abs().amax(-1)<=tolerance
                    hypothesis_values=quality[None].expand(len(candidates),-1,-1).masked_fill(~consistent,0)
                    anchor_values,anchor_winners=hypothesis_values.max(-1)
                    hypothesis_scores=(anchor_values*weights).sum(1)
                    best=hypothesis_scores.argmax();per_anchor=anchor_values[best];winner_index=anchor_winners[best]
                    translation=candidates[best].cpu().tolist()
                score=float((per_anchor*weights).sum())
                contributing=(per_anchor>.1).cpu().numpy()
                distinct_total=len(np.unique(bank.cells));covered=len(np.unique(bank.cells[contributing]))
                recipe_results[recipe]={'score':score,'evidence':{'anchor_count':count,
                    'contributing_anchors':int(contributing.sum()),'reference_cells':distinct_total,
                    'covered_reference_cells':covered,'reference_cell_coverage':covered/max(1,distinct_total),
                    'translation_xy':translation,'position_tolerance':tolerance,
                    'aggregation':'weighted mean across every selected anchor; unmatched anchors contribute zero',
                    'feature_kind':feature_kind,'query_pixels_searched':len(query)}}
            results.append(recipe_results);start+=count
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
