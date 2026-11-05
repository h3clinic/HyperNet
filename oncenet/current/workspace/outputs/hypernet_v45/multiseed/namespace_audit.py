"""Metadata-only train/test namespace separation; never render a test image."""
from itertools import combinations
import hashlib,json
import numpy as np
import train as tr
import generator as gen
from model import R,sha,save
def main():
 protocol,ph=tr.assert_protocol();components=['graph','reference_pose','query_pose','reference_appearance','query_appearance',*gen.NEGATIVE_TYPES,'v45_local_foreground_points'];groups={}
 for seed,offset in zip(tr.SEEDS,tr.TRAIN_OFFSETS):
  ids=tr.order_for(seed);assert len(set(ids.tolist()))==3200;assert np.array_equal(np.sort(ids),np.arange(offset,offset+3200));groups[f'train_{seed}']=('train',ids)
 groups['synthetic_test']=('test',np.arange(tr.TEST_OFFSET,tr.TEST_OFFSET+tr.TEST_N))
 allseeds=set();records={};cross=[];seen_ids=set()
 for name,(split,ids) in groups.items():
  idset=set(int(i) for i in ids);assert not (idset&seen_ids);seen_ids|=idset
  seed_values=[gen._c._g.seed_for(int(identity),split,component) for identity in ids for component in components];ss=set(seed_values);assert len(ss)==len(seed_values);assert not(ss&allseeds);allseeds|=ss
  records[name]=dict(split=split,identities=len(ids),minimum_identity=int(min(ids)),maximum_identity=int(max(ids)),order_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),component_seeds=len(seed_values),component_seeds_sha256=hashlib.sha256(np.array(seed_values,dtype='<u8').tobytes()).hexdigest())
  if split=='train':assert records[name]['order_sha256']==protocol['plan']['order_sha256'][name.split('_')[1]]
 for a,b in combinations(groups,2):cross.append(dict(left=a,right=b,identity_overlap=0,component_seed_overlap=0))
 result=dict(passed=True,protocol_sha256=ph,audit_code_sha256=sha(__file__),components=components,groups=records,pairwise=cross,global_unique_component_seeds=len(allseeds),test_images_rendered=0,test_models_executed=0,natural_images_read=0,limitation='Distinct deterministic latent seeds prevent exact seed reuse. Training and test share the graph grammar, renderer and edit types; this does not establish semantic family separation.')
 save(R/'namespace_audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
