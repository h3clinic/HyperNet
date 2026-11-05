"""CPU mathematical generator, namespace, coordinate and gradient checks."""
import train as tr
import numpy as np,torch,json,hashlib
from pathlib import Path
def main():
 assert not(tr.R/'protocol.json').exists();torch.set_num_threads(2);tr.dv.assert_protocol();geometry=[];negative_deltas=[];families={k:0 for k in tr.gen.FAMILIES};rgb_exact=True
 for seed,offset in zip(tr.SEEDS,tr.OFFSETS):
  order=tr.order_for(seed);original=tr.dv.tr.order_for(seed);assert np.array_equal(order-offset,original-tr.dv.tr.TRAIN_OFFSETS[tr.SEEDS.index(seed)])
  for identity in order[:20]:
   e=tr.gen.annotated_episode(int(identity),'train');families[e['metadata']['family']]+=1;q=e['local'];error=float(np.max(np.abs(q['reference']@q['affine'].T+q['offset']-q['positive'])));assert error<2e-5
   geometry.append(dict(identity=int(identity),error=error,available=q['available'],replacement=q['replacement']))
   for k in tr.gen.VIEWS:
    for suffix in ['', '_low']:
     a=e[k+suffix];assert a.shape==(128,128,3)and a.dtype==np.float32 and np.isfinite(a).all()and a.min()>=0 and a.max()<=1;assert np.array_equal(a[...,0],a[...,1])and np.array_equal(a[...,0],a[...,2])
    assert abs(e['metadata']['render_checks'][k]['foreground_rms']-1)<2e-6
   for k in tr.gen.NEGATIVE_TYPES:
    delta=float(np.max(np.abs(e[k]-e['positive'])));assert delta>1e-7,(identity,k,delta);negative_deltas.append(delta)
   # Replaying the episode must not change any image or annotation.
   if len(geometry)<=3:
    replay=tr.gen.annotated_episode(int(identity),'train');assert all(np.array_equal(e[k+s],replay[k+s])for k in tr.gen.VIEWS for s in ['', '_low']);assert np.array_equal(e['local']['reference'],replay['local']['reference'])
 # Noise fields and appearance are the exact same object for all query edits;
 # verify the reconstruction and that mutating a scene cannot mutate the source.
 identity=int(tr.order_for(tr.SEEDS[0])[0]);g=tr.gen.scene(identity);before=json.dumps(g,sort_keys=True);p=tr.gen.pose(tr.gen.rng(identity,'train','query_pose'));a=tr.gen.appearance(tr.gen.rng(identity,'train','query_appearance'));bg=a['background'].copy();e=tr.gen.episode(identity)
 for kind in tr.gen.NEGATIVE_TYPES:
  changed=tr.gen.mutate(g,kind,tr.gen.rng(identity,'train',kind));full,low,_=tr.gen.render(changed,p,a);assert np.array_equal(full,e[kind])and np.array_equal(low,e[kind+'_low']);assert np.array_equal(a['background'],bg)
 assert json.dumps(g,sort_keys=True)==before
 # Check all deterministic seed namespaces without rendering any heldout image.
 sets={};allseeds=set();ids=set()
 for seed,offset in zip(tr.SEEDS,tr.OFFSETS):sets[f'train_{seed}']=('v46','train',tr.order_for(seed))
 sets['spectrogram_test']=('v46','test',np.arange(tr.TEST_OFFSET,tr.TEST_OFFSET+tr.TEST_N));sets['retention_test']=('v44','test',np.arange(tr.RETENTION_OFFSET,tr.RETENTION_OFFSET+tr.TEST_N));namespace={}
 old_ids=set(range(4500000,4503200))|set(range(4510000,4513200))|set(range(4520000,4523200))|set(range(4590000,4590256))
 old_keys=['graph','reference_pose','query_pose','reference_appearance','query_appearance',*tr.gen.NEGATIVE_TYPES,'v45_local_foreground_points']
 previous_seeds={int.from_bytes(hashlib.sha256(f'v44-part-graphs|{"test"if identity>=4590000 else "train"}|{identity}|{key}'.encode()).digest()[:8],'little')for identity in old_ids for key in old_keys}
 for name,(grammar,split,order)in sets.items():
  current=set(map(int,order));assert not(current&ids)and not(current&old_ids);ids|=current
  component_seeds=[]
  for identity in order:
   for key in (tr.gen.COMPONENT_KEYS if grammar=='v46'else old_keys):
    if grammar=='v46':value=tr.gen.seed_for(int(identity),split,key)
    else:value=int.from_bytes(hashlib.sha256(f'v44-part-graphs|{split}|{int(identity)}|{key}'.encode()).digest()[:8],'little')
    component_seeds.append(value)
  assert len(set(component_seeds))==len(component_seeds)and not(set(component_seeds)&allseeds)and not(set(component_seeds)&previous_seeds);allseeds|=set(component_seeds);namespace[name]=dict(identities=len(order),minimum=int(min(order)),maximum=int(max(order)),component_keys=list(tr.gen.COMPONENT_KEYS if grammar=='v46'else old_keys),seed_sha256=hashlib.sha256(np.array(component_seeds,dtype='<u8').tobytes()).hexdigest())
 eps=[tr.gen.annotated_episode(int(i))for i in tr.order_for(tr.SEEDS[0])[:4]];full,low=tr.dv.tr.pack(eps,'cpu');net=tr.load_encoder(tr.M/f'seed_{tr.SEEDS[0]}/random.pt').train();loss,stats=tr.dv.objective(net,full,low,eps);loss.backward();assert torch.isfinite(loss)and all(p.grad is not None and torch.isfinite(p.grad).all()for p in net.parameters());assert sum(p.numel()for p in net.parameters())==42480
 result=dict(passed=True,cpu_only=True,heldout_rendered=0,real_images_read=0,files={str(p):tr.sha(p)for p in tr.sources()},original_objective_called_directly=True,unchanged_architecture_parameters=42480,paired_relative_order_exact=True,gradient_norm=float(torch.cat([p.grad.flatten()for p in net.parameters()]).norm()),objective_stats=stats,family_counts_checked=families,training_identities_rendered=len(geometry),geometry=dict(maximum_affine_error=max(x['error']for x in geometry),minimum_foreground_points=min(x['available']for x in geometry),replacement_identities=sum(x['replacement']for x in geometry)),minimum_negative_max_pixel_difference=min(negative_deltas),query_backgrounds_exact=True,reference_scene_not_mutated=True,namespace=namespace,unique_component_seeds=len(allseeds),limits='Geometry tracks analytic centers; different widths and raster interpolation make edge intensities approximate. Namespace disjointness does not imply new semantic families.')
 tr.save(tr.R/'checks.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
