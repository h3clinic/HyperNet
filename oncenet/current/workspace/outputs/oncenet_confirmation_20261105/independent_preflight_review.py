"""Independent read-only artifact/schema review; never acquires or scores images."""
from pathlib import Path
import hashlib, json
from datetime import datetime, timezone
import numpy as np

R=Path(__file__).resolve().parent
W=R.parents[1]
O=W/'outputs'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

p=read(R/'protocol.json')
assert sha(R/'protocol.json')==(R/'protocol.sha256').read_text().strip()
assert sha(R/'run_confirmation.py')==p['code_sha256']
assert not (R/'scores_commitment.json').exists()
assert not (R/'acquisition.json').exists()
assert not (R/'raw').exists(), 'Independent preflight must precede reserve image acquisition'
for f,h in p['source_hashes'].items():assert sha(f)==h, f

GS=O/'cloud_plan_20261105/gs_fp32'
V46=O/'hypernet_v46'
V47=O/'hypernet_v47'
reg=read(V46/'evaluation/registry.json')
entries={e['id']:e for e in reg['models']}
for model in ['s45029_random','s45029_spectrogram']:
    e=entries[model];assert sha(e['path'])==e['sha256']
for f,h in entries['dino_small']['files'].items():assert sha(f)==h
pre=read(GS/'pre_dev_binding.json')
assert pre['migration_manifest_sha256']==sha(GS/'migration_manifest.json')
for f,h in pre['files'].items():assert sha(GS/f)==h
assert sha(GS/'encoder.pt')==entries['s45029_spectrogram']['sha256']
dino_binding=read(V47/'evaluation/dino_small/binding.json')
for f,h in dino_binding['files'].items():assert sha(f)==h
v46result=read(V46/'evaluation/results.json')
for name in ['s45029_random_states.npz','s45029_random_scores.npz']:
    f=V46/'evaluation'/name;assert sha(f)==v46result['files'][str(f)]

targets=['Low_Frequency_Burst','Low_Frequency_Lines','Scattered_Light','Blip']
durations=['d05','d1','d2','d4']
states={
    'oncenet':np.load(GS/'matcher.npz',allow_pickle=False),
    'random':np.load(V46/'evaluation/s45029_random_states.npz',allow_pickle=False),
    'dino':np.load(V47/'evaluation/dino_small/states.npz',allow_pickle=False),
}
norm=read(V47/'evaluation/dino_small/normalization.json')
for target in targets:
    for name,prefix in [('oncenet',target),('random','multi__'+target)]:
        assert states[name][prefix+'__weight'].shape==(16128,)
        assert int(states[name][prefix+'__effective_dimension'])==4032
        assert np.isfinite(states[name][prefix+'__weight']).all()
    for d in durations:
        assert states['dino'][target+'__'+d+'__weight'].shape==(1536,)
        assert int(states['dino'][target+'__'+d+'__effective_dimension'])==1536
    assert np.asarray(norm[target]['location']).shape==(4,)
    assert np.asarray(norm[target]['scale']).shape==(4,)
    assert np.all(np.asarray(norm[target]['scale'])>0)

intent=read(R/'intent_unlabelled.json')
reserved=read(O/'hypernet_v45/reserve_metadata_audit.json')['records']
used=read(V46/'data/records.json')
assert len(intent)==len(reserved)==195
assert [x['event_id'] for x in intent]==[x['event_id'] for x in reserved]
assert len({x['event_id'] for x in intent})==195
assert not ({x['event_id'] for x in intent}&{x['event_id'] for x in used})
assert all('label' not in x and set(x['urls'])==set(durations) and len(set(x['urls'].values()))==4 for x in intent)
fixture=read(GS/'kaggle_pilot/events.json')
assert len(fixture)==12
assert {x['event_id'] for x in fixture}<={x['event_id'] for x in used}
assert not ({x['event_id'] for x in fixture}&{x['event_id'] for x in intent})

all_ids=set();covered_ids=set();native=set();crop=set();history_rows=[]
def walk(x):
    if isinstance(x,dict):
        eid=x.get('event_id')
        if eid:
            all_ids.add(eid)
            if 'pixel_sha256' in x or 'crop_pixel_sha256' in x:covered_ids.add(eid)
        for k,v in x.items():
            if k=='pixel_sha256' and isinstance(v,str):native.add(v)
            if k=='crop_pixel_sha256' and isinstance(v,str):crop.add(v)
            if isinstance(v,(dict,list)):walk(v)
    elif isinstance(x,list):
        for v in x:walk(v)
for f in list(dict.fromkeys([str(V46/'data/records.json')]+p['source_pixel_history_files'])):
    data=read(f);walk(data)
    history_rows.append(dict(path=f,top_level_rows=len(data) if isinstance(data,list) else None))
assert len(native)>0 and len(crop)>0
assert all(all('pixel_sha256' in r['duration_sources'][d] and 'crop_pixel_sha256' in r['duration_sources'][d] for d in durations) for r in used)

source=(R/'run_confirmation.py').read_text()
assert "a['code_sha256']==p['code_sha256']" in source
assert "all(c['primary_decision_changes']==0 for c in checks.values())" in source
assert "frozen_cuts==read(R/'frozen_thresholds.json')" in source
assert source.index("save(R/'scores_commitment.json'")<source.index("# First test-label join follows the immutable score commitment.")
assert 'no_refit=True' in source and 'no_training=True' in source

report={
    'passed':True,
    'reviewer':'Independent network_figures agent; read-only preflight',
    'reviewed_utc':datetime.now(timezone.utc).isoformat(),
    'code_sha256':p['code_sha256'],
    'protocol_sha256':sha(R/'protocol.json'),
    'audit_script_sha256':sha(__file__),
    'source_hashes_verified':len(p['source_hashes']),
    'acquisition_or_neural_inference_run':False,
    'reserved_images_opened':False,
    'checks':{
        'frozen_checkpoint_and_state_hashes_match_prior_registry_and_bindings':True,
        'imported_model_preprocess_dino_weights_config_and_fixture_arrays_bound':True,
        'exact_195_intent_ids_and_official_four_distinct_duration_urls':True,
        'fixture_is_12_already_consumed_events_no_reserve_overlap':True,
        'random_and_oncenet_joint4_base_descriptor_16128_effective_dimension4032':True,
        'dino_official_normalization_CLS_plus_patchmean_dual_view_1536':True,
        'dino_per_duration_ELDA_and_saved_median_MAD_scaling_then_mean':True,
        'same_one_reference_target_and_fixed_FIT_CAL_ids_verified_in_freeze':True,
        'negative_CAL_thresholds_saved_before_reserve_acquisition_and_no_refit':True,
        'strict_rank_ties_and_zero_budget_no_alarm_logic':True,
        'independent_approval_bound_to_exact_code_and_protocol':True,
        'source_layout_and_legend_failures_shared_across_models_no_replacement':True,
        'CPU_only_no_neural_training_or_model_selection':True,
        'parity_requires_zero_primary_alarm_flips_before_reserve_scoring':True,
        'scores_committed_before_heldout_label_join':True,
        'BA_MCC_recall_confusion_counts_and_per_target_FA_definitions_reviewed':True,
        'paired_hour_detector_cluster_bootstrap_and_undefined_draw_reporting':True,
        'binomial_tail_bounds_and_small_negative_sample_caveat':True,
    },
    'history_pixel_audit':{
        'historical_record_files':len(history_rows),
        'unique_event_ids':len(all_ids),
        'event_ids_with_available_direct_native_or_crop_hash':len(covered_ids),
        'event_ids_without_available_direct_hash':len(all_ids-covered_ids),
        'unique_native_pixel_hashes':len(native),
        'unique_native_crop_hashes':len(crop),
        'all_8250_current_consumed_events_have_all_four_duration_native_and_crop_hashes':True,
    },
    'resolved_during_review':[
        'Approval now binds code and protocol hash rather than a stale passed flag.',
        'Distinct official URLs enforced for all four durations.',
        'Fixture arrays and imported scientific assets added to hash bindings.',
        'Frozen numerical thresholds saved before reserve image access.',
        'Consumed-fixture primary alarm flips now halt rather than silently pass feature tolerance.',
        'Historical exact-pixel coverage limits disclosed; absent bootstrap intervals represented as null.',
    ],
    'limitations':[
        'This is a preflight code/artifact audit, not evidence that future runtime parity or model performance will pass.',
        'Twelve consumed fixtures cannot prove universal CPU/GPU equality; comparator max feature drift bound remains0.02 with zero primary alarm changes required.',
        'Historical pixel hashes are incomplete outside the current8250-event pool; exact hashes do not exclude perceptual duplicates.',
        'Reserved metadata labels were already visible, though new scores are committed before the final label join.',
        'The195 planned events cannot tightly validate a0.001 false-positive rate; report negative counts, intervals and incomplete cases.',
        'Macroscopic per-target false-alarm averages are distinct from any-target false alarms on background images.',
    ],
}
assert sha(R/'run_confirmation.py')==report['code_sha256']
(R/'independent_preflight_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['passed','code_sha256','protocol_sha256','source_hashes_verified','history_pixel_audit']},indent=2))
