"""Independently reconstruct the frozen roster from raw human votes and times.

Uses no production planner functions, SQLite aggregates, images or predictions.
"""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict,Counter
from datetime import datetime,timezone
from decimal import Decimal
import bz2,csv,json,time
DATA=Path(__file__).resolve().parent;ROOT=DATA.parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    h=sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def rank(event_id,namespace):return sha256(('HyperNet-v16-1616-'+namespace+':'+event_id).encode()).hexdigest()
def main():
    started=time.perf_counter();output=DATA/'freshness_consensus_independent_audit.json';require(not output.exists(),'Preserve prior audit')
    plan=read(DATA/'plan.json');freeze=read(DATA/'roster_freeze.json');require(digest(DATA/'plan.json')==freeze['plan_sha256'],'Plan changed')
    before={DATA/'plan.json':digest(DATA/'plan.json'),DATA/'roster_freeze.json':digest(DATA/'roster_freeze.json')}
    for relative,expected in plan['source_files'].items():
        path=ROOT/relative;require(digest(path)==expected,'Metadata hash changed '+relative);before[path]=expected
    source=DATA/'official_metadata/classifications.csv.bz2';votes=defaultdict(lambda:defaultdict(set));row_count=kept=0
    with bz2.open(source,'rt',encoding='utf-8',newline='')as stream:
        for row in csv.DictReader(stream):
            row_count+=1
            if row['Anonymous_user'].lower()=='false'and row['Workflow']=='2360'and row['Workflow_version']=='92.12':
                votes[row['Subject_id']][row['User_hash']].add(row['Classification']);kept+=1
            if row_count%2_000_000==0:print(json.dumps({'raw_annotation_rows_audited':row_count}),flush=True)
    stable={};derived={};rejected_users=0
    for event,users in votes.items():
        counts=Counter(next(iter(labels))for labels in users.values()if len(labels)==1)
        rejected_users+=sum(len(labels)>1 for labels in users.values())
        for label,n in counts.items():derived[(event,label)]=n
        if not counts:continue
        ordered=counts.most_common();total=sum(counts.values());winner,winning=ordered[0]
        if total>=5 and winning*5>=total*4 and(len(ordered)==1 or winning>ordered[1][1]):stable[event]=(winner,winning,total)
    stored={(row['subject_id'],row['label']):row['votes']for row in read(DATA/'private_consensus_counts_2360_92p12.json')}
    require(derived==stored,'Independent raw user-vote counts differ from SQLite-derived artifact')
    workflow=read(DATA/'official_metadata/workflow_2360_current.json')['workflows'][0]
    expected_choices={'BLIP','CHIRP','HELIX','TOMTE','KOIFISH','WHISTLE','1080LINE','NOGLITCH','SCRATCHY','1400RIPPLE','PAIREDDOVES',
        'EXTREMELYLOUD','POWERLINE60HZ','WANDERINGLINE','NONEOFTHEABOVE','REPEATINGBLIPS','SCATTEREDLIGHT','LIGHTMODULATION',
        'LOWFREQUENCYLINE','AIRCOMPRESSOR50HZ','LOWFREQUENCYBURST','VIOLINMODEHARMONIC'}
    require(str(workflow['id'])=='2360'and workflow['version']=='92.12'and set(workflow['tasks']['T1']['choices'])==expected_choices,'Verified common vocabulary differs')
    old=ROOT.parent/'hypernet_v15/data/gravity_spy_v15_final';old_plan=read(old/'plan.json');old_quality=read(old/'quality_admission.json')
    prior={row['event_id']:(row['ifo'],row['peak_time_ns'])for row in old_plan['events']}
    blacklist=set(prior);old10=ROOT.parent/'hypernet_v15/data/gravity_spy_official/trainingset_v1d0_metadata.csv'
    with old10.open(encoding='utf-8-sig',newline='')as stream:blacklist.update(row['gravityspy_id']for row in csv.DictReader(stream))
    require(len(prior)==7966 and len(blacklist)==10823,'Conservative prior exposure pool changed')
    relevant=set(stable)|blacklist;metadata={};conflicting=set();event_rows=0
    files=read(DATA/'official_metadata/record_5649212.json')['files']
    for source_info in files:
        with(DATA/'official_metadata'/source_info['key']).open(encoding='utf-8-sig',newline='')as stream:
            for row in csv.DictReader(stream):
                event_rows+=1;event=row['gravityspy_id']
                if event not in relevant:continue
                timestamp=int(Decimal(row['peak_time']))*1_000_000_000+int(Decimal(row['peak_time_ns']))
                item=(row['ifo'],timestamp,row['url3'])
                if event in blacklist:
                    if event not in prior:prior[event]=item[:2]
                    continue
                if event in metadata and metadata[event]!=item:conflicting.add(event)
                else:metadata[event]=item
    for event in conflicting:metadata.pop(event,None)
    candidates={event:item for event,item in metadata.items()if event in stable and event not in blacklist and item[2].startswith('https://panoptes-uploads.zooniverse.org/')}
    nodes=[(ifo,t,event,True)for event,(ifo,t)in prior.items()]+[(item[0],item[1],event,False)for event,item in candidates.items()]
    nodes.sort();components=[];last=None
    for node in nodes:
        if last is None or node[0]!=last[0]or node[1]-last[1]>4_000_000_000:components.append([])
        components[-1].append(node);last=node
    fresh=[];overlap_excluded=[]
    for component in components:
        new=[node[2]for node in component if not node[3]]
        if any(node[3]for node in component):overlap_excluded.extend(new);continue
        if not new:continue
        event=min(new,key=lambda value:rank(value,'group_representative'))
        group_id=component[0][0]+'_'+sha256('|'.join(sorted(new)).encode()).hexdigest()[:20]
        fresh.append({'event_id':event,'group_id':group_id,'detector':component[0][0],'label':stable[event][0],'members':sorted(new)})
    old_targets=old_quality['eligible_target_labels'];probes=[]
    for detector in ('H1','L1'):
        choices=[row for row in fresh if row['detector']==detector and row['label']not in old_targets]
        if choices:probes.append(min(choices,key=lambda row:rank(row['event_id'],'format_probe')))
    require([row['event_id']for row in probes]==[row['event_id']for row in plan['format_probes']],'Probe selection differs')
    probe_groups={row['group_id']for row in probes};expected=sorted((row for row in fresh if row['group_id']not in probe_groups),key=lambda row:row['event_id'])
    require([row['event_id']for row in expected]==[row['event_id']for row in plan['events']],'All-eligible TEST roster differs')
    for independently,row in zip(expected,plan['events']):
        require(all(independently[key]==row[key]for key in ('event_id','group_id','detector','label')),'Frozen event/group/label differs')
        require(independently['members']==row['group_candidate_event_ids'],'Connected group members differ')
        event=row['event_id'];require((row['detector'],row['gps_ns'],row['image_url'])==metadata[event],'Official time/URL differs')
        winner,winning,total=stable[event];lp=row['label_provenance']
        require(lp['winning_votes']==winning and lp['consistent_registered_users']==total and lp['agreement']==winning/total,'Human label provenance differs')
    counts=Counter(row['label']for row in expected);targets=[label for label in old_targets if counts[label]>=10]
    require(targets==plan['target_labels']and dict(counts)==plan['label_counts'],'Eligibility/counts differ')
    refs=plan['reference_reuse']['reference_by_label']
    require(refs=={label:old_plan['reference_by_label'][label]for label in targets},'Reference changed')
    original_calib=[row['event_id']for row in old_plan['events']if row['split']=='calibration'and row['event_id']not in set(old_quality['excluded_event_ids'])]
    require(original_calib==plan['calibration_reuse']['events']and len(original_calib)==1579,'Calibration changed')
    require(len(expected)==len({row['group_id']for row in expected})==2419 and len(targets)==14,'Frozen counts differ')
    require(not(DATA/'image_records').exists(),'TEST images were acquired before this pre-acquisition audit')
    for path,value in before.items():require(digest(path)==value,'Source changed during independent audit')
    report={'format':'hypernet.gravity_fresh.independent_metadata_audit.v16','valid':True,'created_utc':datetime.now(timezone.utc).isoformat(),
        'script_sha256':digest(__file__),'elapsed_seconds':time.perf_counter()-started,'plan_sha256':digest(DATA/'plan.json'),
        'raw_annotation_rows_streamed':row_count,'registered_exact_workflow_version_rows':kept,'conflicting_or_multilabel_user_event_pairs_excluded':rejected_users,
        'strong_consensus_subjects':len(stable),'official_event_rows_streamed':event_rows,'eligible_metadata_candidates':len(candidates),
        'candidates_excluded_by_prior_connected_groups':len(overlap_excluded),'effective_test_events_and_groups':len(expected),
        'target_labels':targets,'label_counts':dict(sorted(counts.items())),'same_references':True,'same1579calibration_events':True,
        'all_known_prior_pixel_exposure_source_ids_with_time':7966,'additional_metadata_only_ids_blacklisted':2857,
        'checks':{'independent_python_user_sets_match_sqlite':True,'single_verified_full_vocabulary_version':True,'no_ml_label_or_confidence_use':True,
            'source_time_url_match':True,'transitive_prior_overlap_excluded':True,'one_query_per_fresh_group':True,
            'all_eligible_groups_kept_after_probe_quarantine':True,'references_and_calibration_unchanged':True,'source_hashes_unchanged':True},
        'bound_artifacts':{str(path):value for path,value in before.items()},'test_pixels_read':0,'model_calls':0,'prediction_scores_read':0,
        'limits':'Freshness covers all known prior event identities/4s groups, not a proof of physical independence. Human consensus remains a selected fallible label source; no claims about naturally occurring class prevalence.'}
    with output.open('x',encoding='utf-8')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'path':str(output),'sha256':digest(output),'valid':True,'events':len(expected),'targets':len(targets)}))
if __name__=='__main__':main()
