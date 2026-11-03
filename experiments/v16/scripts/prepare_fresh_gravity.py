"""Metadata-first fresh Gravity Spy cohort; TEST pixels stay sealed until admission."""
from pathlib import Path
from hashlib import sha256,md5
from datetime import datetime,timezone
import argparse,bz2,csv,json,time,urllib.request,sqlite3,shutil,re
from collections import Counter,defaultdict
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/gravity_fresh_v16'

def digest(path):
    value=sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8')as stream:json.dump(value,stream,indent=2);stream.write('\n')
def now():return datetime.now(timezone.utc).isoformat()

def acquire_record(record_id,names=None):
    folder=DATA/'official_metadata';folder.mkdir(parents=True,exist_ok=True)
    record_path=folder/('record_'+str(record_id)+'.json')
    if record_path.exists():record=read(record_path)
    else:
        with urllib.request.urlopen('https://zenodo.org/api/records/'+str(record_id),timeout=60)as response:record=json.load(response)
        write_new(record_path,record)
    acquired=[]
    for entry in record['files']:
        if names is not None and entry['key']not in names:continue
        name=entry['key']
        if Path(name).name!=name or not(name.endswith('.csv')or name.endswith('.csv.bz2')or(str(record_id)=='5911227'and name=='retired_fulldata_min2_max50_ret0p9.hdf5')):raise ValueError('Only named metadata files accepted')
        path=folder/name;url=entry['links']['self']
        if urlsplit(url).scheme!='https' or urlsplit(url).hostname!='zenodo.org':raise ValueError('Official HTTPS metadata URL required')
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.partial')
            if temporary.exists():raise FileExistsError('Preserve incomplete download '+str(temporary))
            count=0;shown=0
            with urllib.request.urlopen(url,timeout=60)as response,temporary.open('xb')as stream:
                while True:
                    block=response.read(1<<20)
                    if not block:break
                    stream.write(block);count+=len(block)
                    if count-shown>=32*(1<<20):print(json.dumps({'metadata':name,'bytes':count}),flush=True);shown=count
            temporary.replace(path)
        m=md5()
        with path.open('rb')as stream:
            for block in iter(lambda:stream.read(1<<20),b''):m.update(block)
        if path.stat().st_size!=entry['size']or 'md5:'+m.hexdigest()!=entry['checksum']:raise ValueError('Publisher file digest/size mismatch '+name)
        proof={'record_id':str(record_id),'record_url':'https://zenodo.org/records/'+str(record_id),'metadata_url':url,
            'file':name,'bytes':path.stat().st_size,'publisher_checksum':entry['checksum'],'sha256':digest(path),
            'record_json_sha256':digest(record_path),'license':record['metadata'].get('license'),'acquired_utc':now(),
            'https_certificate_validation':True,'test_image_pixels_acquired':False}
        proof_path=path.with_name(path.name+'.provenance.json')
        if not proof_path.exists():write_new(proof_path,proof)
        elif read(proof_path)['sha256']!=proof['sha256']:raise ValueError('Existing metadata provenance changed')
        acquired.append(proof)
    return acquired

def ingest_votes():
    """Stream local human votes into SQLite; no images or machine-label fields."""
    source=DATA/'official_metadata/classifications.csv.bz2';database=DATA/'private_volunteer_votes.sqlite'
    if database.exists():raise FileExistsError('Preserve original vote database')
    connection=sqlite3.connect(database);connection.execute('PRAGMA journal_mode=WAL');connection.execute('PRAGMA synchronous=NORMAL')
    connection.execute('CREATE TABLE votes(subject TEXT,user TEXT,workflow INTEGER,version TEXT,ballot INTEGER,label TEXT)')
    workflows={2360,2117,7766,7767};counts={};rows=0;kept=0;batch=[]
    with bz2.open(source,'rt',encoding='utf-8',newline='')as stream:
        for row in csv.DictReader(stream):
            rows+=1;workflow=int(row['Workflow']);counts[workflow]=counts.get(workflow,0)+1
            if row['Anonymous_user'].lower()!='false'or workflow not in workflows:continue
            batch.append((row['Subject_id'],row['User_hash'],workflow,row['Workflow_version'],int(row['Classification_id']),row['Classification']))
            if len(batch)>=20000:
                connection.executemany('INSERT INTO votes VALUES(?,?,?,?,?,?)',batch);connection.commit();kept+=len(batch);batch=[]
                if kept%200000==0:print(json.dumps({'volunteer_rows_read':rows,'full_workflow_registered_rows':kept}),flush=True)
    if batch:connection.executemany('INSERT INTO votes VALUES(?,?,?,?,?,?)',batch);connection.commit();kept+=len(batch)
    connection.execute('CREATE INDEX user_event_workflow ON votes(workflow,subject,user)');connection.commit()
    # A registered user contributes at most one vote per event/workflow, only if
    # all their ballots/selected labels agree. Multi-label and changing users
    # are excluded rather than silently weighted multiple times.
    connection.execute('CREATE TABLE stable_votes AS SELECT subject,user,workflow,MIN(label) AS label FROM votes GROUP BY workflow,subject,user HAVING COUNT(DISTINCT label)=1')
    connection.execute('CREATE INDEX stable_subject ON stable_votes(workflow,subject)');connection.commit()
    totals={}
    for workflow in sorted(workflows):
        result=[]
        for row in connection.execute('SELECT subject,label,COUNT(*) FROM stable_votes WHERE workflow=? GROUP BY subject,label ORDER BY subject,label',(workflow,)):
            result.append({'subject_id':row[0],'label':row[1],'votes':row[2]})
        write_new(DATA/('private_consensus_counts_'+str(workflow)+'.json'),result)
        subjects={}
        for row in result:subjects.setdefault(row['subject_id'],[]).append(row)
        totals[workflow]={}
        for minimum in(3,5):
            for fraction in(.8,.9,1.):
                eligible=[]
                for subject,items in subjects.items():
                    total=sum(item['votes']for item in items);top=sorted(items,key=lambda item:(-item['votes'],item['label']))
                    if total>=minimum and top[0]['votes']/total>=fraction and(len(top)==1 or top[0]['votes']>top[1]['votes']):eligible.append(top[0]['label'])
                from collections import Counter
                totals[workflow][str(minimum)+'votes_'+str(fraction)]={'subjects':len(eligible),'labels':dict(sorted(Counter(eligible).items()))}
    connection.execute('PRAGMA wal_checkpoint(TRUNCATE)');connection.close()
    report={'format':'hypernet.gravity_fresh.human_vote_availability.v16','created_utc':now(),'source_sha256':digest(source),
        'raw_rows':rows,'retained_registered_rows':kept,'raw_rows_per_workflow':counts,'availability':totals,
        'rule':'Within each full-vocabulary workflow, one registered user/event vote only when all their labels agree; no anonymous votes or machine labels.',
        'metadata_only':True,'test_pixels_acquired':False,'model_scores_used':False,
        'license':'CC BY-NC-ND4.0; raw/derived row-level annotations stay local, no public-repository export.'}
    write_new(DATA/'human_vote_availability.json',report)
    print(json.dumps(report,indent=2))

def consensus_counts(rows,minimum=5,fraction=.8):
    grouped=defaultdict(list)
    for row in rows:grouped[str(row['subject_id'])].append(row)
    accepted={}
    for subject,items in grouped.items():
        total=sum(item['votes']for item in items);ordered=sorted(items,key=lambda item:(-item['votes'],item['label']))
        if total<minimum or ordered[0]['votes']/total<fraction or(len(ordered)>1 and ordered[0]['votes']==ordered[1]['votes']):continue
        accepted[subject]={'label':ordered[0]['label'],'winning_votes':ordered[0]['votes'],'consistent_registered_users':total,'agreement':ordered[0]['votes']/total}
    return accepted

def connected_time_groups(records,gap_ns=4_000_000_000):
    """Conservative same-detector connected4s windows, including prior events."""
    groups=[];current=[];previous=None
    for row in sorted(records,key=lambda row:(row['detector'],row['gps_ns'],row['event_id'])):
        if previous is not None and(row['detector']!=previous['detector']or row['gps_ns']-previous['gps_ns']>gap_ns):
            groups.append(current);current=[]
        current.append(row);previous=row
    if current:groups.append(current)
    return groups

def seeded_rank(value,namespace='test'):
    return sha256(('HyperNet-v16-1616-'+namespace+':'+str(value)).encode()).hexdigest()

def candidate_inventory():
    """Availability only; no images, no score reads, no model selection."""
    destination=DATA/'metadata_candidate_inventory.json'
    if destination.exists():raise FileExistsError(destination)
    metadata=DATA/'official_metadata';record=read(metadata/'record_5649212.json')
    for entry in record['files']:
        path=metadata/entry['key'];proof=read(path.with_name(path.name+'.provenance.json'))
        if digest(path)!=proof['sha256']or proof['publisher_checksum']!=entry['checksum']:raise ValueError('Official event metadata changed')
    counts_path=DATA/'private_consensus_counts_2360_92p12.json';consensus=consensus_counts(read(counts_path))
    old_root=ROOT.parent/'hypernet_v15';old_plan_path=old_root/'data/gravity_spy_v15_final/plan.json';old_plan=read(old_plan_path)
    prior={row['event_id']:{'event_id':row['event_id'],'detector':row['ifo'],'gps_ns':row['peak_time_ns'],'prior':True}for row in old_plan['events']}
    v10=old_root/'data/gravity_spy_official/trainingset_v1d0_metadata.csv'
    with v10.open(newline='',encoding='utf-8-sig')as stream:extra_prior_ids={row['gravityspy_id']for row in csv.DictReader(stream)}
    blacklist=set(prior)|extra_prior_ids
    wanted=set(consensus)|blacklist;found={};ambiguous=set();extra_times={};source_files={};total_rows=0
    for entry in sorted(record['files'],key=lambda item:item['key']):
        path=metadata/entry['key'];source_files[path.relative_to(ROOT).as_posix()]=digest(path)
        with path.open(newline='',encoding='utf-8-sig')as stream:
            for row in csv.DictReader(stream):
                total_rows+=1;event_id=row['gravityspy_id']
                if event_id not in wanted:continue
                seconds=int(Decimal(row['peak_time']));ns=int(Decimal(row['peak_time_ns']));detector=row['ifo']
                gps_ns=seconds*1_000_000_000+ns
                if detector not in ('H1','L1')or not 0<=ns<1_000_000_000:raise ValueError('Invalid official time/detector')
                if event_id in blacklist:
                    if event_id not in prior:extra_times[event_id]={'event_id':event_id,'detector':detector,'gps_ns':gps_ns,'prior':True}
                    continue
                url=row['url3'];parsed=urlsplit(url)
                if parsed.scheme!='https'or parsed.hostname!='panoptes-uploads.zooniverse.org':continue
                item={'event_id':event_id,'detector':detector,'gps':seconds+ns/1e9,'gps_ns':gps_ns,'image_url':url,
                    'image_path':'images/'+event_id+'_2.0.png','duration_seconds':2.,'source_csv':entry['key'],
                    'source_url':'https://zenodo.org/records/5649212','label':consensus[event_id]['label'],'prior':False,
                    'label_provenance':{'source':'https://zenodo.org/records/13904422','human_only':True,'machine_labels_used':False,
                        'workflow':2360,'workflow_version':'92.12',**consensus[event_id]}}
                if event_id in found and any(found[event_id][key]!=item[key]for key in ('detector','gps_ns','image_url')):ambiguous.add(event_id)
                else:found[event_id]=item
    for event_id in ambiguous:found.pop(event_id,None)
    prior.update(extra_times)
    fresh=[];near_prior=set();groups=connected_time_groups([*prior.values(),*found.values()])
    for members in groups:
        candidates=[row for row in members if not row['prior']]
        if any(row['prior']for row in members):near_prior.update(row['event_id']for row in candidates);continue
        if not candidates:continue
        group_id=members[0]['detector']+'_'+sha256('|'.join(sorted(row['event_id']for row in members)).encode()).hexdigest()[:20]
        representative=min(candidates,key=lambda row:seeded_rank(row['event_id'],'group_representative'))
        fresh.append({**representative,'group_id':group_id,'group_candidate_event_ids':sorted(row['event_id']for row in candidates)})
    old_quality=read(old_root/'data/gravity_spy_v15_final/quality_admission.json');old_targets=old_quality['eligible_target_labels']
    # Two metadata-predeclared non-target examples may inspect the historical
    # plotting format. Their entire groups are quarantined from every TEST row.
    probes=[]
    for detector in ('H1','L1'):
        options=[row for row in fresh if row['detector']==detector and row['label']not in old_targets]
        if options:probes.append(min(options,key=lambda row:seeded_rank(row['event_id'],'format_probe')))
    probe_groups={row['group_id']for row in probes};fresh=[row for row in fresh if row['group_id']not in probe_groups]
    availability=Counter(row['label']for row in fresh);targets=[label for label in old_targets if availability[label]>=10]
    selected=[]
    for label in targets:
        selected.extend(sorted((row for row in fresh if row['label']==label),key=lambda row:seeded_rank(row['event_id']))[:128])
    background=sorted((row for row in fresh if row['label']not in targets),key=lambda row:seeded_rank(row['event_id'],'background'))
    selected.extend(background[:max(0,4096-len(selected))]);selected.sort(key=lambda row:row['event_id'])
    # Only stored label/time/URL metadata is consulted. Combined ML+human HDF5
    # was acquired for possible historical ID linkage but is not read or used.
    result={'format':'hypernet.gravity_fresh.metadata_inventory.v16','created_utc':now(),'metadata_only':True,
        'source_files':source_files,'volunteer_csv_sha256':digest(metadata/'classifications.csv.bz2'),
        'consensus_counts_sha256':digest(counts_path),'verified_workflow_sha256':digest(metadata/'workflow_2360_current.json'),
        'prior_v15_plan_sha256':digest(old_plan_path),'additional_prior_v10_metadata_sha256':digest(v10),
        'rule':{'workflow':2360,'version':'92.12','minimum_distinct_consistent_registered_users':5,'minimum_consensus_fraction':.8,
            'time_group_seconds':4,'one_event_per_time_group':True,'minimum_fresh_groups_per_target':10,'maximum_per_target':128,
            'maximum_test_events':4096,'seed':1616,'background':'All available other-label groups up to remaining budget, ranked independently by fixed hash; no relabelling or model scores.'},
        'counts':{'official_event_metadata_rows':total_rows,'human_consensus_subjects':len(consensus),'blacklisted_ids':len(blacklist),
            'prior_ids_with_known_times':len(prior),'additional_metadata_only_blacklisted_ids_without_time':len(blacklist-set(prior)),
            'consensus_ids_also_blacklisted':len(set(consensus)&blacklist),'unmatched_nonblacklisted_consensus_ids':len(set(consensus)-blacklist-set(found)-ambiguous),
            'ambiguous_event_metadata_ids':len(ambiguous),'eligible_metadata_events_before_time_filter':len(found),
            'events_removed_by_prior_connected_group':len(near_prior),'format_probe_groups':len(probes),'available_fresh_groups':len(fresh),
            'eligible_targets':len(targets),'selected_test_events':len(selected),'selected_target_events':sum(row['label']in targets for row in selected),
            'selected_background_events':sum(row['label']not in targets for row in selected)},
        'available_groups_per_label':dict(sorted(availability.items())),'target_labels':targets,
        'excluded_old_targets':{label:availability[label]for label in old_targets if label not in targets},
        'selected_counts_per_label':dict(sorted(Counter(row['label']for row in selected).items())),
        'events':selected,'format_probes':probes,'all_fresh_group_representatives':fresh,
        'prior_ids':sorted(blacklist),'prior_time_records':list(prior.values()),'ambiguous_metadata_ids':sorted(ambiguous),
        'unmatched_consensus_ids':sorted(set(consensus)-blacklist-set(found)-ambiguous),
        'scope':'Human-only high-consensus sample from one verified task/version; ML-routed volunteer exposure and consensus filtering induce selection bias. Class stratification is not natural event prevalence. Row-level annotations remain local under source license.',
        'combined_ml_volunteer_labels_used':False,'test_image_pixels_acquired':False,'model_scores_used':False}
    write_new(destination,result)
    print(json.dumps({key:result[key]for key in ('counts','available_groups_per_label','target_labels','excluded_old_targets','selected_counts_per_label')},indent=2))

def freeze_plan():
    """Freeze all strict eligible groups, per metadata-only approved amendment."""
    destination=DATA/'plan.json'
    if destination.exists():raise FileExistsError(destination)
    inventory_path=DATA/'metadata_candidate_inventory.json';inventory=read(inventory_path)
    old=ROOT.parent/'hypernet_v15/data/gravity_spy_v15_final';old_plan=read(old/'plan.json');old_qc=read(old/'quality_admission.json')
    targets=inventory['target_labels'];events=[]
    for row in sorted(inventory['all_fresh_group_representatives'],key=lambda row:row['event_id']):
        event={key:value for key,value in row.items()if key!='prior'};event['split']='test';events.append(event)
    if len({row['event_id']for row in events})!=len(events)or len({row['group_id']for row in events})!=len(events):raise ValueError('Duplicate TEST event/group')
    excluded=set(old_qc['excluded_event_ids']);calibration=[row['event_id']for row in old_plan['events']if row['split']=='calibration'and row['event_id']not in excluded]
    if len(calibration)!=1579 or len(targets)!=14 or len(events)!=2419:raise ValueError('Reviewed metadata counts changed')
    sources=dict(inventory['source_files'])
    for name in ('data/gravity_fresh_v16/official_metadata/classifications.csv.bz2','data/gravity_fresh_v16/official_metadata/workflow_2360_current.json',
        'data/gravity_fresh_v16/private_consensus_counts_2360_92p12.json','data/gravity_fresh_v16/metadata_candidate_inventory.json'):
        sources[name]=digest(ROOT/name)
    snapshot=DATA/'planner_at_roster_freeze.py';shutil.copyfile(__file__,snapshot);sources[snapshot.relative_to(ROOT).as_posix()]=digest(snapshot)
    counts=Counter(row['label']for row in events)
    plan={'format':'hypernet.gravity_fresh.plan.v16','created_utc':now(),'status':'frozen_metadata_only_test_sealed',
        'test_pixels_acquired':False,'test_scores_computed':False,'source_files':sources,
        'sampling_amendment':{'original_inventory_sha256':digest(inventory_path),'original_capped_sample_events':1558,
            'final_rule':'All eligible strict-vocabulary fresh group representatives; budget4096not reached;128per-target cap removed before pixel acquisition or model scoring.',
            'reason':'Avoid arbitrary caps in the small high-consensus pool and increase actual negative denominators without weakening vocabulary, agreement or freshness requirements.',
            'approved_before_pixels_and_scores':True,'selection_seed':1616,'model_scores_used':False},
        'consensus':{'workflow':2360,'workflow_version':'92.12','minimum_consistent_registered_users':5,'agreement_fraction':.8,
            'one_vote_per_user_event':True,'inconsistent_or_multilabel_users':'Exclude that user/event within the fixed workflow-version when their ballots contain more than one distinct label.',
            'anonymous_votes_used':False,'machine_labels_used':False,'source':'https://zenodo.org/records/13904422','license':'CC BY-NC-ND4.0'},
        'event_metadata':{'source':'https://zenodo.org/records/5649212','license':'CC BY4.0','columns_used':'GravitySpy ID, detector, exact GPS seconds/nanoseconds and official2-second URL. No ML labels/confidences used.'},
        'freshness':{'same_detector_connected_window_seconds':4,'one_event_per_group':True,
            'known_exposure_basis':'Entire7,966-event v1.1 source pool, covering allv15 and earlierv0.1/v0.2/v0.3 scored, reference and600standardization images; earlier scripts explicitly load v1d1.',
            'additional_conservative_id_blacklist':'Allv1d0metadata IDs;2,857additional metadata-only IDs lack event times in the official O1–O3 index but are excluded by identity. No known pixel exposure depends on those missing times.',
            'prior_v15_plan_sha256':digest(old/'plan.json'),'prior_quality_sha256':digest(old/'quality_admission.json'),
            'metadata_candidate_inventory_sha256':digest(inventory_path),'format_probe_groups_quarantined':[row['group_id']for row in inventory['format_probes']]},
        'target_labels':targets,'excluded_old_targets_below10fresh_groups':inventory['excluded_old_targets'],
        'reference_reuse':{'root':str(old.resolve()),'reference_by_label':{label:old_plan['reference_by_label'][label]for label in targets},'images_per_target':1,'weight_updates':0},
        'calibration_reuse':{'root':str(old.resolve()),'events':calibration,'count':1579,'unchanged_from_v15':True,'extra_real_calibration_data':True},
        'image_contract':{'duration_seconds':2.,'expected_raw_size_wh':[800,600],'fixed_crop_xyxy':[105,66,671,532],
            'decoded_rgb_shape':[466,566,3],'pixel_sha256':'SHA256 of contiguous uint8RGB bytes, matching v15; no dtype/shape prefix.',
            'resize':False,'exif_orientation':'Must be1or absent; otherwise reject instead of silently transforming coordinates.',
            'image_records_path':'image_records/<event_id>.json','raw_path':'raw/<event_id>_2.0.png','image_path':'images/<event_id>_2.0.png'},
        'blind_quality_rule':{'exact_constant_rgb_plot':'Exclude iff every croppedRGB pixel equals its top-leftRGBpixel; same rule asv15.',
            'exact_prior_pixel_overlap':'Exclude any raw or cropped exact duplicate of an available earlier-project image or quarantined format probe.',
            'within_test_nonconstant_duplicates':'Exclude all members of an identical croppedRGB group; no score-driven representative selection.',
            'unavailable_or_invalid_source':'Record failed acquisition/invalid dimensions; no replacement. Scoring admission requires explicit final inventory binding.',
            'replacements':False,'manual_test_image_curation':False},
        'counts':{'test_events_before_blind_quality':len(events),'test_groups':len(events),'target_types':len(targets),
            'events_with_enrolled_target':sum(counts[label]for label in targets),'other_label_background_events':sum(n for label,n in counts.items()if label not in targets),
            'reference_images':len(targets),'additional_calibration_events':1579,'natural_weight_training_images':0},
        'label_counts':dict(sorted(counts.items())),
        'per_target':{label:{'positive_events':counts[label],'negative_events':len(events)-counts[label]}for label in targets},
        'events':events,'format_probes':inventory['format_probes'],
        'limitations':['Human consensus is a fallible proxy, not independently adjudicated detector truth. Volunteer routing used earlier ML models even though our labels use only registered human votes.',
            'The fixed task/version, agreement rule and availability filters create a selected cohort; its label mixture is not a claim about natural event prevalence.',
            'Roughly2,000negatives per target cannot certify extremely rare false-alarm rates. Normalize per10,000 only with actual denominators and binomial uncertainty.',
            'Individual volunteer data and derived row-level human annotations remain local; no export to the public repository under the project license.']}
    write_new(destination,plan)
    write_new(DATA/'roster_freeze.json',{'format':'hypernet.gravity_fresh.roster_freeze.v16','created_utc':now(),'plan_sha256':digest(destination),
        'planner_snapshot_sha256':digest(snapshot),'test_pixels_acquired':False,'test_scores_computed':False,'label_counts':plan['label_counts'],'target_labels':targets})
    write_new(DATA/'format_probe_plan.json',{'format':'hypernet.gravity_fresh.format_probe_plan.v16','plan_sha256':digest(destination),
        'events':inventory['format_probes'],'selection':'First fixedseed1616hash non-target fresh consensus representative per detector; whole groups quarantined before any pixel acquisition.',
        'test_pixels':False,'model_inference_allowed':False})
    print(json.dumps({'plan_sha256':digest(destination),'counts':plan['counts'],'label_counts':plan['label_counts'],'targets':targets},indent=2))

VERSION_VOTE_QUERY='''SELECT subject,MIN(label),COUNT(*) FROM
 (SELECT subject,user,MIN(label)label FROM votes WHERE workflow=? AND version=?
 GROUP BY subject,user HAVING COUNT(DISTINCT label)=1)
 GROUP BY subject,label ORDER BY subject,label'''

def verified_version_counts():
    connection=sqlite3.connect(DATA/'private_volunteer_votes.sqlite')
    rows=connection.execute(VERSION_VOTE_QUERY,(2360,'92.12')).fetchall();connection.close()
    records=[{'subject_id':subject,'label':label,'votes':n}for subject,label,n in rows]
    path=DATA/'private_consensus_counts_2360_92p12.json'
    if path.exists():
        if read(path)!=records:raise ValueError('Existing verified-version vote counts differ')
    else:path.write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps({'subject_label_counts':len(records),'consensus_subjects':len(consensus_counts(records)),'sha256':digest(path)}))

def verify_plan():
    plan=read(DATA/'plan.json');freeze=read(DATA/'roster_freeze.json')
    if digest(DATA/'plan.json')!=freeze['plan_sha256']:raise ValueError('Frozen data plan changed')
    for name,expected in plan['source_files'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen metadata changed '+name)
    return plan

def verify_model_protocol(protocol_path,expected_sha):
    protocol_path=Path(protocol_path).resolve()
    if digest(protocol_path)!=expected_sha:raise ValueError('Actual model protocol hash differs')
    protocol=read(protocol_path)
    if protocol.get('fresh_plan_sha256')!=digest(DATA/'plan.json'):raise ValueError('Protocol does not bind this fresh plan')
    if protocol.get('model_selection_complete')is not True or protocol.get('test_scores_computed_before_freeze')is not False:raise ValueError('Model must be frozen before TEST')
    for name in ('behavior_files','checkpoint_files'):
        if not isinstance(protocol.get(name),dict)or not protocol[name]:raise ValueError('Missing frozen '+name)
        for key,expected in protocol[name].items():
            path=(ROOT/key).resolve()
            if digest(path)!=expected:raise ValueError('Frozen model/source binding differs '+key)
    return protocol

def fetch_event(event,destination,role):
    """Exact official native plot and fixed crop; never loads a model."""
    import io
    from PIL import Image
    destination=Path(destination);event_id=event['event_id']
    if not re.fullmatch(r'[A-Za-z0-9]{10}',event_id)or event['image_path']!='images/'+event_id+'_2.0.png':raise ValueError('Noncanonical source ID/path')
    record_path=destination/'image_records'/f'{event_id}.json'
    raw_path=destination/'raw'/f'{event_id}_2.0.png';crop_path=destination/event['image_path']
    for path in (record_path,raw_path,crop_path):path.parent.mkdir(parents=True,exist_ok=True)
    if record_path.exists():
        record=read(record_path)
        if record['image_url']!=event['image_url']or digest(raw_path)!=record['raw_sha256']or digest(crop_path)!=record['crop_png_sha256']:raise ValueError('Existing acquisition changed')
        return record
    parsed=urlsplit(event['image_url'])
    if parsed.scheme!='https'or parsed.hostname!='panoptes-uploads.zooniverse.org':raise ValueError('Official HTTPS image URL required')
    raw=None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(event['image_url'],timeout=45)as response:
                if urlsplit(response.url).scheme!='https':raise ValueError('NonHTTPS image redirect')
                raw=response.read(4_000_001);final_url=response.url
            break
        except OSError:
            if attempt==2:raise
            time.sleep(.25*(attempt+1))
    if len(raw)>4_000_000:raise ValueError('Image exceeds source size bound')
    with Image.open(io.BytesIO(raw))as image:
        if image.size!=(800,600)or image.getexif().get(274,1)!=1:raise ValueError('Unexpected source dimensions/orientation; do not adapt TEST crop')
        crop=image.convert('RGB').crop((105,66,671,532))
    if raw_path.exists()or crop_path.exists():raise FileExistsError('Preserve incomplete acquisition files')
    raw_path.write_bytes(raw);crop.save(crop_path)
    record={**event,'role':role,'raw_path':raw_path.relative_to(destination).as_posix(),'raw_sha256':sha256(raw).hexdigest(),
        'raw_bytes':len(raw),'raw_dimensions':[800,600],'crop_png_sha256':digest(crop_path),'crop_rgb_sha256':sha256(crop.tobytes()).hexdigest(),
        'crop_dimensions':[566,466],'crop_xyxy':[105,66,671,532],'https_certificate_validation':True,'response_url':final_url,
        'image_source_license':'CC BY4.0 per official event release','human_annotation_source_license':'CC BY-NC-ND4.0; keep row-level annotations local',
        'acquired_utc':now()}
    write_new(record_path,record);return record

def fetch_probes():
    plan=verify_plan();probe_plan=read(DATA/'format_probe_plan.json')
    if probe_plan['plan_sha256']!=digest(DATA/'plan.json')or probe_plan['events']!=plan['format_probes']:raise ValueError('Quarantined probe plan changed')
    records=[fetch_event(event,DATA/'format_probes','format_probe')for event in plan['format_probes']]
    audit={'format':'hypernet.gravity_fresh.format_probe_audit.v16','created_utc':now(),'plan_sha256':digest(DATA/'plan.json'),
        'probe_plan_sha256':digest(DATA/'format_probe_plan.json'),'valid_native_dimensions_and_fixed_crop':True,
        'test_events_acquired':0,'model_calls':0,'quarantined_group_ids':[event['group_id']for event in plan['format_probes']],
        'records':{record['event_id']:digest(DATA/'format_probes/image_records'/f"{record['event_id']}.json")for record in records}}
    path=DATA/'format_probe_audit.json'
    if not path.exists():write_new(path,audit)
    print(json.dumps({'probes':[record['event_id']for record in records],'audit_sha256':digest(path),'test_events_acquired':0}))

def fetch_test(protocol_path,expected_sha,workers=4):
    plan=verify_plan();verify_model_protocol(protocol_path,expected_sha)
    if workers not in range(1,9):raise ValueError('Bound download workers to1..8')
    admission={'protocol_path':str(Path(protocol_path).resolve()),'protocol_sha256':expected_sha,'fresh_plan_sha256':digest(DATA/'plan.json'),
        'test_pixel_acquisition_authorized':True,'created_utc':now(),'before_first_test_fetch':not(DATA/'image_records').exists()}
    gate=DATA/'acquisition_admission.json'
    if gate.exists():
        prior=read(gate)
        if any(prior[key]!=admission[key]for key in ('protocol_path','protocol_sha256','fresh_plan_sha256')):raise ValueError('Acquisition protocol changed')
    else:write_new(gate,admission)
    records=[];failures=[]
    with ThreadPoolExecutor(max_workers=workers)as pool:
        jobs={pool.submit(fetch_event,event,DATA,'test'):event['event_id']for event in plan['events']}
        for job in as_completed(jobs):
            try:records.append(job.result())
            except Exception as error:failures.append({'event_id':jobs[job],'error':str(error)})
            if (len(records)+len(failures))%100==0:print(json.dumps({'acquired':len(records),'failures':len(failures),'requested':len(jobs)}),flush=True)
    verify_model_protocol(protocol_path,expected_sha)
    result={'format':'hypernet.gravity_fresh.acquisition.v16','created_utc':now(),'plan_sha256':digest(DATA/'plan.json'),
        'acquisition_admission_sha256':digest(gate),'protocol_sha256':expected_sha,'requested':len(plan['events']),
        'downloaded_and_verified':len(records),'failures':failures,'event_ids':sorted(record['event_id']for record in records),'replacement_sampling':False}
    attempt=len(list(DATA.glob('acquisition_attempt_*.json')))+1;write_new(DATA/f'acquisition_attempt_{attempt:02d}.json',result)
    if failures:raise RuntimeError('Acquisition failed for fixed events; preserve failures, no replacement')
    write_new(DATA/'acquisition_test.json',result);print(json.dumps({'complete':True,'events':len(records),'sha256':digest(DATA/'acquisition_test.json')}))

def blind_quality_audit():
    import numpy as np
    from PIL import Image
    plan=verify_plan();acquisition=read(DATA/'acquisition_test.json');gate=read(DATA/'acquisition_admission.json')
    verify_model_protocol(gate['protocol_path'],gate['protocol_sha256'])
    if acquisition['failures']or acquisition['event_ids']!=sorted(event['event_id']for event in plan['events']):raise ValueError('Incomplete fixed acquisition')
    prior_raw=set();prior_pixels=set();old=ROOT.parent/'hypernet_v15/data/gravity_spy_v15_final/image_records'
    prior_records=list(old.glob('*.json'))+list((DATA/'format_probes/image_records').glob('*.json'))
    for path in prior_records:
        record=read(path);prior_raw.add(record['raw_sha256']);prior_pixels.add(record['crop_rgb_sha256'])
    rows=[];groups=defaultdict(list);bindings={}
    for event in plan['events']:
        path=DATA/'image_records'/f"{event['event_id']}.json";record=read(path);bindings[event['event_id']]=digest(path)
        if any(record[key]!=value for key,value in event.items()):raise ValueError('Acquired event metadata changed')
        raw=DATA/record['raw_path'];crop=DATA/event['image_path']
        if digest(raw)!=record['raw_sha256']or digest(crop)!=record['crop_png_sha256']:raise ValueError('Image hashes changed')
        with Image.open(raw)as image:expected=np.asarray(image.convert('RGB').crop((105,66,671,532)))
        with Image.open(crop)as image:rgb=np.asarray(image.convert('RGB'))
        if rgb.shape!=(466,566,3)or not np.array_equal(rgb,expected)or sha256(rgb.tobytes()).hexdigest()!=record['crop_rgb_sha256']:raise ValueError('Decoded/fixed-crop pixels changed')
        reasons=[]
        if np.all(rgb==rgb[0,0]):reasons.append('exact_constant_rgb_plot')
        if record['raw_sha256']in prior_raw or record['crop_rgb_sha256']in prior_pixels:reasons.append('exact_prior_or_probe_pixels')
        row={'event_id':event['event_id'],'label':event['label'],'group_id':event['group_id'],'reasons':reasons,'crop_rgb_sha256':record['crop_rgb_sha256']}
        rows.append(row)
        if 'exact_constant_rgb_plot'not in reasons:groups[record['crop_rgb_sha256']].append(row)
    for identical in groups.values():
        if len(identical)>1:
            for row in identical:row['reasons'].append('within_test_nonconstant_duplicate')
    eligible=[row for row in rows if not row['reasons']]
    result={'format':'hypernet.gravity_fresh.blind_quality.v16','created_utc':now(),'fresh_plan_sha256':digest(DATA/'plan.json'),
        'acquisition_sha256':digest(DATA/'acquisition_test.json'),'protocol_sha256':gate['protocol_sha256'],
        'eligible_event_ids':[row['event_id']for row in eligible],'excluded':[row for row in rows if row['reasons']],
        'image_record_sha256':bindings,'requested':len(rows),'eligible':len(eligible),'label_counts':dict(sorted(Counter(row['label']for row in eligible).items())),
        'prior_and_probe_image_records_checked':len(prior_records),'model_scores_used':False,'manual_image_curation':False,'replacement_sampling':False}
    write_new(DATA/'blind_quality_audit.json',result);print(json.dumps({'requested':len(rows),'eligible':len(eligible),'excluded':len(rows)-len(eligible),'sha256':digest(DATA/'blind_quality_audit.json')}))

class FreshGravityDataset:
    """RGB-only loader gated by exact frozen protocol/QC/record bindings."""
    def __init__(self,admission_path):
        self.plan=verify_plan();self.admission_path=Path(admission_path).resolve();self.admission=read(self.admission_path)
        a=self.admission
        if a.get('test_scoring_authorized')is not True or a.get('fresh_plan_sha256')!=digest(DATA/'plan.json'):raise PermissionError('Fresh scoring admission required')
        verify_model_protocol(a['protocol_path'],a['protocol_sha256'])
        if a['blind_quality_audit_sha256']!=digest(DATA/'blind_quality_audit.json'):raise ValueError('Blind QC binding differs')
        self.quality=read(DATA/'blind_quality_audit.json')
        if self.quality['fresh_plan_sha256']!=a['fresh_plan_sha256']or self.quality['protocol_sha256']!=a['protocol_sha256']:raise ValueError('QC/plan/protocol binding differs')
        if a['image_record_sha256']!=self.quality['image_record_sha256']:raise ValueError('Full image-record inventory binding differs')
        if a.get('effective_event_ids')!=self.quality['eligible_event_ids']:raise ValueError('Effective scoring roster differs from blind QC')
        self.records={event['event_id']:event for event in self.plan['events']};self.eligible=set(self.quality['eligible_event_ids'])
        self.initial_admission_sha256=digest(self.admission_path)

    def query_events(self):return tuple(dict(event)for event in self.plan['events']if event['event_id']in self.eligible)

    def load_rgb(self,event_id):
        import numpy as np
        from PIL import Image
        if event_id not in self.eligible:raise PermissionError('Event outside frozen QC-valid TEST roster')
        if digest(self.admission_path)!=self.initial_admission_sha256:raise ValueError('Scoring admission changed')
        path=DATA/'image_records'/f'{event_id}.json'
        if digest(path)!=self.admission['image_record_sha256'][event_id]:raise ValueError('Bound image record changed')
        record=read(path);event=self.records[event_id]
        if any(record[key]!=value for key,value in event.items()):raise ValueError('Event record changed')
        image_path=DATA/event['image_path']
        if digest(image_path)!=record['crop_png_sha256']:raise ValueError('Crop file changed')
        with Image.open(image_path)as image:rgb=np.asarray(image.convert('RGB')).copy()
        if rgb.shape!=(466,566,3)or sha256(rgb.tobytes()).hexdigest()!=record['crop_rgb_sha256']:raise ValueError('Crop pixels changed')
        return rgb

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['volunteer-metadata','event-metadata','subject-mapping-metadata','ingest-votes','verified-version-counts','candidate-inventory','freeze-plan','fetch-probes','fetch-test','blind-quality','inspect-header'])
    parser.add_argument('--protocol',type=Path);parser.add_argument('--expected-protocol-sha');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='volunteer-metadata':print(json.dumps(acquire_record('13904422'),indent=2))
    elif args.stage=='event-metadata':print(json.dumps(acquire_record('5649212'),indent=2))
    elif args.stage=='subject-mapping-metadata':print(json.dumps(acquire_record('5911227'),indent=2))
    elif args.stage=='ingest-votes':ingest_votes()
    elif args.stage=='candidate-inventory':candidate_inventory()
    elif args.stage=='freeze-plan':freeze_plan()
    elif args.stage=='verified-version-counts':verified_version_counts()
    elif args.stage=='fetch-probes':fetch_probes()
    elif args.stage=='fetch-test':
        if not args.protocol or not args.expected_protocol_sha:raise ValueError('Frozen protocol path/hash required')
        fetch_test(args.protocol,args.expected_protocol_sha,args.workers)
    elif args.stage=='blind-quality':blind_quality_audit()
    else:
        path=DATA/'official_metadata/classifications.csv.bz2'
        with bz2.open(path,'rt',encoding='utf-8',newline='')as stream:
            reader=csv.DictReader(stream);print(json.dumps({'columns':reader.fieldnames,'first_rows':[next(reader)for _ in range(3)]},indent=2))

if __name__=='__main__':main()
