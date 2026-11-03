"""Bind completed training and DEV calibration before the first fresh TEST pixel.

This creates an immutable admission record, not a claim of model superiority.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_coco_v17 import sha, read, write_new, verify_plan, PLAN, DATA


def main(args):
    plan = verify_plan()
    if (DATA/'acquisition_admission.json').exists() or (DATA/'acquisition.json').exists():
        raise ValueError('Fresh TEST acquisition already admitted; do not reselect a model')
    if (DATA/'images').exists() and any((DATA/'images').iterdir()):
        raise ValueError('Fresh TEST pixels already exist before freeze')
    primary, control, random = (Path(p).resolve() for p in (args.primary, args.control, args.random))
    completions = [read(p.parent/'completion.json') for p in (primary, control)]
    for d in completions:
        if d['status'] != 'complete' or d['optimizer_steps'] != 2048 or d['training_episodes'] != 32768 or d['training_views'] != 65536:
            raise ValueError('The predeclared fixed-budget pilot is incomplete or changed')
        if d['natural_training_images'] != 0 or d['borrowed_pretrained_weights'] is not False:
            raise ValueError('HyperNet data/weight constraint violated')
        for name, expected in d['source_sha256'].items():
            if sha(ROOT/name) != expected: raise ValueError('Training source changed: '+name)
    for key in ('input_stream_sha256', 'source_sha256', 'family_episode_counts', 'unique_structural_programs'):
        if completions[0][key] != completions[1][key]:
            raise ValueError('Matched control differs in '+key)
    for completed, variant, geometry in zip(completions, ('primary', 'bag'), (True, False)):
        config = completed['config']
        if config['variant'] != variant or config['geometry'] is not geometry or config['smoke'] is not False:
            raise ValueError('Wrong model arm or a smoke run was supplied')
    comparable_configs = [{k: v for k, v in d['config'].items() if k not in ('variant', 'geometry')} for d in completions]
    if comparable_configs[0] != comparable_configs[1]:
        raise ValueError('Matched arms differ in another training setting')
    initial_states = [d['checkpoints'][0]['state_sha256'] for d in completions]
    if initial_states[0] != initial_states[1]: raise ValueError('Initialization differs')
    if sha(random) != completions[0]['checkpoints'][0]['sha256']:
        raise ValueError('Random comparator is not the trained primary initialization')
    for checkpoint, completed in zip((primary, control), completions):
        if sha(checkpoint) != completed['checkpoints'][-1]['sha256']:
            raise ValueError('Final checkpoint differs from completed training')
    checkpoints = {str(p): sha(p) for p in (primary, control, random)}
    behavior = {}
    for path in sorted((ROOT/'hypernet').glob('*.py')):
        behavior[str(path.resolve())] = sha(path)
    for name in ('prepare_coco_v17.py', 'evaluate_coco_v17.py', 'evaluate_comparison_v17.py', 'train_local.py', 'freeze_comparison.py'):
        path = ROOT/'scripts'/name; behavior[str(path.resolve())] = sha(path)
    methods = ['v17', 'v17_no_geometry', 'random', 'owlvit_base_patch32', 'raw_template']
    expected_checkpoints = {'v17': sha(primary), 'v17_no_geometry': sha(control), 'random': sha(random)}
    method_records = {}
    for method in methods:
        directory = ROOT/'experiments'/f'{method}_dev_v2'
        result = read(directory/'result.json'); calibration = read(directory/'calibration.json')
        if result['method'] != method or result['split'] != 'dev' or result['data_plan_sha256'] != sha(PLAN):
            raise ValueError('DEV result belongs to another experiment')
        if result['score_file_sha256'] != sha(directory/'arrays.npz'):
            raise ValueError('DEV score arrays changed')
        for filename in ('result.json', 'calibration.json', 'arrays.npz'):
            path = directory/filename; behavior[str(path.resolve())] = sha(path)
        provenance = result['provenance']
        if method in expected_checkpoints:
            if provenance['checkpoint_sha256'] != expected_checkpoints[method] or provenance['max_query_side'] != 384:
                raise ValueError('Wrong HyperNet checkpoint/preprocessing in DEV')
            if provenance['natural_training_images'] != 0 or provenance['borrowed_pretrained_weights'] is not False:
                raise ValueError('HyperNet inference provenance differs')
        elif method == 'owlvit_base_patch32':
            checkpoints.update(provenance['checkpoint_files'])
        for key, budget in [('primary_5pct', .05), ('secondary_10_per_10000', .001)]:
            if calibration[key]['nominal_dev_negative_fpr'] != budget:
                raise ValueError('Operating point changed')
        method_records[method] = {'dev_result_sha256': sha(directory/'result.json'),
            'dev_calibration_sha256': sha(directory/'calibration.json'),
            'cutoffs': {k: v['cutoff'] for k, v in calibration.items()}, 'provenance': provenance}
    for path, expected in checkpoints.items():
        if sha(path) != expected: raise ValueError('Checkpoint binding changed')
    protocol = {
        'format': 'hypernet.v17.fresh_test_admission.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_selection_complete': True, 'test_scores_computed_before_freeze': False,
        'data_plan_sha256': sha(PLAN), 'behavior_files': behavior, 'checkpoint_files': checkpoints,
        'method_records': method_records, 'matched_training': {
            'updates_per_arm': 2048, 'episodes_per_arm': 32768, 'image_views_per_arm': 65536,
            'input_stream_sha256': completions[0]['input_stream_sha256'], 'initial_state_sha256': initial_states[0],
            'source_sha256': completions[0]['source_sha256'],
            'completion_files': {str(p.parent/'completion.json'): sha(p.parent/'completion.json') for p in (primary, control)}},
        'primary': plan['protocol']['primary'], 'threshold_rule': plan['protocol']['cutoff'],
        'decision_rule': 'Higher macro localized recall without higher observed macro negative false-positive rate; show paired confidence intervals.',
        'limitations': plan['limitations'],
        'claim': 'An experiment freeze. Neither successful transfer nor algorithmic novelty is assumed.'
    }
    output = Path(args.output).resolve(); write_new(output, protocol)
    print(json.dumps({'protocol': str(output), 'sha256': sha(output), 'fresh_test_pixels_accessed': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary', 'control', 'random', 'output'): parser.add_argument('--'+name, required=True, type=Path)
    main(parser.parse_args())
