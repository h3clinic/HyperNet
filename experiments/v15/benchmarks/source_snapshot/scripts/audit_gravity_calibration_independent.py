"""Independent score-log arithmetic audit before GravitySpy TEST access."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(__file__).resolve().parents[1]
RECIPES = ['bag', 'translation_006', 'translation_012']
POINTS = {'budget_10_per10000': 10., 'zero_calibration_false_alarms': 0.}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, context):
    need(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'experiments/gravity_spy_protocol_freeze.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier audit')
    protocol = read(args.protocol)
    for group in ('behavior_files', 'checkpoint_files'):
        for name, digest in protocol[group].items():
            need(sha(ROOT / name) == digest, 'Protocol-bound file changed: ' + name)
    need(protocol['primary_checkpoint'] == 'final', 'Primary checkpoint differs')
    targets = protocol['target_labels']
    need(len(targets) == 19 and len(set(targets)) == 19, 'Target roster differs')
    models = {}
    event_roster = None
    selection_by_model = {}
    verified_cutoffs = {}
    for model_name, model in protocol['models'].items():
        result = read(ROOT / model['result_path'])
        need(result['completed'] and result['mode'] == 'calibration', 'Non-calibration result supplied')
        score_path = ROOT / model['score_path']
        need(sha(score_path) == model['scores_sha256'] == result['scores_sha256'], 'Score digest differs')
        rows = [json.loads(line) for line in score_path.read_text().splitlines() if line]
        need(len(rows) == protocol['additional_calibration_events'] == result['events'], 'CALIB count differs')
        roster = [(r['event_id'], r['group_id'], r['label'], r['pixel_sha256']) for r in rows]
        need(len(set(r[0] for r in roster)) == len(rows), 'Duplicate event ID')
        if event_roster is None:
            event_roster = roster
        else:
            need(roster == event_roster, 'CALIB pixels/event roster differs across checkpoints')
        labels = np.asarray([r['label'] for r in rows])
        method_results = {}
        for method, reported in result['methods'].items():
            scores = np.asarray([[row['scores'][method][target]['score'] for target in targets] for row in rows])
            per_target = {}
            recalls = []
            aucs = []
            for index, target in enumerate(targets):
                y = labels == target
                need(y.any() and (~y).any(), 'Both CALIB outcomes required')
                vector = scores[:, index]
                negatives = vector[~y]
                auc = float(roc_auc_score(y, vector))
                ap = float(average_precision_score(y, vector))
                per_target[target] = {}
                for point, budget in POINTS.items():
                    permitted = math.floor(len(negatives) * budget / 10000)
                    threshold = math.nextafter(sorted(negatives, reverse=True)[permitted], math.inf)
                    alarm = vector >= threshold
                    tp = int((alarm & y).sum())
                    fp = int((alarm & ~y).sum())
                    saved = reported['per_target'][target][point]
                    need(threshold == saved['cutoff']['threshold'], 'Threshold differs')
                    need(permitted == saved['cutoff']['allowed_false_alarms'], 'Allowed CALIB errors differ')
                    need(fp == saved['cutoff']['observed_false_alarms'] <= permitted, 'CALIB cutoff budget differs')
                    need(tp == saved['metrics']['true_positive'] and fp == saved['metrics']['false_positive'], 'TP/FP differs')
                    same(auc, saved['metrics']['auroc'], 'Target AUROC differs')
                    same(ap, saved['metrics']['average_precision'], 'Target AP differs')
                    per_target[target][point] = threshold
                    if point == 'budget_10_per10000':
                        recalls.append(tp / int(y.sum()))
                        aucs.append(auc)
            macro_recall = float(np.mean(recalls))
            macro_auc = float(np.mean(aucs))
            same(macro_recall, reported['macro']['recall'], 'Macro recall differs')
            same(macro_auc, reported['macro']['auroc'], 'Macro AUROC differs')
            unions = {}
            for point in POINTS:
                cutoffs = np.asarray([per_target[target][point] for target in targets])
                accepted = scores >= cutoffs[None]
                true_target = labels[:, None] == np.asarray(targets)[None]
                wrong = (accepted & ~true_target).any(axis=1)
                recovered = (accepted & true_target).any(axis=1)
                present = true_target.any(axis=1)
                group_any = {}
                for row, flag in zip(rows, wrong):
                    group_any[row['group_id']] = group_any.get(row['group_id'], False) or bool(flag)
                calculated = {'screened_events': len(rows), 'wrong_target_alarm_events': int(wrong.sum()),
                              'target_recovered_events': int(recovered.sum()),
                              'events_containing_an_enrolled_target': int(present.sum()),
                              'background_events': int((~present).sum()),
                              'background_events_with_any_alarm': int(accepted[~present].any(axis=1).sum()),
                              'screened_time_groups': len(group_any),
                              'time_groups_with_any_wrong_alarm': sum(group_any.values())}
                saved = reported['union_events'][point]
                for key, value in calculated.items():
                    need(value == saved[key], 'Event/group union differs: ' + key)
                same(10000 * int(wrong.sum()) / len(rows), saved['wrong_target_alarm_events_per10000_screened_events'], 'Union event rate differs')
                unions[point] = calculated
            method_results[method] = {'macro_recall': macro_recall, 'macro_auroc': macro_auc,
                                      'targets': len(per_target), 'unions': unions}
            verified_cutoffs[(model_name, method)] = per_target
        candidates = ['learned:' + recipe for recipe in RECIPES]
        selected = max(candidates, key=lambda key: (method_results[key]['macro_recall'],
                       method_results[key]['macro_auroc'], -candidates.index(key)))
        need(selected == model['selected_method'] == result['selection']['selected_method'], 'Global recipe selection differs')
        need(model['thresholds'] == {target: verified_cutoffs[(model_name, selected)][target]['budget_10_per10000'] for target in targets},
             'Model protocol thresholds differ')
        selection_by_model[model_name] = selected
        models[model_name] = {'events': len(rows), 'selected_method': selected, 'methods': method_results}
    final = models['final']['methods']
    analytic_candidates = ['analytic:' + recipe for recipe in RECIPES]
    analytic = max(analytic_candidates, key=lambda key: (final[key]['macro_recall'], final[key]['macro_auroc'], -analytic_candidates.index(key)))
    need(analytic == protocol['reported_arms']['analytic']['method'], 'Independent analytic recipe differs')
    for arm, spec in protocol['reported_arms'].items():
        need(spec['thresholds'] == verified_cutoffs[(spec['model'], spec['method'])], 'Reported-arm thresholds differ: ' + arm)
    need(protocol['primary_method'] == selection_by_model['final'], 'Primary method differs')
    need(protocol['thresholds'] == protocol['models']['final']['thresholds'], 'Top-level thresholds differ')
    report = {'format': 'hypernet.gravity_calibration_independent_audit.v15', 'valid': True,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(args.protocol),
              'audit_source_sha256': sha(__file__), 'models': models, 'analytic_selected_method': analytic,
              'reported_arms': len(protocol['reported_arms']), 'targets_per_arm': len(targets),
              'all_calibration_score_rosters_identical': True, 'test_pixels_read': 0,
              'model_inference_calls': 0, 'new_threshold_or_recipe_selection': False,
              'scope': 'Independent saved-score calculation of CALIB thresholds, TP/FP, sklearn AUROC/AP, '
              'global recipe selection and event/group wrong-target union counts. No image reading or model calls.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'valid': True, 'output': str(args.output), 'sha256': sha(args.output),
                      'selected_methods': selection_by_model, 'analytic': analytic}))


if __name__ == '__main__':
    main()
