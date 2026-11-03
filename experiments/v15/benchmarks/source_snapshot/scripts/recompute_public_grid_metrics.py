"""Recompute public compact Grid image-score statistics; reads no images."""
from pathlib import Path
import json, math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT / 'benchmarks/grid/scores.npz'
    historical = ROOT / 'benchmarks/grid/provenance/results/grid_final_test/grid_results.json'
    reported = json.loads(historical.read_text())
    out = {}
    with np.load(archive, allow_pickle=False) as saved:
        labels = saved['is_anomaly']
        assert saved['test_scores'].shape == (78, 3)
        assert saved['calibration_scores'].shape == (64, 3)
        for column, alias in enumerate(saved['method_names'].tolist()):
            scores = saved['test_scores'][:, column]
            threshold = float(np.nextafter(saved['calibration_scores'][:, column].max(), math.inf))
            assert threshold == saved['thresholds'][column]
            alarm = scores >= threshold
            metrics = {'positive_events': int(labels.sum()), 'negative_events': int((~labels).sum()),
                       'true_positive': int((alarm & labels).sum()), 'false_positive': int((alarm & ~labels).sum()),
                       'recall': float(np.mean(alarm[labels])),
                       'false_alarms_per_10000_negatives': 10000 * float(np.mean(alarm[~labels])),
                       'auroc': float(roc_auc_score(labels, scores)),
                       'average_precision': float(average_precision_score(labels, scores)), 'threshold': threshold}
            for name, value in metrics.items():
                assert math.isclose(value, reported['methods'][alias]['image'][name], rel_tol=0, abs_tol=1e-12), (alias, name)
            out[alias] = metrics
    print(json.dumps({'matches_published_image_statistics': True, 'real_images_read': 0, 'methods': out}, indent=2))


if __name__ == '__main__':
    main()
