"""Predeclared exact-constant source-plot validity rule; no score thresholds."""
import numpy as np

RULE_ID = 'exact_constant_rgb_plot_v1'


def is_constant_plot(rgb):
    array=np.asarray(rgb)
    if array.dtype!=np.uint8 or array.ndim!=3 or array.shape[2]!=3 or min(array.shape[:2])<1:
        raise ValueError('Expected nonempty uint8 RGB image')
    return bool(np.all(array==array[0,0]))


def classify_plot(rgb):
    invalid=is_constant_plot(rgb)
    return {'rule':RULE_ID,'eligible':not invalid,'reason':'exact_constant_rgb_plot'if invalid else None,
        'constant_rgb':np.asarray(rgb)[0,0].tolist()if invalid else None}
