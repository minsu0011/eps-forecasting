import numpy as np
import pandas as pd
from research.eps_model_lab_v1.independent_metric_audit import independent


def test_dtype_profile_explicit_and_point_metrics_unchanged():
    f = pd.DataFrame({'actual_eps': [1., 2., 3.], 'predicted_eps': [1.1, 2.2, 3.3],
        'prediction_valid': [True]*3, 'origin_price': [10., 20., 30.],
        'p10_eps': np.array([.1, .2, .3], dtype=np.float32),
        'p50_eps': np.array([1.1, 2.2, 3.3], dtype=np.float32),
        'p90_eps': np.array([2.5, 3.7, 4.9], dtype=np.float32)})
    native = independent(f, True); promoted = independent(f, False)
    assert native['interval80_width'] == float((f.p90_eps.to_numpy()-f.p10_eps.to_numpy()).mean())
    assert promoted['interval80_width'] == float((f.p90_eps.to_numpy().astype(float)-f.p10_eps.to_numpy().astype(float)).mean())
    assert native['MAE'] == promoted['MAE']
    assert native['interval80_coverage'] == promoted['interval80_coverage']


def test_invalid_prediction_not_rewarded():
    f = pd.DataFrame({'actual_eps': [1., 2.], 'predicted_eps': [1., 999.],
        'prediction_valid': [True, False], 'origin_price': [10., 20.],
        'p10_eps': [np.nan]*2, 'p50_eps': [np.nan]*2, 'p90_eps': [np.nan]*2})
    result = independent(f)
    assert result['truth_rows'] == 2
    assert result['predicted_rows'] == 1
    assert result['coverage'] == .5
    assert result['MAE'] == 0.
