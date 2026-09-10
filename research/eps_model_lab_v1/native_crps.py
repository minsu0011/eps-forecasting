"""Exact CRPS only for known native Normal/Laplace predictive families.

Their two parameters are identified by the saved native median and P90. This
does NOT approximate a general distribution from three quantiles. No fitting,
calibration, member selection or prediction mutation is performed here.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import atomic_csv

FAMILIES = {'ngboost_normal': 'normal', 'ngboost_laplace': 'laplace',
            'bayesian_ridge_distribution': 'normal',
            'gaussian_process_matern_distribution': 'normal'}


def crps(y, location, scale, family):
    y, location, scale = np.broadcast_arrays(y, location, scale)
    if not all(np.isfinite(v).all() for v in [y, location, scale]) or (scale <= 0).any():
        raise ValueError('Finite location/truth and strictly positive native scale required')
    z = (y-location)/scale
    if family == 'normal':
        value = scale*(z*(2*ndtr(z)-1) + 2*np.exp(-z*z/2)/np.sqrt(2*np.pi) - 1/np.sqrt(np.pi))
    elif family == 'laplace':
        value = scale*(np.abs(z)+np.exp(-np.abs(z))-.75)
    else:
        raise ValueError('No analytic CRPS contract for this family')
    if (value < 0).any():
        raise ArithmeticError('Negative CRPS')
    return value


def run():
    rows = []; bindings = {}
    for name, family in FAMILIES.items():
        path = RUN/'predictions'/f'{name}.parquet'
        receipt = json.loads((RUN/'model_receipts'/f'{name}.json').read_text(encoding='utf-8'))
        if receipt['status'] != 'FULL_RESEARCH_SCORED' or sha(path) != receipt['prediction_sha256']:
            raise RuntimeError('Unqualified or changed native distribution '+name)
        replay = json.loads((RUN/'probabilistic_replay'/f'{name}.json').read_text(encoding='utf-8'))
        if replay['completed'] != 40 or not replay['all_pass']:
            raise RuntimeError('Actual native distribution replay incomplete '+name)
        g = pd.read_parquet(path)
        location = g.p50_eps.to_numpy()
        scale = (g.p90_eps.to_numpy()-location)/(norm.ppf(.9) if family == 'normal' else np.log(5))
        np.testing.assert_allclose(g.predicted_eps, location, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(location-g.p10_eps, g.p90_eps-location, rtol=1e-10, atol=1e-10)
        valid = np.isfinite(g.actual_eps) & g.prediction_valid
        g = g.assign(native_crps=np.nan)
        g.loc[valid, 'native_crps'] = crps(g.loc[valid, 'actual_eps'].to_numpy(), location[valid], scale[valid], family)
        for target, group in g.groupby('target_key'):
            masks = {'SELECTION_OOF_AVAILABLE_BEFORE_2022': (group.split == 'VALIDATION_OOF') & (group.label_asof < pd.Timestamp('2022-01-01', tz='UTC')),
                     'RESEARCH_TEST_DESCRIPTIVE_ONLY': group.split == 'RESEARCH_TEST'}
            for surface, mask in masks.items():
                sub = group[mask & group.native_crps.notna()]
                if not len(sub):
                    continue
                rows.append({'eps_model_id': name, 'target_key': target, 'surface': surface,
                             'native_family': family, 'rows': len(sub), 'CRPS': float(sub.native_crps.mean()),
                             'median_CRPS': float(sub.native_crps.median()),
                             'price_scaled_CRPS': float((sub.native_crps/sub.origin_price.where(sub.origin_price > 0)).mean()),
                             'formal_certified': False, 'selection_used': False})
        bindings[name] = {'prediction_sha256': sha(path), 'actual_native_replay_sha256': sha(RUN/'probabilistic_replay'/f'{name}.json')}
    atomic_csv(pd.DataFrame(rows), RUN/'EPS_NATIVE_ANALYTIC_CRPS_V1.csv')
    save_json(RUN/'NATIVE_CRPS_AUDIT.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'model_bindings': bindings, 'source_sha256': sha(__file__), 'rows': len(rows),
              'scope': 'Exact known Normal/Laplace families only; analytic parameters recovered from their native quantiles after actual model replay.',
              'test_used_to_select_or_calibrate': False, 'formal_certified': False})
    print('NATIVE_CRPS_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    run()
