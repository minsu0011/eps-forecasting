import ast
from pathlib import Path
import pandas as pd
from research.eps_model_lab_v1.common import dataset, features
from research.eps_model_lab_v1.share_unit_diagnostic import NoShareFeatureAdapter


def test_ablation_removes_exactly_one_feature():
    frame = dataset().head(8)
    adapter = NoShareFeatureAdapter('ridge_raw')
    original = features(frame, scaled=False)
    result = adapter.prepare_data(frame)
    pd.testing.assert_frame_equal(result, original.drop(columns=['account_diluted_shares']))


def test_ablation_does_not_consume_changed_share_values():
    frame = dataset().head(8); changed = frame.copy()
    changed['account_diluted_shares'] = 1e30
    adapter = NoShareFeatureAdapter('hist_gradient_boosting')
    pd.testing.assert_frame_equal(adapter.prepare_data(frame), adapter.prepare_data(changed))


def test_all_lab_python_source_parses():
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
