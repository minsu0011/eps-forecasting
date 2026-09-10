import pandas as pd
import pytest
from research.eps_model_lab_v1.common import dataset, require_exact_frozen_frame


def test_exact_copy_allowed():
    frame = dataset()
    pd.testing.assert_frame_equal(require_exact_frozen_frame(frame.copy()), frame)


@pytest.mark.parametrize('column', ['y_h1', 'scale', 'eps_lag_0', 'asof_date'])
def test_same_ids_do_not_authorize_changed_values(column):
    frame = dataset().copy()
    index = frame.index[0]
    frame.loc[index, column] = pd.Timestamp('2099-01-01', tz='UTC') if column == 'asof_date' else 1e12
    with pytest.raises(RuntimeError, match='exact frozen frame'):
        require_exact_frozen_frame(frame)


def test_reordered_rows_rejected():
    with pytest.raises(RuntimeError): require_exact_frozen_frame(dataset().iloc[::-1])


def test_reordered_columns_rejected():
    frame = dataset()
    with pytest.raises(RuntimeError): require_exact_frozen_frame(frame[frame.columns[::-1]])
