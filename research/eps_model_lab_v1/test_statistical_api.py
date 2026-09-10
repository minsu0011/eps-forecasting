import numpy as np
import pandas as pd
from research.eps_model_lab_v1.statistical_models import StatisticalAdapter


def test_statistical_common_predict_supports_native_ttm_approximation():
    adapter=StatisticalAdapter('sf_Naive').fit()
    adapter.forecast=lambda frame:(np.array([[1.,-2.,3.,4.]]),[],[])
    assert adapter.predict(pd.DataFrame(index=[0]),'ttm')[0]==6.
    assert adapter.predict(pd.DataFrame(index=[0]),'h2')[0]==-2.
