# Copyright (c) 2025 takotime808

import pandas as pd

from multioutreg.utils.metric_utils import flatten_metrics_df


def test_flatten_metrics_df():
    data = {
        'accuracy': [{'mae': 1.0, 'rmse': 2.0}],
        'sharpness': [{'var': 0.1}],
        'output': [0]
    }
    df = pd.DataFrame(data)
    flat = flatten_metrics_df(df)
    assert set(flat.columns) == {'mae', 'rmse', 'var', 'output'}
    assert flat.loc[0, 'mae'] == 1.0
    assert flat.loc[0, 'var'] == 0.1