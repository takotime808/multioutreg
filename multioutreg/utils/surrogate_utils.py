# Copyright (c) 2025 takotime808

import numpy as np

def predict_with_std(multioutput_model, X):
    preds = []
    stds = []
    for est in multioutput_model.estimators_:
        pred, std = est.predict(X, return_std=True)
        preds.append(pred)
        stds.append(std)
    return np.column_stack(preds), np.column_stack(stds)
