import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error

print('boston house price predict')
boston = datasets.load_boston()
X = boston.data
y = boston.target
kf = KFold(n_splits=3, shuffle=True)
i = 0
for train_idx, test_idx in kf.split(X):
    model = xgb.XGBRegressor().fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    labels = y[test_idx]
    print('kfold-%d MSE: %f' % (i, mean_squared_error(labels, preds)))
    i += 1
