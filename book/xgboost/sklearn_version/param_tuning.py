import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

model = xgb.XGBClassifier()
clf = GridSearchCV(model, {'max_depth': [4, 5, 6], 'n_estimators': [20, 50, 70], 'learning_rate': [0.05, 0.1, 0.2]})
clf.fit(X, y)
print(clf.best_score_)
print(clf.best_params_)

