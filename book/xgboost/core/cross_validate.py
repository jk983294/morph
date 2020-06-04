import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import scipy

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test = xgb.DMatrix(X_test, label=y_test)

param = {'objective': 'binary:logistic', 'booster': 'gbtree', 'max_depth': 5, 'eta': 0.1}
train_round_num = 50
watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]
bst = xgb.cv(param, xgb_train, train_round_num, nfold=5, metrics=['auc'], seed=0,
             callbacks=[xgb.callback.print_evaluation(show_stdv=False), xgb.callback.early_stop(5)])
