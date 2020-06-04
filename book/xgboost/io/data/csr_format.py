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

row = []
col = []
dat = []
i = 0
for arr in X_train:
    for k, v in enumerate(arr):
        row.append(i)
        col.append(int(k))
        dat.append(float(v))
    i += 1

csr = scipy.sparse.csr_matrix((dat, (row, col)))
xgb_train = xgb.DMatrix(csr, label=y_train)
xgb_test = xgb.DMatrix(X_test, label=y_test)

param = {'objective': 'binary:logistic', 'booster': 'gbtree', 'max_depth': 5, 'eta': 0.1, 'min_child_weight': 1}
train_round_num = 50
watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]
bst = xgb.train(param, xgb_train, train_round_num, watch_list)

# convert to numpy array format
nparray = csr.todense()
xgb_train = xgb.DMatrix(nparray, label=y_train)
watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]
bst = xgb.train(param, xgb_train, train_round_num, watch_list)
