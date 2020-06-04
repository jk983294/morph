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
train_round_num = 10
watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]
bst = xgb.train(param, xgb_train, train_round_num, watch_list)

# predict, output_margin=True means original value without sigmoid converted value
pred_train = bst.predict(xgb_train, output_margin=True)
pred_test = bst.predict(xgb_test, output_margin=True)

# set init predict
xgb_train.set_base_margin(pred_train)
xgb_test.set_base_margin(pred_test)

print("retraining")
bst = xgb.train(param, xgb_train, train_round_num, watch_list)
