import xgboost as xgb
import pandas as pd
import numpy as np

data = pd.read_excel('/home/kun/github/barn/train/Concrete_Data.xls')
data.rename(columns={"Concrete compressive strength(MPa, megapascals) ": 'label'}, inplace=True)
print(data.head())

# generate test/train set
mask = np.random.rand(len(data)) < 0.8
train = data[mask]
test = data[~mask]
xgb_train = xgb.DMatrix(train.iloc[:, :7], label=train.label)
xgb_test = xgb.DMatrix(test.iloc[:, :7], label=test.label)

# specify parameters
param = {'objective': 'reg:linear', 'booster': 'gbtree', 'max_depth': 5, 'eta': 0.1, 'min_child_weight': 1}
train_round_num = 50
watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]
bst = xgb.train(param, xgb_train, train_round_num, watch_list)

# make prediction
preds = bst.predict(xgb_test)
print('predict value', preds)
