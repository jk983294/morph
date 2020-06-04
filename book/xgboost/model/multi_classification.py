import xgboost as xgb
import pandas as pd
import numpy as np

# converters make label column to [0, num_class - 1], because we need argmax to get class num which start from 0
data = pd.read_csv('/home/kun/github/barn/train/seeds_dataset.txt', header=None, sep='\s+',
                   converters={7: lambda x: int(x) - 1})
data.rename(columns={7: 'label'}, inplace=True)
print(data.head())

# generate test/train set
mask = np.random.rand(len(data)) < 0.8
train = data[mask]
test = data[~mask]
xgb_train = xgb.DMatrix(train.iloc[:, :6], label=train.label)
xgb_test = xgb.DMatrix(test.iloc[:, :6], label=test.label)

# specify parameters via map
param = {'objective': 'multi:softmax', 'max_depth': 5, 'eta': 0.1, 'num_class': 3}
train_round_num = 50
watch_list = [(xgb_train, 'train'), (xgb_test, 'test')]
bst = xgb.train(param, xgb_train, train_round_num, watch_list)

# make prediction
preds = bst.predict(xgb_test)
print('predict label', preds)

error_rate = np.sum(preds != test.label) / test.shape[0]
print('error_rate(softmax): {}'.format(error_rate))

# output probability
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xgb_train, train_round_num, watch_list)
pred_prob = bst.predict(xgb_test)
print('predict probability', pred_prob)
pred_label = np.argmax(pred_prob, axis=1)
print('predict label', pred_label)
error_rate = np.sum(pred_label != test.label) / test.shape[0]
print('error_rate(softprob): {}'.format(error_rate))
