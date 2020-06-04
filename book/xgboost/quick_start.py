import xgboost as xgb
import matplotlib.pyplot as plt

# read in data
dtrain = xgb.DMatrix('/home/kun/github/barn/train/agaricus.txt.train')
dtest = xgb.DMatrix('/home/kun/github/barn/train/agaricus.txt.test')

# specify parameters via map
param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic', 'gamma': 1.0}
train_round_num = 2
watch_list = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(param, dtrain, train_round_num, watch_list)

# make prediction
preds = bst.predict(dtest)

print(preds)

# visualization
feature_map_path = "/home/kun/github/barn/train/featmap.txt"
xgb.plot_tree(bst, fmap=feature_map_path)
plt.show()
