import xgboost as xgb

# read in data
dtrain = xgb.DMatrix('/home/kun/github/barn/train/agaricus.txt.train')
dtest = xgb.DMatrix('/home/kun/github/barn/train/agaricus.txt.test')

# specify parameters via map
param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic', 'gamma': 1.0, 'booster': 'gbtree'}
train_round_num = 2
watch_list = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(param, dtrain, train_round_num, watch_list)

model_path = "/tmp/agaricus.model.txt"
feature_map_path = "/home/kun/github/barn/train/featmap.txt"
bst.dump_model(model_path, feature_map_path)
print("save model to", model_path)
