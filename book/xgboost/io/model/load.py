import xgboost as xgb

dtest = xgb.DMatrix('/home/kun/github/barn/train/agaricus.txt.test')

model_path = "/tmp/agaricus.model"
bst = xgb.Booster()
bst.load_model(model_path)

# make prediction
preds = bst.predict(dtest)

print(preds)
