from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
train_data = lgb.Dataset(X_train, label=y_train)

# specify parameters via map
param = {'objective': 'binary', 'num_leaves': 31, 'eta': 1, 'gamma': 1.0, 'metric': ['auc', 'binary_logloss']}
num_round = 10
bst = lgb.train(param, train_data, num_round)

# make prediction
preds = bst.predict(X_test)
print(preds)
print('The rmse of prediction is:', mean_squared_error(y_test, preds) ** 0.5)

