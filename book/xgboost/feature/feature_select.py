import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.feature_selection import SelectFromModel


cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, n_estimators=50, booster='gbtree')
model.fit(X_train, y_train)

# test on test set
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)
print('auc score: %.2f' % auc)

# importance analysis
importances = model.feature_importances_
thresholds = []
for importance in importances:
    if importance not in thresholds:        # remove dupe
        thresholds.append(importance)

thresholds = sorted(thresholds)

for threshold in thresholds:
    selection = SelectFromModel(model, threshold=threshold, prefit=True)
    select_X_train = selection.transform(X_train)

    # now train new model after we select feature
    selection_model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, n_estimators=50,
                                        booster='gbtree')
    selection_model.fit(select_X_train, y_train)

    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    auc = roc_auc_score(y_test, y_pred)
    print('threshold: %f, feature number: %d auc: %.2f' % (threshold, select_X_train.shape[1], auc))
    print('selected feature index', selection.get_support(indices=True))
