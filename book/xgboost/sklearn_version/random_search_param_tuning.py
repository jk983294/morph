import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

model = xgb.XGBClassifier()

param_dist = {'max_depth': sp_randint(2, 10), 'n_estimators': sp_randint(20, 70), 'learning_rate': [0.05, 0.1]}
search_n_iter = 10
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=search_n_iter)
random_search.fit(X, y)

# output progress
means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, random_search.cv_results_['params']):
    print('mean=%0.3f std=%0.3f for %r' % (mean, std, params))

print('best_score_', random_search.best_score_)
print('best_params_', random_search.best_params_)
