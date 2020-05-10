from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn import datasets, svm
import numpy as np
import pandas as pd


def demo1():
    X = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
    k_fold = KFold(n_splits=5)
    for train_indices, test_indices in k_fold.split(X):
        print('train: %s | test: %s' % (train_indices, test_indices))


def cross_val_score_demo():
    X, y = datasets.load_digits(return_X_y=True)
    svc = svm.SVC(kernel='linear')
    k_fold = KFold(n_splits=5)
    print(cross_val_score(svc, X, y, cv=k_fold, n_jobs=-1))


if __name__ == '__main__':
    cross_val_score_demo()
