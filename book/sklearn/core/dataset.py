import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def random_split_iris_dataset_fancy(test_size=0.2):
    """ split iris data in train and test data A random permutation"""
    np.random.seed(0)
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    # indices = np.random.permutation(len(iris_X))
    # test_set_size = int(len(iris_X) * test_ratio)
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=test_size, random_state=0)
    return X_train, y_train, X_test, y_test


def random_split_iris_dataset(test_ratio=0.1):
    """ split iris data in train and test data A random permutation"""
    np.random.seed(0)
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    indices = np.random.permutation(len(iris_X))
    test_set_size = int(len(iris_X) * test_ratio)
    X_train = iris_X[indices[:-test_set_size]]
    y_train = iris_y[indices[:-test_set_size]]
    X_test = iris_X[indices[-test_set_size:]]
    y_test = iris_y[indices[-test_set_size:]]
    return X_train, y_train, X_test, y_test


def random_split_diabetes_dataset():
    np.random.seed(0)
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    X_train = diabetes_X[:-20]
    X_test = diabetes_X[-20:]
    y_train = diabetes_y[:-20]
    y_test = diabetes_y[-20:]
    return X_train, y_train, X_test, y_test


def iris_dataset():
    iris = datasets.load_iris()
    print(iris.data.shape)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # print(iris)
    print(df.head())


def digits_dataset():
    # predict, given an image, which digit it represents
    digits = datasets.load_digits()
    print("raw data", digits.images[0])  # 8 * 8 pixel matrix
    plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
    plt.show()
    # transform each 8x8 image into a feature vector of length 64 using below:
    # data = digits.images.reshape((digits.images.shape[0], -1))
    print(digits.data.size)  # used to classify the digits samples
    print(len(digits.target))  # he ground truth for the digit dataset


def load_housing_data(housing_path="~/github/barn"):
    csv_path = os.path.join(housing_path, "train/housing.csv")
    return pd.read_csv(csv_path)


if __name__ == '__main__':
    # iris_dataset()
    # digits_dataset()
    # X_train, y_train, X_test, y_test = random_split_iris_dataset_fancy()
    # print(X_train.shape)

    housing = load_housing_data()
    print(housing.head())
    print(housing.info())
