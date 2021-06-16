import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from book.sklearn.core.dataset import random_split_iris_dataset

iris_X_train, iris_y_train, iris_X_test, iris_y_test = random_split_iris_dataset()

# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
print("predict:", knn.predict(iris_X_test))
print(" target:", iris_y_test)
