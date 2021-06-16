from sklearn import linear_model
import numpy as np
from book.sklearn.core.dataset import random_split_diabetes_dataset

diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = random_split_diabetes_dataset()

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)
print(model.coef_)
print("mean square error", np.mean((model.predict(diabetes_X_test) - diabetes_y_test) ** 2))

# explained variance score:
# 1 is perfect prediction
# 0 means that there is no linear relationship between X and y
print("score", model.score(diabetes_X_test, diabetes_y_test))
