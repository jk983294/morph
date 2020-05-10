from sklearn import linear_model
import numpy as np
from book.sklearn.core.dataset import random_split_diabetes_dataset


diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = random_split_diabetes_dataset()

# regularization, Ridge regression
# Ridge regression will decrease feature contribution, but not set them to zero
model = linear_model.Ridge(alpha=.1)
alphas = np.logspace(-4, -1, 6)
print([model.set_params(alpha=alpha)
      .fit(diabetes_X_train, diabetes_y_train)
      .score(diabetes_X_test, diabetes_y_test) for alpha in alphas])
