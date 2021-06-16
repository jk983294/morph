from sklearn import linear_model
import numpy as np
from book.sklearn.core.dataset import random_split_diabetes_dataset

diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = random_split_diabetes_dataset()

# regularization
# Lasso (least absolute shrinkage and selection operator), can set some coefficients to zero
model = linear_model.Lasso()
alphas = np.logspace(-4, -1, 6)
scores = [model.set_params(alpha=alpha)
              .fit(diabetes_X_train, diabetes_y_train)
              .score(diabetes_X_test, diabetes_y_test) for alpha in alphas]

best_alpha = alphas[scores.index(max(scores))]
model.alpha = best_alpha
model.fit(diabetes_X_train, diabetes_y_train)
print(model.coef_)
