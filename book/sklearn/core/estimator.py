# All estimators expose a fit method that takes a dataset (usually a 2-d array)
estimator.fit(data)

# supervised estimators in scikit implement:
estimator.fit(X, y)
estimator.predict(X)

# it is instantiated or by modifying the corresponding attribute
estimator = Estimator(param1=1, param2=2)
estimator.param1 = 1

# All estimated parameters are attributes of the estimator object ending by an underscore
print(estimator.estimated_param_)
