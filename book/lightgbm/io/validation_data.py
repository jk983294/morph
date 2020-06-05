import lightgbm as lgb
import numpy as np

data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
label = np.random.randint(2, size=500)  # binary target
train_data = lgb.Dataset(data, label=label)

# the validation data should be aligned with training data
validation_data = train_data.create_valid('validation.svm')
