import lightgbm as lgb
import numpy as np

data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
label = np.random.randint(2, size=500)  # binary target
train_data = lgb.Dataset(data, label=label)

save_path = '/tmp/lgb.train.bin'
train_data.save_binary(save_path)
print('save to', save_path)

train_data = lgb.Dataset(save_path)
print('load from', save_path)
