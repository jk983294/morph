import lightgbm as lgb
import numpy as np

data = np.random.randint(0, 10, size=(500, 3))
label = np.random.randint(2, size=500)  # binary target

# specific feature names and categorical features
# LightGBM can use categorical features as input directly.
# It doesn’t need to convert to one-hot coding, and is much faster than one-hot coding (about 8x speed-up).
# Note: You should convert your categorical features to int type before you construct Dataset.
train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])
