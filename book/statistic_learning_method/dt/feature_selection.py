import numpy as np
import pandas as pd
from math import log


def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    column_names_ = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    return datasets, column_names_


datasets, column_names = create_data()
train_data = pd.DataFrame(datasets, columns=column_names)
print(train_data.head())


def calc_entropy(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    """use label_count[label] / data_length to estimate probability of label (p_i)"""
    ent = -sum([(p / data_length) * log(p / data_length, 2) for p in label_count.values()])
    return ent


def condition_entropy(datasets, feature_idx=0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][feature_idx]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    return sum([(len(p) / data_length) * calc_entropy(p) for p in feature_sets.values()])


def info_gain(entropy_, condition_entropy_):
    return entropy_ - condition_entropy_


def get_best_feature(datasets):
    feature_count = len(datasets[0]) - 1
    ent = calc_entropy(datasets)
    best_feature_ = []
    for f in range(feature_count):
        f_info_gain = info_gain(ent, condition_entropy(datasets, feature_idx=f))
        best_feature_.append((f, f_info_gain))
        print('feature({}) - info_gain - {:.3f}'.format(column_names[f], f_info_gain))
    return max(best_feature_, key=lambda x: x[-1])


best_feature = get_best_feature(np.array(datasets))
print('特征(%s)的信息增益最大，选择为根节点特征' % column_names[best_feature[0]])
