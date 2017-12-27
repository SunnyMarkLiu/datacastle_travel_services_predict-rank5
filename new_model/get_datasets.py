#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-22 下午5:30
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from conf.configure import Configure
from utils import data_utils


def load_train_test():
    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')

    # 加载特征， 并合并
    features_merged_dict = Configure.features
    for feature_name in Configure.features:
        train_feature, test_feature = data_utils.load_features(feature_name)
        train = train.merge(train_feature,
                            on=features_merged_dict[feature_name]['on'],
                            how=features_merged_dict[feature_name]['how'])
        test = test.merge(test_feature,
                          on=features_merged_dict[feature_name]['on'],
                          how=features_merged_dict[feature_name]['how'])

    return train, test


def load_new_datsets():
    # 待预测订单的数据 （原始训练集和测试集）
    train, test = data_utils.load_new_train_test()

    # 加载特征， 并合并
    features_merged_dict = Configure.new_features
    for feature_name in features_merged_dict:
        train_feature, test_feature = data_utils.load_new_features(feature_name)
        train = train.merge(train_feature,
                            on=features_merged_dict[feature_name]['on'],
                            how=features_merged_dict[feature_name]['how'])
        test = test.merge(test_feature,
                          on=features_merged_dict[feature_name]['on'],
                          how=features_merged_dict[feature_name]['how'])

    return train, test
