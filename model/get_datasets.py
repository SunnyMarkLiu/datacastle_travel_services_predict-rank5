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

import numpy as np
import pandas as pd
from conf.configure import Configure
from utils import data_utils


def feature_selection(train, test):
    """ 特征选择 """

def feature_bincut_rank(train, test):
    """ 对特征进行 bin、rank、count 等操作 """
    # basic_user_info

    # basic_user_action_features
    test['orderType'] = np.array([0] * test.shape[0])
    conbined_data = pd.concat([train, test])

    # 对连续特征进行离散化
    numerical_features = ['browse_product_ratio', 'browse_product2_ratio',
                          'browse_product3_ratio', 'fillin_form5_ratio', 'fillin_form6_ratio',
                          'fillin_form7_ratio', 'open_app_ratio', 'pay_money_ratio',
                          'submit_order_ratio', 'action_counts']
    for feature in numerical_features:
        conbined_data[feature] = pd.cut(conbined_data[feature].values, bins=int(len(set(conbined_data[feature])) * 0.6)).codes

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    del test['orderType']
    return train, test


def load_train_test():
    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')

    # 加载特征， 并合并
    features_merged_dict = Configure.features
    for feature_name in Configure.features:
        print('merge', feature_name)
        train_feature, test_feature = data_utils.load_features(feature_name)
        train = train.merge(train_feature,
                            on=features_merged_dict[feature_name]['on'],
                            how=features_merged_dict[feature_name]['how'])
        test = test.merge(test_feature,
                          on=features_merged_dict[feature_name]['on'],
                          how=features_merged_dict[feature_name]['how'])

    # # 过采样处理样本不均衡
    # pos_train = train[train['orderType'] == 1]
    # neg_train = train[train['orderType'] == 0]
    # print('train, ordertype1: ', pos_train.shape[0], ', ordertype0: ', neg_train.shape[0], ', 1:0 = ', 1.0 * pos_train.shape[0] / neg_train.shape[0])
    #
    # sample_pos_size = int(pos_train.shape[0] * 0.05)
    # sample_pos_train = pos_train.sample(sample_pos_size, random_state=42)
    # train = pd.concat([neg_train, pos_train, sample_pos_train])
    # pos_train = train[train['orderType'] == 1]
    # print('train, ordertype1: ', pos_train.shape[0], ', ordertype0: ', neg_train.shape[0], ', 1:0 = ', 1.0 * pos_train.shape[0] / neg_train.shape[0])

    train.drop(['gender', 'province', 'age', 'has_history_flag'], axis=1, inplace=True)
    test.drop(['gender', 'province', 'age', 'has_history_flag'], axis=1, inplace=True)

    # # 去掉 importance 很低的特征
    # droped_features = ['user_rating_std']
    # train.drop(droped_features, axis=1, inplace=True)
    # test.drop(droped_features, axis=1, inplace=True)

    print('连续特征离散化')
    train, test = feature_bincut_rank(train, test)


    return train, test
