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

    # 去掉 importance 很低的特征
    droped_features = ['actiontimespanlast_7_8', 'fillin_form7_std_delta', 'fillin_form7_mean_delta',
                       'year_action_count', 'actiontypeproplast20_3', 'pay_money_max_delta',
                       'pay_money_std_delta', 'last_time_order_now_has_paied_money', '2016_order_month_count',
                       'fillin_form7_max_delta']
    train.drop(droped_features, axis=1, inplace=True)
    test.drop(droped_features, axis=1, inplace=True)

    return train, test
