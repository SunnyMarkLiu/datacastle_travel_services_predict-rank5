#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-22 下午5:30
"""
from __future__ import absolute_import, division, print_function

import os
import sys

import cPickle

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from conf.configure import Configure
from utils import data_utils


def discretize_features(train, test):
    """ 连续特征离散化 """
    test['orderType'] = np.array([0] * test.shape[0])
    conbined_data = pd.concat([train, test])

    # basic_user_action_features
    numerical_features = ['browse_product_ratio', 'browse_product2_ratio', 'browse_product3_ratio', 'fillin_form5_ratio', 'fillin_form6_ratio',
                          'fillin_form7_ratio', 'open_app_ratio', 'pay_money_ratio', 'submit_order_ratio', 'open_app_pay_money_ratio',
                          'browse_product_pay_money_ratio', 'browse_product2_pay_money_ratio', 'browse_product3_pay_money_ratio',
                          'fillin_form5_pay_money_ratio', 'fillin_form6_pay_money_ratio', 'fillin_form7_pay_money_ratio','submit_order_pay_money_ratio']
    for feature in numerical_features:
        conbined_data[feature] = pd.cut(conbined_data[feature].values, bins=int(len(set(conbined_data[feature])) * 0.6)).codes

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]
    del test['orderType']
    return train, test


def feature_interaction(train, test):
    """ 特征交叉等操作 """
    test['orderType'] = np.array([0] * test.shape[0])
    conbined_data = pd.concat([train, test])

    print('一些类别特征进行 one-hot')
    # basic_user_info， bad！
    # dummies = pd.get_dummies(conbined_data['province_code'], prefix='province_code')
    # conbined_data[dummies.columns] = dummies
    # del conbined_data['province_code']

    # # basic_user_action_features， bad！
    # dummies = pd.get_dummies(conbined_data['most_free_month'], prefix='most_free_month')
    # conbined_data[dummies.columns] = dummies
    # del conbined_data['most_free_month']

    # user_order_history_features，improve cv a little
    # dum_features = ['last_time_continent', 'last_time_country', 'last_time_city']
    # for f in dum_features:
    #     dummies = pd.get_dummies(conbined_data[f], prefix=f)
    #     conbined_data[dummies.columns] = dummies
    #     del conbined_data[f]

    print('特征组合')
    # conbined_data['has_good_order_x_country_rich'] = conbined_data['has_good_order'] * conbined_data['country_rich']

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
        print('pd merge', feature_name)
        train_feature, test_feature = data_utils.load_features(feature_name)
        train = pd.merge(train, train_feature,
                         on=features_merged_dict[feature_name]['on'],
                         how=features_merged_dict[feature_name]['how'])
        test = pd.merge(test, test_feature,
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

    print('特征组合')
    train, test = feature_interaction(train, test)

    print('连续特征离散化')
    train, test = discretize_features(train, test)

    return train, test


def load_571_all_feature_datasets():
    with open('all_571_features_train.pkl', "rb") as f:
        train = cPickle.load(f)
    with open('all_571_features_test.pkl', "rb") as f:
        test = cPickle.load(f)

    return train, test


def load_0_97210_datasets():
    with open('train_0.97210.pkl', "rb") as f:
        train = cPickle.load(f)
    with open('test_0.97210.pkl', "rb") as f:
        test = cPickle.load(f)

    return train, test


def load_datasets():
    print('load baseline features')
    train, test = load_0_97210_datasets()

    # 这些特征 和 性能更好的 history_order_type_sum_lg0 存在共线性
    # train.drop(['2016_2017_first_last_ordertype'], axis=1, inplace=True)
    # test.drop(['2016_2017_first_last_ordertype'], axis=1, inplace=True)

    # 加载特征， 并合并
    features_merged_dict = Configure.new_features
    for feature_name in features_merged_dict:
        print('merge', feature_name)
        train_feature, test_feature = data_utils.load_features(feature_name)
        train = pd.merge(train, train_feature,
                         on=features_merged_dict[feature_name]['on'],
                         how=features_merged_dict[feature_name]['how'])
        test = pd.merge(test, test_feature,
                        on=features_merged_dict[feature_name]['on'],
                        how=features_merged_dict[feature_name]['how'])

    # # 按照规则，讲测试集中的类别为1的测试数据添加到训练集中，线上爆炸！
    # sample_pos_test = test[test['history_order_type_sum_lg0'] == 1]
    # sample_pos_test['orderType'] = 1
    # train = pd.concat([train, sample_pos_test], axis=0)

    train.drop(['history_order_type_sum_lg0'], axis=1, inplace=True)
    test.drop(['history_order_type_sum_lg0'], axis=1, inplace=True)

    # train, test = remove_some_features(train, test)

    # with open('train_0.97329.pkl', "wb") as f:
    #     cPickle.dump(train, f, -1)
    # with open('test_0.97329.pkl', "wb") as f:
    #     cPickle.dump(test, f, -1)
    #
    return train, test

def remove_some_features(train, test):
    features_weights = pd.read_csv('0.97329_xgb_features_weights.csv')
    removed_features = features_weights[features_weights['weights'] == 0]['feature'].values

    train.drop(removed_features, axis=1, inplace=True)
    test.drop(removed_features, axis=1, inplace=True)

    return train, test
