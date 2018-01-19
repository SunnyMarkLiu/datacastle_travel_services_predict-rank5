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
from utils import data_utils, xgb_feature_selector
from sklearn.model_selection import train_test_split


def feature_selection(train, selected_size):
    """ 特征选择 """
    train_df, _ = train_test_split(train, test_size=0.7, random_state=42, shuffle=True, stratify=train['orderType'])
    selector = xgb_feature_selector.XgboostGreedyFeatureSelector(train_df.drop(['orderType'], axis=1), train_df['orderType'])

    xgb_params = {
        'eta': 0.05,
        'colsample_bytree': 0.8,
        'max_depth': 4,
        'subsample': 0.9,
        'lambda': 2.0,
        'scale_pos_weight': 1,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'updater': 'grow_gpu',
        'gpu_id': 0,
        'nthread': -1,
        'silent': 1,
        'booster': 'gbtree'
    }
    base_features = ['actiontimespanlast_5_6', 'last_-1_x_actiontype', 'country_avg_rich', 'timespan_action6tolast',
                     'timespanmin_last_3', 'actionratio_24_59', 'actiontimespancount_5_6', 'action_type_56_time_delta_std',
                     'timespan_action24tolast']

    best_subset_features = selector.select_best_subset_features(xgb_params=xgb_params, cv_nfold=4,
                                                                selected_feature_size=int(train_df.shape[1] * selected_size),
                                                                num_boost_round=1000, base_features=base_features,
                                                                early_stopping_rounds=50, maximize=True,
                                                                stratified=True, shuffle=True)
    return best_subset_features


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

    print('特征组合')
    train, test = feature_interaction(train, test)

    print('连续特征离散化')
    train, test = discretize_features(train, test)

    print('贪心算法特征选择')
    selected_size = 0.9
    best_subset_features_path = 'best_subset_{}_features.pkl'.format(selected_size)
    if not os.path.exists(best_subset_features_path):
        best_subset_features = feature_selection(train, selected_size)
        with open(best_subset_features_path, "wb") as f:
            cPickle.dump(best_subset_features, f, -1)
    else:
        with open(best_subset_features_path, "rb") as f:
            best_subset_features = cPickle.load(f)
    #
    # with open('./xgboost_best_subfeatures/best_subset_10_features_cv_0.956655386364.pkl', "rb") as f:
    #     best_subset_features = cPickle.load(f)

    return train, test, best_subset_features
