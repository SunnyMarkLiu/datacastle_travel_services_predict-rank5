#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-28 下午5:03
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm
import numpy as np
import pandas as pd
from conf.configure import Configure
from utils import data_utils


def days_action_count(uid, action_grouped, days, actiontype=None):
    """ days_from_now 的 action 数量 """
    df = action_grouped[uid]
    if actiontype is None:
        return df[df['days_from_now'] == days].shape[0]
    else:
        df = df[df['days_from_now'] == days]
        return df[df['actionType'] == actiontype].shape[0]


def build_basic_action_features(df, action):
    features = pd.DataFrame({'userid': df['userid']})
    # days_from_nows = range(1, 5)
    days_from_nows = range(1, 395)
    action_grouped = dict(list(action.groupby('userid')))

    for i in tqdm(days_from_nows):
        features['days_{}_action_count'.format(i)] = features.apply(lambda row: days_action_count(row['userid'], action_grouped, i), axis=1)
        features['days_{}_action_open_app_count'.format(i)] = features.apply(lambda row: days_action_count(row['userid'], action_grouped, i, 'open_app'), axis=1)
        features['days_{}_action_browse_product_count'.format(i)] = features.apply(lambda row: days_action_count(row['userid'], action_grouped, i, 'browse_product'), axis=1)
        features['days_{}_action_fillin_form5_count'.format(i)] = features.apply(lambda row: days_action_count(row['userid'], action_grouped, i, 'fillin_form5'), axis=1)
        features['days_{}_action_fillin_form6_count'.format(i)] = features.apply(lambda row: days_action_count(row['userid'], action_grouped, i, 'fillin_form6'), axis=1)
        features['days_{}_action_fillin_form7_count'.format(i)] = features.apply(lambda row: days_action_count(row['userid'], action_grouped, i, 'fillin_form7'), axis=1)
        features['days_{}_action_submit_order_count'.format(i)] = features.apply(lambda row: days_action_count(row['userid'], action_grouped, i, 'submit_order'), axis=1)
        features['days_{}_action_pay_money_count'.format(i)] = features.apply(lambda row: days_action_count(row['userid'], action_grouped, i, 'pay_money'), axis=1)

    return features


def main():
    feature_name = 'basic_action_features'
    # if data_utils.is_feature_created(feature_name):
    #     return

    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')

    train_action = pd.read_csv(Configure.cleaned_path + 'cleaned_action_train.csv')
    test_action = pd.read_csv(Configure.cleaned_path + 'cleaned_action_test.csv')

    print('build train ', feature_name)
    train_features = build_basic_action_features(train, train_action)
    print('build test ', feature_name)
    test_features = build_basic_action_features(test, test_action)
    print('save ', feature_name)

    data_utils.save_features(train_features, test_features, features_name=feature_name)


if __name__ == "__main__":
    print("========== 暴力抽取 action 基本信息 ==========")
    main()
