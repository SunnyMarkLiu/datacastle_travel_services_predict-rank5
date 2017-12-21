#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-12-10 上午11:27
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
from pypinyin import lazy_pinyin
from sklearn.preprocessing import LabelEncoder
from conf.configure import Configure
from optparse import OptionParser
from utils import data_utils


def user_basic_info():
    """
    用户个人信息
    """
    train_user = pd.read_csv(Configure.base_path + 'train/userProfile_train.csv', encoding='utf8')
    test_user = pd.read_csv(Configure.base_path + 'test/userProfile_test.csv', encoding='utf8')

    def gender_convert(gender):
        if gender == gender:
            return 'man' if gender == u'男' else 'woman'
        return 'None'

    train_user['gender'] = train_user['gender'].map(gender_convert)
    test_user['gender'] = test_user['gender'].map(gender_convert)
    dummies = pd.get_dummies(train_user['gender'], prefix='gender')
    train_user[dummies.columns] = dummies
    dummies = pd.get_dummies(test_user['gender'], prefix='gender')
    test_user[dummies.columns] = dummies

    def province_convert(province):
        if province == province:
            return '_'.join(lazy_pinyin(province))
        return 'None'

    train_user['province'] = train_user['province'].map(province_convert)
    test_user['province'] = test_user['province'].map(province_convert)

    le = LabelEncoder()
    le.fit(train_user['province'].values)
    train_user['province_code'] = le.transform(train_user['province'])
    test_user['province_code'] = le.transform(test_user['province'])

    train_user['age'] = train_user['age'].map(lambda age: 'lg' + age[:2] if age == age else 'None')
    test_user['age'] = test_user['age'].map(lambda age: 'lg' + age[:2] if age == age else 'None')
    dummies = pd.get_dummies(train_user['age'], prefix='age')
    train_user[dummies.columns] = dummies
    dummies = pd.get_dummies(test_user['age'], prefix='age')
    test_user[dummies.columns] = dummies

    return train_user, test_user


def basic_action_info():
    """
    用户行为信息
    """
    action_train = pd.read_csv(Configure.base_path + 'train/action_train.csv')
    action_test = pd.read_csv(Configure.base_path + 'test/action_test.csv')
    action_train['actionTime'] = pd.to_datetime(action_train['actionTime'], unit='s')
    action_test['actionTime'] = pd.to_datetime(action_test['actionTime'], unit='s')
    train_action_features = action_train.groupby(['userid']).count().reset_index()[['userid', 'actionTime']].rename(
        columns={'actionTime': 'action_counts'})
    test_action_features = action_train.groupby(['userid']).count().reset_index()[['userid', 'actionTime']].rename(
        columns={'actionTime': 'action_counts'})

    return train_action_features, test_action_features


def main(op_scope):
    op_scope = int(op_scope)
    # if os.path.exists(Configure.processed_train_path.format(op_scope)):
    #     return

    print("---> load datasets")
    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')
    print("train: {}, test: {}".format(train.shape, test.shape))

    print('---> 合并用户基本信息')
    train_user, test_user = user_basic_info()
    train = train.merge(train_user, on='userid', how='left')
    test = test.merge(test_user, on='userid', how='left')

    # print('---> 合并用户行为信息')
    # train_action_features, test_action_features = basic_action_info()
    # train = train.merge(train_action_features, on='userid', how='left')
    # test = test.merge(test_action_features, on='userid', how='left')

    # 用户历史订单数据
    orderHistory_train = pd.read_csv(Configure.base_path + 'train/orderHistory_train.csv',encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.base_path + 'test/orderHistory_test.csv',encoding='utf8')

    # 评论数据
    userComment_train = pd.read_csv(Configure.base_path + 'train/userComment_train.csv', encoding='utf8')
    userComment_test = pd.read_csv(Configure.base_path + 'test/userComment_test.csv', encoding='utf8')

    print("train: {}, test: {}".format(train.shape, test.shape))
    print("---> save datasets")
    data_utils.save_dataset(train, test, op_scope)


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "-o", "--op_scope",
        dest="op_scope",
        default="0",
        help="""operate scope: 0, 1, 2, 3..."""
    )

    options, _ = parser.parse_args()
    print("========== generate some simple features ==========")
    main(options.op_scope)
