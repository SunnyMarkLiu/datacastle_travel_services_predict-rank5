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

import pandas as pd
from pypinyin import lazy_pinyin
from sklearn.preprocessing import LabelEncoder
from conf.configure import Configure
from optparse import OptionParser
from utils import data_utils


def user_basic_info():
    """
    用户个人基本信息
    """
    train_user = pd.read_csv(Configure.base_path + 'train/userProfile_train.csv', encoding='utf8')
    test_user = pd.read_csv(Configure.base_path + 'test/userProfile_test.csv', encoding='utf8')

    def gender_convert(gender):
        if gender == gender:
            return 'man' if gender == u'男' else 'woman'
        return 'None'

    # 性别 dummy code
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
    # 省份进行 LabelEncoder
    le = LabelEncoder()
    le.fit(train_user['province'].values)
    train_user['province_code'] = le.transform(train_user['province'])
    test_user['province_code'] = le.transform(test_user['province'])

    # 年龄段进行 dummy code
    train_user['age'] = train_user['age'].map(lambda age: 'lg' + age[:2] if age == age else 'None')
    test_user['age'] = test_user['age'].map(lambda age: 'lg' + age[:2] if age == age else 'None')
    dummies = pd.get_dummies(train_user['age'], prefix='age')
    train_user[dummies.columns] = dummies
    dummies = pd.get_dummies(test_user['age'], prefix='age')
    test_user[dummies.columns] = dummies

    return train_user, test_user


def basic_action_info(action_df):
    """
    用户行为信息
    """
    action_df['actionTime'] = pd.to_datetime(action_df['actionTime'], unit='s')

    def action_type_convert(action):
        if action == 1:
            return 'open_app'
        elif 2 <= action <= 4:
            return 'browse_product'
        elif action == 5:
            return 'fillin_form5'
        elif action == 6:
            return 'fillin_form6'
        elif action == 7:
            return 'fillin_form7'
        elif action == 8:
            return 'submit_order'
        elif action == 9:
            return 'pay_money'

    # action 操作，合并无序的浏览特征
    action_df['actionType'] = action_df['actionType'].map(action_type_convert)

    print('用户不同操作的购买率')
    action_features = action_df.groupby(['userid', 'actionType']).actionTime.count().groupby(level=0).apply(lambda x: x.astype(float) / x.sum()).reset_index()
    action_features = action_features.pivot('userid', 'actionType', 'actionTime').reset_index().fillna(0)

    action_features = action_features.merge(
        action_df.groupby(['userid']).count().reset_index()[['userid', 'actionTime']].rename(columns={'actionTime': 'action_counts'}),
        on='userid', how='left'
    )
    action_features.rename(columns={'open_app': 'open_app_ratio', 'browse_product': 'browse_product_ratio',
                                    'fillin_form5': 'fillin_form5_ratio', 'fillin_form6': 'fillin_form6_ratio',
                                    'fillin_form7': 'fillin_form7_ratio', 'submit_order': 'submit_order_ratio',
                                    'pay_money': 'pay_money_ratio'}, inplace=True)
    action_features['has_pay_money'] = (action_features['pay_money_ratio'] > 0).astype(int)

    print('拉普拉斯平滑计算基本的转化率')
    action_features['open_app_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['open_app_ratio']+ 0.2)
    action_features['browse_product_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['browse_product_ratio']+ 0.2)
    action_features['fillin_form5_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['fillin_form5_ratio']+ 0.2)
    action_features['fillin_form6_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['fillin_form6_ratio']+ 0.2)
    action_features['fillin_form7_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['fillin_form7_ratio']+ 0.2)
    action_features['submit_order_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['submit_order_ratio']+ 0.2)

    action_df['action_year'] = action_df['actionTime'].dt.year
    action_df['action_month'] = action_df['actionTime'].dt.month
    action_df['action_day'] = action_df['actionTime'].dt.day
    action_df['action_weekofyear'] = action_df['actionTime'].dt.weekofyear
    action_df['action_weekday'] = action_df['actionTime'].dt.weekday
    action_df['action_hour'] = action_df['actionTime'].dt.hour
    action_df['action_minute'] = action_df['actionTime'].dt.minute
    action_df['action_is_weekend'] = action_df['action_weekday'].map(lambda d: 1 if (d == 0) | (d == 6) else 0)
    action_df['action_week_hour'] = action_df['action_weekday'] * 24 + action_df['action_hour']

    print('每年 action 的情况')
    action_features = action_features.merge(
        action_df.groupby(['userid', 'action_year']).count().reset_index()[['userid', 'action_year', 'actionTime']] \
            .pivot('userid', 'action_year', 'actionTime').reset_index().fillna(0).rename(
            columns={2016: '2016_year_pay_money_count', 2017: '2017_year_pay_money_count'}),
        on='userid', how='left'
    )
    action_features['year_action_count'] = action_features['userid'].map(lambda userid: len(action_df[action_df['userid'] == userid]['action_year'].unique()))

    print('每月 action 的情况')
    action_features = action_features.merge(
        action_df.groupby(['userid', 'action_month']).count().reset_index()[['userid', 'action_month', 'actionTime']] \
            .pivot('userid', 'action_month', 'actionTime').reset_index().fillna(0).rename(columns={i: 'month_{}_count'.format(i) for i in range(1, 13)}),
        on='userid', how='left'
    )
    action_features['month_action_count'] = action_features['userid'].map(lambda userid: len(action_df[action_df['userid'] == userid]['action_month'].unique()))

    print('最清闲的在几月，以及 action 的次数')
    def get_most_free_month(row):
        most_count = 0
        most_month = -1
        for i in range(1,13):
            if row['month_{}_count'.format(i)] > most_count:
                most_month = i
        return most_month

    action_features['most_free_month'] = action_features.apply(lambda row: get_most_free_month(row), axis=1)
    action_features['most_free_month_action_count'] = action_features.apply(lambda row: row['month_{}_count'.format(int(row['most_free_month']))], axis=1)

    return action_features


def main(op_scope):
    op_scope = int(op_scope)
    if os.path.exists(Configure.processed_train_path.format(op_scope)):
        return

    print("---> load datasets")
    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')
    print("train: {}, test: {}".format(train.shape, test.shape))

    print('---> 合并用户基本信息')
    train_user, test_user = user_basic_info()
    train = train.merge(train_user, on='userid', how='left')
    test = test.merge(test_user, on='userid', how='left')

    print('---> 合并用户行为信息')
    action_train = pd.read_csv(Configure.base_path + 'train/action_train.csv')
    action_test = pd.read_csv(Configure.base_path + 'test/action_test.csv')

    train_action_features = basic_action_info(action_train)
    train = train.merge(train_action_features, on='userid', how='left')
    test_action_features = basic_action_info(action_test)
    test = test.merge(test_action_features, on='userid', how='left')

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
