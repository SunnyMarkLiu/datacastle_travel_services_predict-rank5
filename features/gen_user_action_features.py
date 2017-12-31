#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-22 下午5:15
"""
from __future__ import absolute_import, division, print_function

# remove warnings
import warnings
warnings.filterwarnings('ignore')

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import datetime
import pandas as pd
from conf.configure import Configure
from utils import data_utils


def basic_action_info(action_df):
    """
    用户行为信息
    """
    print('用户不同操作的购买率')
    action_features = action_df.groupby(['userid', 'actionType']).actionTime.count().groupby(level=0).apply(lambda x: x.astype(float) / x.sum()).reset_index()
    action_features = action_features.pivot('userid', 'actionType', 'actionTime').reset_index().fillna(0)

    action_features = action_features.merge(
        action_df.groupby(['userid']).count().reset_index()[['userid', 'actionTime']].rename(columns={'actionTime': 'action_counts'}),
        on='userid', how='left'
    )
    action_features.rename(columns={'open_app': 'open_app_ratio', 'browse_product': 'browse_product_ratio',
                                    'browse_product2': 'browse_product2_ratio', 'browse_product3': 'browse_product3_ratio',
                                    'fillin_form5': 'fillin_form5_ratio', 'fillin_form6': 'fillin_form6_ratio',
                                    'fillin_form7': 'fillin_form7_ratio', 'submit_order': 'submit_order_ratio',
                                    'pay_money': 'pay_money_ratio'}, inplace=True)
    action_features['has_pay_money'] = (action_features['pay_money_ratio'] > 0).astype(int)

    print('拉普拉斯平滑计算基本的转化率')
    action_features['open_app_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['open_app_ratio']+ 0.2)
    action_features['browse_product_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['browse_product_ratio']+ 0.2)
    action_features['browse_product2_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['browse_product2_ratio']+ 0.2)
    action_features['browse_product3_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['browse_product3_ratio']+ 0.2)
    action_features['fillin_form5_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['fillin_form5_ratio']+ 0.2)
    action_features['fillin_form6_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['fillin_form6_ratio']+ 0.2)
    action_features['fillin_form7_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['fillin_form7_ratio']+ 0.2)
    action_features['submit_order_pay_money_ratio'] = (action_features['pay_money_ratio']+ 0.1) / (action_features['submit_order_ratio']+ 0.2)

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


def build_time_features(action_df):

    action_df['actionTime'] = pd.to_datetime(action_df['actionTime'], unit='s')
    # 训练集和测试集最后一天是 2017-09-11
    now = datetime.datetime(2017, 9, 12)
    action_df['days_from_now'] = action_df['actionTime'].map(lambda order: (now - order).days)

    action_df['action_year'] = action_df['actionTime'].dt.year
    action_df['action_month'] = action_df['actionTime'].dt.month
    action_df['action_day'] = action_df['actionTime'].dt.day
    action_df['action_weekofyear'] = action_df['actionTime'].dt.weekofyear
    action_df['action_weekday'] = action_df['actionTime'].dt.weekday
    action_df['action_hour'] = action_df['actionTime'].dt.hour
    action_df['action_minute'] = action_df['actionTime'].dt.minute
    action_df['action_is_weekend'] = action_df['action_weekday'].map(lambda d: 1 if (d == 0) | (d == 6) else 0)
    action_df['action_week_hour'] = action_df['action_weekday'] * 24 + action_df['action_hour']

    def action_type_convert(action):
        if action == 1:
            return 'open_app'
        elif action == 2:
            return 'browse_product'
        elif action == 3:
            return 'browse_product2'
        elif action == 4:
            return 'browse_product3'
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

    return action_df

def main():
    feature_name = 'basic_user_action_features'
    if data_utils.is_feature_created(feature_name):
        return

    train_action = pd.read_csv(Configure.base_path + 'train/action_train.csv')
    test_action = pd.read_csv(Configure.base_path + 'test/action_test.csv')

    train_action = build_time_features(train_action)
    test_action = build_time_features(test_action)

    print('save cleaned datasets')
    train_action.to_csv(Configure.cleaned_path + 'cleaned_action_train.csv', index=False, columns=train_action.columns)
    test_action.to_csv(Configure.cleaned_path + 'cleaned_action_test.csv', index=False, columns=test_action.columns)

    train_action_features = basic_action_info(train_action)
    test_action_features = basic_action_info(test_action)

    print('save ', feature_name)
    data_utils.save_features(train_action_features, test_action_features, features_name=feature_name)


if __name__ == "__main__":
    print("========== 生成用户行为 action 的基本特征 ==========")
    main()
