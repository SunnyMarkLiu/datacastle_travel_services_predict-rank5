#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-27 下午6:02
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings

warnings.filterwarnings('ignore')

import datetime
import pandas as pd
from pypinyin import lazy_pinyin
from sklearn.preprocessing import LabelEncoder
from conf.configure import Configure
from utils import data_utils


# 训练集和测试集最后一天是 2017-09-11
now = datetime.datetime(2017, 9, 12)

def has_history_flag(uid, cur_orderTime, history_grouped, in_total_flag):
    if in_total_flag == 0:
        return 0

    df = history_grouped[uid]
    if df.shape[0] == 0:
        return 0
    else:
        sub_df = df[df['orderTime'] < cur_orderTime]
        return int(sub_df.shape[0] > 0)


def check_last_time_order_info(uid, cur_orderTime, history_grouped, in_total_flag, check_name):
    if in_total_flag == 0:
        return -1

    df = history_grouped[uid]
    if df.shape[0] == 0:
        return -1
    else:
        sub_df = df[df['orderTime'] < cur_orderTime]
        if sub_df.shape[0] == 0:
            return -1
        return sub_df.iloc[-1][check_name]


def pre_order_count(uid, cur_orderTime, history_grouped, in_total_flag):
    """ 以往交易的次数 """
    if in_total_flag == 0:
        return 0
    df = history_grouped[uid]
    if df.shape[0] == 0:
        return 0
    else:
        sub_df = df[df['orderTime'] < cur_orderTime]
        if sub_df.shape[0] == 0:
            return 0
        return sub_df.shape[0]

def pre_good_order_count(uid, cur_orderTime, history_grouped, in_total_flag):
    """ 以往交易的次数 """
    if in_total_flag == 0:
        return 0
    df = history_grouped[uid]
    if df.shape[0] == 0:
        return 0
    else:
        sub_df = df[df['orderTime'] < cur_orderTime]
        if sub_df.shape[0] == 0:
            return 0
        return sum(sub_df['orderType'])


def pre_days_order_count(uid, cur_orderTime, history_grouped, in_total_flag, days):
    """ 往前 days 的 order 数量 """
    if in_total_flag == 0:
        return 0
    df = history_grouped[uid]
    if df.shape[0] == 0:
        return 0
    else:
        sub_df = df[df['orderTime'] < cur_orderTime]
        sub_df = sub_df.loc[df['days_from_now'] < ((now - cur_orderTime).days - days)]
        return sub_df.shape[0]


def pre_good_days_order_count(uid, cur_orderTime, history_grouped, in_total_flag, days):
    """ 往前 days 的 order 数量 """
    if in_total_flag == 0:
        return 0
    df = history_grouped[uid]
    if df.shape[0] == 0:
        return 0
    else:
        sub_df = df[df['orderTime'] < cur_orderTime]
        sub_df = sub_df.loc[df['days_from_now'] < ((now - cur_orderTime).days - days)]
        return sum(sub_df['orderType'])


def build_basic_history_features(df, history):
    """ 构造 order 历史的基本特征 """
    features = pd.DataFrame({'userid': df['userid'], 'orderTime': df['orderTime']})

    history_uids = history['userid'].unique()
    history_grouped = dict(list(history.groupby('userid')))
    print('flag 标记特征')
    # 是否在全局的 history 出现过，方便 code，删除该特征
    features['uid_in_history_flag'] = features['userid'].map(lambda uid: uid in history_uids).astype(int)
    # 给 trade 表打标签，是否在之前的 history 中出现过
    features['has_pre_history_flag'] = features.apply(lambda row: has_history_flag(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag']), axis=1)

    print('最近的一次交易类型特征')
    features['last_time_orderType'] = features.apply(lambda row: check_last_time_order_info(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 'orderType'), axis=1)

    print('最近的一次交易时间特征')
    features['last_time_days_from_now'] = features.apply(lambda row: check_last_time_order_info(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 'days_from_now'), axis=1)
    features['last_time_order_year'] = features.apply(lambda row: check_last_time_order_info(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 'order_year'), axis=1)
    features['last_time_order_month'] = features.apply(lambda row: check_last_time_order_info(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 'order_month'), axis=1)
    features['last_time_order_day'] = features.apply(lambda row: check_last_time_order_info(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 'order_day'), axis=1)
    features['last_time_order_weekofyear'] = features.apply(lambda row: check_last_time_order_info(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 'order_weekofyear'), axis=1)
    features['last_time_order_order_weekday'] = features.apply(lambda row: check_last_time_order_info(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 'order_weekday'), axis=1)

    print('最近的一次交易地区特征')
    features['last_time_order_continent'] = features.apply(lambda row: check_last_time_order_info(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 'continent'), axis=1)
    features['last_time_order_country'] = features.apply(lambda row: check_last_time_order_info(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 'country'), axis=1)
    features['last_time_order_city'] = features.apply(lambda row: check_last_time_order_info(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 'city'), axis=1)

    print('以往交易的次数及比例')
    features['pre_order_count'] = features.apply(lambda row: pre_order_count(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag']), axis=1)
    features['pre_good_order_count'] = features.apply(lambda row: pre_good_order_count(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag']), axis=1)
    features['pre_order_good_ratio'] = (features['pre_good_order_count'] + 1) / (features['pre_order_count'] + 2) - 0.5

    print('往前 90days 的计数特征')
    features['pre_90days_order_count'] = features.apply(lambda row: pre_days_order_count(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 90), axis=1)
    features['pre_90days_good_order_count'] = features.apply(lambda row: pre_good_days_order_count(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag'], 90), axis=1)
    features['pre_order_good_ratio'] = (features['pre_90days_order_count'] + 1) / (features['pre_90days_good_order_count'] + 2) - 0.5

    del features['uid_in_history_flag']
    return features


def build_time_category_encode(history):
    history['orderTime'] = pd.to_datetime(history['orderTime'], unit='s')

    history['days_from_now'] = history['orderTime'].map(lambda order: (now - order).days)
    history['order_year'] = history['orderTime'].dt.year
    history['order_month'] = history['orderTime'].dt.month
    history['order_day'] = history['orderTime'].dt.day
    history['order_weekofyear'] = history['orderTime'].dt.weekofyear
    history['order_weekday'] = history['orderTime'].dt.weekday
    history['order_hour'] = history['orderTime'].dt.hour
    history['order_minute'] = history['orderTime'].dt.minute
    history['order_is_weekend'] = history['orderTime'].map(lambda d: 1 if (d == 0) | (d == 6) else 0)
    history['order_week_hour'] = history['order_weekday'] * 24 + history['order_hour']
    # 按照时间排序
    history = history.sort_values(by='orderTime')

    history['continent'] = history['continent'].map(lambda c: '_'.join(lazy_pinyin(c)) if c == c else 'None')
    history['country'] = history['country'].map(lambda c: '_'.join(lazy_pinyin(c)) if c == c else 'None')
    history['city'] = history['city'].map(lambda c: '_'.join(lazy_pinyin(c)) if c == c else 'None')

    le = LabelEncoder()
    le.fit(history['continent'].values)
    history['continent'] = le.transform(history['continent'])
    le = LabelEncoder()
    le.fit(history['country'].values)
    history['country'] = le.transform(history['country'])
    le = LabelEncoder()
    le.fit(history['city'].values)
    history['city'] = le.transform(history['city'])

    history.sort_values(by='orderTime', inplace=True)

    return history

def main():
    feature_name = 'basic_history_features'
    if data_utils.is_new_feature_created(feature_name):
        return

    train, test = data_utils.load_new_train_test()

    orderHistory_train = pd.read_csv(Configure.base_path + 'train/orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.base_path + 'test/orderHistory_test.csv', encoding='utf8')

    orderHistory_train = build_time_category_encode(orderHistory_train)
    orderHistory_test = build_time_category_encode(orderHistory_test)

    print('save cleaned datasets')
    orderHistory_train.to_csv(Configure.new_cleaned_path + 'cleaned_orderHistory_train.csv', index=False, columns=orderHistory_train.columns)
    orderHistory_test.to_csv(Configure.new_cleaned_path + 'cleaned_orderHistory_test.csv', index=False, columns=orderHistory_test.columns)

    print('build train ', feature_name)
    train_features = build_basic_history_features(train, orderHistory_train)
    print('build test ', feature_name)
    test_features = build_basic_history_features(test, orderHistory_test)
    print('save ', feature_name)
    data_utils.save_new_features(train_features, test_features, features_name=feature_name)

if __name__ == "__main__":
    print("========== 订单历史的基本特征 ==========")
    main()
