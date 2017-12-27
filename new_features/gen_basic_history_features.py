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


def has_history_flag(uid, cur_orderTime, history_grouped, in_total_flag):
    if in_total_flag == 0:
        return 0

    df = history_grouped[uid]
    if df.shape[0] == 0:
        return 0
    else:
        sub_df = df[df['orderTime'] < cur_orderTime]
        return int(sub_df.shape[0] > 0)


def build_basic_history_features(df, history):
    """ 构造 order 历史的基本特征 """
    features = pd.DataFrame({'userid': df['userid'], 'orderTime': df['orderTime']})

    history_uids = history['userid'].unique()
    history_grouped = dict(list(history.groupby('userid')))
    # 是否在全局的 history 出现过，方便 code，删除该特征
    features['uid_in_history_flag'] = features['userid'].map(lambda uid: uid in history_uids).astype(int)
    # 给 trade 表打标签，是否在之前的 history 中出现过
    features['has_pre_history_flag'] = features.apply(lambda row: has_history_flag(row['userid'], row['orderTime'], history_grouped, row['uid_in_history_flag']), axis=1)
    # 最近的一次交易信息


    del features['uid_in_history_flag']
    return features


def build_time_category_encode(history):
    history['orderTime'] = pd.to_datetime(history['orderTime'], unit='s')

    # 训练集和测试集最后一天是 2017-09-11
    now = datetime.datetime(2017, 9, 12)
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
    # if data_utils.is_new_feature_created(feature_name):
    #     return

    train, test = data_utils.load_new_train_test()

    orderHistory_train = pd.read_csv(Configure.base_path + 'train/orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.base_path + 'test/orderHistory_test.csv', encoding='utf8')

    orderHistory_train = build_time_category_encode(orderHistory_train)
    orderHistory_test = build_time_category_encode(orderHistory_test)

    print('save cleaned datasets')
    orderHistory_train.to_csv(Configure.new_cleaned_path + 'cleaned_orderHistory_train.csv', index=False, columns=orderHistory_train.columns)
    orderHistory_test.to_csv(Configure.new_cleaned_path + 'cleaned_orderHistory_test.csv', index=False, columns=orderHistory_test.columns)

    print('build train features')
    train_features = build_basic_history_features(train, orderHistory_train)
    print('build test features')
    test_features = build_basic_history_features(test, orderHistory_test)

    print('save ', feature_name)
    data_utils.save_new_features(train_features, test_features, features_name=feature_name)




if __name__ == "__main__":
    print("========== order 历史的基本特征 ==========")
    main()
