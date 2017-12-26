#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-25 下午9:18
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


def contain_father_son_order(uid, userid_grouped, flag):
    """ 是否存在父子订单 """
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    if df.shape[0] == 0:
        return 0
    else:
        return int(len(set(df['orderTime'])) < df.shape[0])


def last_two_order_days_delta(uid, userid_grouped, flag):
    """ 最近两次 order 的 days 间隔 """
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    if df.shape[0] < 2:
        return -1
    else:
        return df.iloc[-2]['days_from_now'] - df.iloc[-1]['days_from_now']


def last_two_order_same_country(uid, userid_grouped, flag):
    """ 最近两次 order 是否为同一城市 """
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    if df.shape[0] < 2:
        return -1
    else:
        return int(df.iloc[-2]['country'] == df.iloc[-1]['country'])


def last_order_is_not_first_use(uid, userid_grouped, flag):
    """ 最近一个月是否是第一次使用 order """
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    if df.shape[0] == 0:
        return -1
    elif df.shape[0] == 1:
        return 1
    else:
        return int(df['order_month'].min() < df.iloc[-1]['order_month'])


def build_history_merged_features(df, history):
    features = pd.DataFrame({'userid': df['userid']})

    df_ids = history['userid'].unique()
    userid_grouped = dict(list(history.groupby('userid')))
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in df_ids).astype(int)
    # 是否存在父子订单
    features['contain_father_son_order'] = features.apply(lambda row: contain_father_son_order(row['userid'], userid_grouped, row['has_history_flag']), axis=1)
    # 对于存在父子订单的，进行数据清洗，去除 ordertype=0 的重复记录
    print('history: ', history.shape)
    # cleaned_users = features[features['contain_father_son_order'] == 1]['userid']
    # for userid in cleaned_users:
    #     ordertime_occre = {}
    #     deleted_index = []
    #     father_son_order_times = []
    #     unique_time_indexs = []
    #     for index, row_ in history[history['userid'] == userid].iterrows():
    #         ordertime_occre[row_['orderTime']] = ordertime_occre.get(row_['orderTime'], 0) + 1
    #         if ordertime_occre[row_['orderTime']] == 2:
    #             ordertime_occre[row_['orderTime']] = ordertime_occre.get(row_['orderTime'], 0) - 1
    #             deleted_index.append(index)
    #             father_son_order_times.append(row_['orderTime'])
    #             continue
    #
    #         unique_time_indexs.append(index)
    #
    #     replace_orderType_indexs = []
    #     for i in unique_time_indexs:
    #         if history.loc[i, 'orderTime'] in father_son_order_times:
    #             replace_orderType_indexs.append(i)
    #
    #     history.drop(deleted_index, axis=0, inplace=True)
    #     history.loc[replace_orderType_indexs, 'orderType'] = 1
    # history = history.reset_index(drop=True)
    # userid_grouped = dict(list(history.groupby('userid')))
    print('cleaned history: ', history.shape)

    # 最近两次 order 的 days 间隔
    features['last_two_order_days_delta'] = features.apply(lambda row: last_two_order_days_delta(row['userid'], userid_grouped, row['has_history_flag']), axis=1)
    # 最近两次 order 是否为同一城市
    # features['last_two_order_same_country'] = features.apply(lambda row: last_two_order_same_country(row['userid'], userid_grouped, row['has_history_flag']), axis=1)
    # 最近一个月是否是第一次使用 order
    # features['last_order_is_not_first_use'] = features.apply(lambda row: last_order_is_not_first_use(row['userid'], userid_grouped, row['has_history_flag']), axis=1)

    features.drop(['has_history_flag', 'contain_father_son_order'], axis=1, inplace=True)
    return features


def main():
    feature_name = 'history_merged_features'
    # if data_utils.is_feature_created(feature_name):
    #     return

    print('load cleaned datasets')
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')

    user_train = pd.read_csv(Configure.cleaned_path + 'cleaned_userProfile_train.csv', encoding='utf8')
    user_test = pd.read_csv(Configure.cleaned_path + 'cleaned_userProfile_test.csv', encoding='utf8')
    action_train = pd.read_csv(Configure.cleaned_path + 'cleaned_action_train.csv')
    action_test = pd.read_csv(Configure.cleaned_path + 'cleaned_action_test.csv')
    orderHistory_train = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_test.csv', encoding='utf8')
    userComment_train = pd.read_csv(Configure.cleaned_path + 'cleaned_userComment_train.csv', encoding='utf8')
    userComment_test = pd.read_csv(Configure.cleaned_path + 'cleaned_userComment_test.csv', encoding='utf8')

    train_history = orderHistory_train.merge(userComment_train, on=['userid', 'orderid'], how='left')
    test_history = orderHistory_test.merge(userComment_test, on=['userid', 'orderid'], how='left')
    train_history = train_history.merge(user_train, on='userid', how='left')
    test_history = test_history.merge(user_test, on='userid', how='left')

    print('build train history merged features')
    train_features = build_history_merged_features(train, train_history)
    print('build test history merged features')
    test_features = build_history_merged_features(test, test_history)

    print('save ', feature_name)
    data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 结合 action、 history 和 comment 提取历史特征 ==========")
    main()
