#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
将 order 订单作为一种新的 actiontype

@author: SunnyMarkLiu
@time  : 18-1-30 下午1:00
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
from conf.configure import Configure
from utils import data_utils


def diff_action_type_time_delta(uid, action_grouped, actiontypeA, actiontypeB):

    df_action_of_userid = action_grouped[uid]

    timespan_list = []
    i = 0
    while i < (len(df_action_of_userid)-1):
        if df_action_of_userid['actionType'].iat[i] == actiontypeA:
            timeA = df_action_of_userid['actionTime'].iat[i]
            for j in range(i+1, len(df_action_of_userid)):
                if df_action_of_userid['actionType'].iat[j] == actiontypeA:
                    timeA = df_action_of_userid['actionTime'].iat[j]
                    continue
                if df_action_of_userid['actionType'].iat[j] == actiontypeB:
                    timeB = df_action_of_userid['actionTime'].iat[j]
                    timespan_list.append(timeB-timeA)
                    i = j
        i+=1
    if len(timespan_list) > 0:
        return np.min(timespan_list), np.max(timespan_list), np.mean(timespan_list), np.std(timespan_list)
    else:
        return -999, -999, -999, -999


def build_action_order_features1(df, action_grouped):
    features = pd.DataFrame({'userid': df['userid']})

    # 已初步筛选
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 1, 10), axis=1)
    features['action_type_110_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 2, 10), axis=1)
    features['action_type_210_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 5, 10), axis=1)
    features['action_type_510_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_510_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    features['action_type_510_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 6, 10), axis=1)
    features['action_type_610_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_610_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 7, 10), axis=1)
    features['action_type_710_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_710_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 8, 10), axis=1)
    features['action_type_810_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_810_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 9, 10), axis=1)
    features['action_type_910_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    del features['diff_action_type_time_delta']

    # 已初步筛选
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 5, 11), axis=1)
    features['action_type_511_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_511_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    features['action_type_511_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    del features['diff_action_type_time_delta']

    return features


def two_gram_statistic(uid, action_grouped):
    action_df = action_grouped[uid]
    action_types = action_df['actionType'].values.tolist()

    if len(action_types) < 2:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    two_gram_words = []
    for i in range(len(action_types) - 1):
        two_gram_words.append((action_types[i], action_types[i+1]))

    len_two_gram = 1.0 * len(two_gram_words)
    action_12_ratio = two_gram_words.count((1,2)) / len_two_gram
    action_23_ratio = two_gram_words.count((2,3)) / len_two_gram
    action_34_ratio = two_gram_words.count((3,4)) / len_two_gram
    action_45_ratio = two_gram_words.count((4,5)) / len_two_gram
    action_56_ratio = two_gram_words.count((5,6)) / len_two_gram
    action_67_ratio = two_gram_words.count((6,7)) / len_two_gram
    action_78_ratio = two_gram_words.count((7,8)) / len_two_gram
    action_89_ratio = two_gram_words.count((8,9)) / len_two_gram

    last_sum = sum(two_gram_words[-1])
    last2_sum = 0
    if len_two_gram > 1:
        last2_sum = sum(two_gram_words[-2])

    return action_12_ratio, action_23_ratio, action_34_ratio, action_45_ratio, action_56_ratio, \
           action_67_ratio, action_78_ratio, action_89_ratio, last_sum, last2_sum

def three_gram_statistic(uid, action_grouped):
    action_df = action_grouped[uid]
    action_types = action_df['actionType'].values.tolist()

    if len(action_types) < 3:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    three_gram_words = []
    for i in range(len(action_types) - 2):
        three_gram_words.append((action_types[i], action_types[i+1], action_types[i+2]))

    len_three_gram = 1.0 * len(three_gram_words)
    action_123_ratio = three_gram_words.count((1,2,3)) / len_three_gram
    action_234_ratio = three_gram_words.count((2,3,4)) / len_three_gram
    action_345_ratio = three_gram_words.count((3,4,5)) / len_three_gram
    action_456_ratio = three_gram_words.count((4,5,6)) / len_three_gram
    action_567_ratio = three_gram_words.count((5,6,7)) / len_three_gram
    action_678_ratio = three_gram_words.count((6,7,8)) / len_three_gram
    action_789_ratio = three_gram_words.count((7,8,9)) / len_three_gram

    last_sum = sum(three_gram_words[-1])
    last2_sum = 0
    if len_three_gram > 1:
        last2_sum = sum(three_gram_words[-2])

    return action_123_ratio, action_234_ratio, action_345_ratio, action_456_ratio, action_567_ratio, \
           action_678_ratio, action_789_ratio, last_sum, last2_sum


def build_action_order_features2(df, action_grouped):
    features = pd.DataFrame({'userid': df['userid']})

    # 2-gram 方式统计 5~9 先后出现的次数和最后一次出现的时间等统计特征
    print('two_gram_statistic')
    features['two_gram_statistic'] = features.apply(lambda row: two_gram_statistic(row['userid'], action_grouped), axis=1)
    # features['two_gram_action_12_ratio'] = features['two_gram_statistic'].map(lambda x: x[0])
    features['two_gram_action_23_ratio'] = features['two_gram_statistic'].map(lambda x: x[1])
    features['two_gram_action_34_ratio'] = features['two_gram_statistic'].map(lambda x: x[2])
    features['two_gram_action_45_ratio'] = features['two_gram_statistic'].map(lambda x: x[3])
    # features['two_gram_action_56_ratio'] = features['two_gram_statistic'].map(lambda x: x[4])
    # features['two_gram_action_67_ratio'] = features['two_gram_statistic'].map(lambda x: x[5])
    # features['two_gram_action_78_ratio'] = features['two_gram_statistic'].map(lambda x: x[6])
    features['two_gram_action_89_ratio'] = features['two_gram_statistic'].map(lambda x: x[7])
    features['two_gram_last_sum'] = features['two_gram_statistic'].map(lambda x: x[8])
    # features['two_gram_last2_sum'] = features['two_gram_statistic'].map(lambda x: x[9])
    del features['two_gram_statistic']
    # 3-gram 方式统计 5~9 先后出现的次数和最后一次出现的时间等统计特征
    features['three_gram_statistic'] = features.apply(lambda row: three_gram_statistic(row['userid'], action_grouped), axis=1)
    features['three_gram_action_123_ratio'] = features['three_gram_statistic'].map(lambda x: x[0])
    # features['three_gram_action_234_ratio'] = features['three_gram_statistic'].map(lambda x: x[1])
    # features['three_gram_action_345_ratio'] = features['three_gram_statistic'].map(lambda x: x[2])
    features['three_gram_action_456_ratio'] = features['three_gram_statistic'].map(lambda x: x[3])
    # features['three_gram_action_567_ratio'] = features['three_gram_statistic'].map(lambda x: x[4])
    # features['three_gram_action_678_ratio'] = features['three_gram_statistic'].map(lambda x: x[5])
    features['three_gram_action_789_ratio'] = features['three_gram_statistic'].map(lambda x: x[6])
    # features['three_gram_last_sum'] = features['three_gram_statistic'].map(lambda x: x[7])
    # features['three_gram_last2_sum'] = features['three_gram_statistic'].map(lambda x: x[8])
    del features['three_gram_statistic']

    return features


def generate_new_action(action, history):
    history['actionType'] = history['orderType'].map(lambda x: 10 if x == 0 else 11)
    history['actionTime'] = history['orderTime']
    action = pd.concat([action, history[['userid', 'actionType', 'actionTime']]], axis=0)
    action.sort_values(by='actionTime', inplace=True)
    action = action.reset_index(drop=True)
    return action


def main():
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')

    orderHistory_train = pd.read_csv(Configure.base_path + 'train/orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.base_path + 'test/orderHistory_test.csv', encoding='utf8')

    action_train = pd.read_csv(Configure.base_path + 'train/action_train.csv')
    action_test = pd.read_csv(Configure.base_path + 'test/action_test.csv')

    action_train = generate_new_action(action_train, orderHistory_train)
    action_test = generate_new_action(action_test, orderHistory_test)

    train_action_grouped = dict(list(action_train.groupby('userid')))
    test_action_grouped = dict(list(action_test.groupby('userid')))

    feature_name = 'action_order_features1'
    if not data_utils.is_feature_created(feature_name):
        print('build train action_order_features1')
        train_features = build_action_order_features1(train, train_action_grouped)
        print('build test action_order_features1')
        test_features = build_action_order_features1(test, test_action_grouped)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'action_order_features2'
    if not data_utils.is_feature_created(feature_name):
        print('build train action_order_features2')
        train_features = build_action_order_features2(train, train_action_grouped)
        print('build test action_order_features2')
        test_features = build_action_order_features2(test, test_action_grouped)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 将 order 订单作为一种新的 actiontype 挖掘特征 ==========")
    main()
