#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-25 下午9:18
"""
from __future__ import absolute_import, division, print_function

import hashlib
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
from conf.configure import Configure
from utils import data_utils


def compute_elaspe_time(time1, time2):
    """ 计算间隔时间 """
    elapse = (time1 - time2)
    elapse = elapse.total_seconds() / 60.0  # 计算间隔的分钟数，转换为float类型
    return elapse


def last_time_order_now_action_count(uid, history_grouped, action_grouped, flag):
    """ 最后一次 order 距离现在的 action 操作的次数 """
    a_df = action_grouped[uid]

    if flag == 0:
        sub_action_df = a_df
        if sub_action_df.shape[0] == 0:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        actionTypes = sub_action_df['actionType'].tolist()
        return len(actionTypes), actionTypes.count('browse_product'), actionTypes.count('browse_product2'), \
               actionTypes.count('browse_product3'), actionTypes.count('fillin_form5'), \
               actionTypes.count('fillin_form6'), actionTypes.count('fillin_form7'), actionTypes.count('open_app'), \
               actionTypes.count('pay_money'), actionTypes.count('submit_order')

    h_df = history_grouped[uid]
    if a_df.shape[0] == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        last_order_time = h_df.iloc[-1]['orderTime']
        sub_action_df = a_df[a_df['actionTime'] > last_order_time]
        if sub_action_df.shape[0] == 0:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        actionTypes = sub_action_df['actionType'].tolist()
        return len(actionTypes), actionTypes.count('browse_product'), actionTypes.count('browse_product2') \
            , actionTypes.count('browse_product3') , actionTypes.count('fillin_form5'), \
               actionTypes.count('fillin_form6'), actionTypes.count('fillin_form7'), actionTypes.count('open_app'), \
               actionTypes.count('pay_money'), actionTypes.count('submit_order')


def last_action_type_time_delta(uid, action_grouped, actiontype, last=1):
    """ 距离最近的 action type 的时间距离 """
    action_df = action_grouped[uid]
    if actiontype not in action_df['actionType'].values:
        return -1
    df = action_df[action_df['actionType'] == actiontype]
    if df.shape[0] < last:
        return -1
    return df.iloc[-last]['days_from_now']


def actiontype_timedelta_statistic(uid, action_grouped, actiontype):
    """ 点击 actiontype 的时间间隔统计特征 """
    action_df = action_grouped[uid]
    action_df = action_df[action_df['actionType'] == actiontype]
    if action_df.shape[0] < 3:
        return -1, -1, -1, -1

    action_times = action_df['actionTime'].values
    deltas = []
    for i in range(len(action_times) - 1):
        delta = (action_times[i+1] - action_times[i]) / np.timedelta64(1,'s')
        deltas.append(delta)

    return np.mean(deltas), np.max(deltas), np.min(deltas), np.std(deltas)


def build_action_history_features(df, action, history):
    features = pd.DataFrame({'userid': df['userid']})

    df_ids = history['userid'].unique()
    action_grouped = dict(list(action.groupby('userid')))
    history_grouped = dict(list(history.groupby('userid')))

    # 是否有交易历史
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in df_ids).astype(int)

    # action 表
    print('距离现在每个用户的 action 特征')
    # 最后一次 order 距离现在的 action 操作的次数
    features['last_time_order_now_action_info_count'] = features.apply(lambda row: last_time_order_now_action_count(row['userid'], history_grouped, action_grouped, row['has_history_flag']), axis=1)
    features['last_time_order_now_action_total_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[0])
    features['last_time_order_now_action_browse_product_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[1])
    features['last_time_order_now_action_browse_product2_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[2])
    features['last_time_order_now_action_browse_product3_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[3])
    features['last_time_order_now_action_fillin_form5_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[4])
    features['last_time_order_now_action_fillin_form6_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[5])
    features['last_time_order_now_action_fillin_form7_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[6])
    features['last_time_order_now_action_open_app_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[7])
    features['last_time_order_now_action_pay_money_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[8])
    features['last_time_order_now_action_submit_order_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[9])
    del features['last_time_order_now_action_info_count']
    # 是否有支付操作和提交订单操作
    features['last_time_order_now_has_paied_money'] = features['last_time_order_now_action_pay_money_count'].map(lambda x: int(x > 0))
    features['last_time_order_now_has_submited_order'] = features['last_time_order_now_action_submit_order_count'].map(lambda x: int(x > 0))

    print('距离最近的 action type 的时间距离')
    features['last_action_open_app_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'open_app'), axis=1)
    features['last_action_browse_product_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'browse_product'), axis=1)
    features['last_action_browse_product2_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'browse_product2'), axis=1)
    features['last_action_browse_product3_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'browse_product3'), axis=1)
    features['last_action_fillin_form5_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'fillin_form5'), axis=1)
    features['last_action_fillin_form6_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'fillin_form6'), axis=1)
    features['last_action_fillin_form7_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'fillin_form7'), axis=1)
    features['last_action_submit_order_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'submit_order'), axis=1)
    features['last_action_pay_money_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'pay_money'), axis=1)

    print('距离最近的倒数第二次 action type 的时间距离')
    features['last2_action_open_app_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'open_app', 2), axis=1)
    features['last2_action_browse_product_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'browse_product', 2), axis=1)
    features['last2_action_browse_product2_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'browse_product2', 2), axis=1)
    features['last2_action_browse_product3_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'browse_product3', 2), axis=1)
    features['last2_action_fillin_form5_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'fillin_form5', 2), axis=1)
    features['last2_action_fillin_form6_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'fillin_form6', 2), axis=1)
    features['last2_action_fillin_form7_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'fillin_form7', 2), axis=1)
    features['last2_action_submit_order_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'submit_order', 2), axis=1)
    features['last2_action_pay_money_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'pay_money', 2), axis=1)

    print('点击 actiontype 的时间间隔统计特征')
    features['actiontype_timedelta_statistic'] = features.apply(lambda row: actiontype_timedelta_statistic(row['userid'], action_grouped, 'open_app'), axis=1)
    features['open_app_mean_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[0])
    features['open_app_max_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[1])
    features['open_app_min_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[2])
    features['open_app_std_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[3])

    features['actiontype_timedelta_statistic'] = features.apply(lambda row: actiontype_timedelta_statistic(row['userid'], action_grouped, 'browse_product'), axis=1)
    features['browse_product_mean_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[0])
    features['browse_product_max_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[1])
    features['browse_product_min_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[2])
    features['browse_product_std_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[3])
    features['actiontype_timedelta_statistic'] = features.apply(lambda row: actiontype_timedelta_statistic(row['userid'], action_grouped, 'browse_product2'), axis=1)
    features['browse_product2_mean_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[0])
    features['browse_product2_max_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[1])
    features['browse_product2_min_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[2])
    features['browse_product2_std_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[3])
    features['actiontype_timedelta_statistic'] = features.apply(lambda row: actiontype_timedelta_statistic(row['userid'], action_grouped, 'browse_product3'), axis=1)
    features['browse_product3_mean_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[0])
    features['browse_product3_max_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[1])
    features['browse_product3_min_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[2])
    features['browse_product3_std_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[3])

    features['actiontype_timedelta_statistic'] = features.apply(lambda row: actiontype_timedelta_statistic(row['userid'], action_grouped, 'fillin_form5'), axis=1)
    features['fillin_form5_mean_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[0])
    features['fillin_form5_max_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[1])
    features['fillin_form5_min_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[2])
    features['fillin_form5_std_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[3])

    features['actiontype_timedelta_statistic'] = features.apply(lambda row: actiontype_timedelta_statistic(row['userid'], action_grouped, 'fillin_form6'), axis=1)
    features['fillin_form6_mean_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[0])
    features['fillin_form6_max_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[1])
    features['fillin_form6_min_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[2])
    features['fillin_form6_std_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[3])

    features['actiontype_timedelta_statistic'] = features.apply(lambda row: actiontype_timedelta_statistic(row['userid'], action_grouped, 'fillin_form7'), axis=1)
    features['fillin_form7_mean_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[0])
    features['fillin_form7_max_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[1])
    features['fillin_form7_min_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[2])
    features['fillin_form7_std_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[3])

    features['actiontype_timedelta_statistic'] = features.apply(lambda row: actiontype_timedelta_statistic(row['userid'], action_grouped, 'submit_order'), axis=1)
    features['submit_order_mean_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[0])
    features['submit_order_max_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[1])
    features['submit_order_min_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[2])
    features['submit_order_std_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[3])

    features['actiontype_timedelta_statistic'] = features.apply(lambda row: actiontype_timedelta_statistic(row['userid'], action_grouped, 'pay_money'), axis=1)
    features['pay_money_mean_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[0])
    features['pay_money_max_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[1])
    features['pay_money_min_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[2])
    features['pay_money_std_delta'] = features['actiontype_timedelta_statistic'].map(lambda x: x[3])
    del features['actiontype_timedelta_statistic']

    del features['has_history_flag']
    return features


def last_actiontype_action_count(uid, action_grouped, actiontype):
    """ 距离上一次 pay money 操作到现在 action 的总次数 """
    action_df = action_grouped[uid]
    df = action_df[action_df['actionType'] == actiontype]
    if df.shape[0] == 0:
        return -1

    last_actiontype_time = df['actionTime'].values[-1]
    df = action_df[action_df['actionTime'] > last_actiontype_time]
    return df.shape[0]


def last_target_actiontype_ratio(uid, action_grouped, target_action, actiontype):
    """ 距离上一次 pay money 操作到现在 actiontype 的比例 """
    action_df = action_grouped[uid]
    df = action_df[action_df['actionType'] == target_action]
    if df.shape[0] == 0:
        return -1

    last_actiontype_time = df['actionTime'].values[-1]
    df = action_df[action_df['actionTime'] > last_actiontype_time]

    return (df[df['actionType'] == actiontype].shape[0] + 1) / (df.shape[0] + 2) - 0.5


def build_action_history_features2(df, action, history):
    features = pd.DataFrame({'userid': df['userid']})

    action_grouped = dict(list(action.groupby('userid')))
    history_grouped = dict(list(history.groupby('userid')))

    print('距离上一次 action type 操作到现在的统计特征')
    features['last_pay_money_action_count'] = features.apply(lambda row: last_actiontype_action_count(row['userid'], action_grouped, 'pay_money'), axis=1)
    features['last_submit_order_action_count'] = features.apply(lambda row: last_actiontype_action_count(row['userid'], action_grouped, 'submit_order'), axis=1)
    features['last_fillin_form7_action_count'] = features.apply(lambda row: last_actiontype_action_count(row['userid'], action_grouped, 'fillin_form7'), axis=1)
    features['last_fillin_form6_action_count'] = features.apply(lambda row: last_actiontype_action_count(row['userid'], action_grouped, 'fillin_form6'), axis=1)
    features['last_fillin_form5_action_count'] = features.apply(lambda row: last_actiontype_action_count(row['userid'], action_grouped, 'fillin_form5'), axis=1)
    features['last_browse_product_action_count'] = features.apply(lambda row: last_actiontype_action_count(row['userid'], action_grouped, 'browse_product'), axis=1)
    features['last_browse_product2_action_count'] = features.apply(lambda row: last_actiontype_action_count(row['userid'], action_grouped, 'browse_product2'), axis=1)
    features['last_browse_product3_action_count'] = features.apply(lambda row: last_actiontype_action_count(row['userid'], action_grouped, 'browse_product3'), axis=1)
    features['last_open_app_action_count'] = features.apply(lambda row: last_actiontype_action_count(row['userid'], action_grouped, 'open_app'), axis=1)

    print('距离上一次 pay money 操作到现在 actiontype 的比例')
    features['last_pay_money_now_submit_order_ratio'] = features.apply(lambda row: last_target_actiontype_ratio(row['userid'], action_grouped, 'pay_money', 'submit_order'), axis=1)
    features['last_pay_money_now_fillin_form7_ratio'] = features.apply(lambda row: last_target_actiontype_ratio(row['userid'], action_grouped, 'pay_money', 'fillin_form7'), axis=1)
    features['last_pay_money_now_fillin_form6_ratio'] = features.apply(lambda row: last_target_actiontype_ratio(row['userid'], action_grouped, 'pay_money', 'fillin_form6'), axis=1)
    features['last_pay_money_now_fillin_form5_ratio'] = features.apply(lambda row: last_target_actiontype_ratio(row['userid'], action_grouped, 'pay_money', 'fillin_form5'), axis=1)
    features['last_pay_money_now_browse_product_ratio'] = features.apply(lambda row: last_target_actiontype_ratio(row['userid'], action_grouped, 'pay_money', 'browse_product'), axis=1)
    features['last_pay_money_now_browse_product2_ratio'] = features.apply(lambda row: last_target_actiontype_ratio(row['userid'], action_grouped, 'pay_money', 'browse_product2'), axis=1)
    features['last_pay_money_now_browse_product3_ratio'] = features.apply(lambda row: last_target_actiontype_ratio(row['userid'], action_grouped, 'pay_money', 'browse_product3'), axis=1)
    features['last_pay_money_now_open_app_ratio'] = features.apply(lambda row: last_target_actiontype_ratio(row['userid'], action_grouped, 'pay_money', 'open_app'), axis=1)

    return features


def gen_order_history_sequence(uid, history_grouped, has_history_flag):
    """ 用户订单历史结果构成的序列 """
    # 311 天的操作记录
    sequence = ['0'] * 311
    if has_history_flag == 0:
        return sequence

    df = history_grouped[uid]
    for i in df['days_from_now']:
        sequence[i] = str(df[df['days_from_now'] == i].shape[0])
    return sequence


def numerical_order_history_sequence(sequence):
    weights = [1.0 / np.exp(((day + 1.0) / len(sequence))) for day in range(len(sequence))]
    weights = weights / sum(weights)
    sequence = np.array([int(s) for s in sequence])
    score = np.dot(weights, sequence)
    return score


def getActionTimeSpan(df_action_of_userid, actiontypeA, actiontypeB, timethred=100):
    timespan_list = []
    i = 0
    while i < (len(df_action_of_userid)-1):
        if df_action_of_userid['actionType'].iat[i] == actiontypeA:
            timeA = df_action_of_userid['actionTime'].iat[i]
            for j in range(i+1, len(df_action_of_userid)):
                if df_action_of_userid['actionType'].iat[j] == actiontypeA:
                    timeA = df_action_of_userid['actionTime'].iat[j]
                if df_action_of_userid['actionType'].iat[j] == actiontypeB:
                    timeB = df_action_of_userid['actionTime'].iat[j]
                    timespan_list.append(timeB-timeA)
                    i = j
                    break
        i+=1
    return np.sum(np.array(timespan_list) <= timethred) / (np.sum(np.array(timespan_list)) + 1.0)


def get2ActionTimeSpanLast(df_action_of_userid, actiontypeA, actiontypeB):
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
                    break
        i+=1
    if len(timespan_list) > 0:
        return timespan_list[-1]
    else:
        return -1

def calc_seqentialratio(df_action_of_userid):
    i = 0
    pos_5 = -1
    result = 0
    df_len = len(df_action_of_userid)
    for i in range(0, df_len):
        if df_action_of_userid['actionType'].iat[i] == 5:
            pos_5 = i
    if pos_5 != -1:
        result += 1
        if pos_5+1 < df_len:
            if df_action_of_userid['actionType'].iat[pos_5+1] == 6:
                result += 1
                if pos_5+2 < df_len:
                    if df_action_of_userid['actionType'].iat[pos_5+2] == 7:
                        result += 1
                        if pos_5+3 < df_len:
                            if df_action_of_userid['actionType'].iat[pos_5+3] == 8:
                                result += 1
    return result


def getTagsFromActionByUserid(uid, action_grouped):
    df_action_of_userid = action_grouped[uid]
    sum_action = len(df_action_of_userid)  # 一个用户action的总次数
    actiontime_last_1_year = -1
    actiontime_last_1_month = -1
    action_last_1 = 0   # 倒数第1次actionType
    action_last_2 = 0   # 倒数第2次actionType
    action_last_3 = 0   # 倒数第3次actionType
    action_last_4 = 0   # 倒数第4次actionType
    action_last_5 = 0   # 倒数第5次actionType
    action_last_6 = 0   # 倒数第6次actionType
    action_last_7 = 0   # 倒数第7次actionType
    action_last_8 = 0   # 倒数第8次actionType
    action_last_9 = 0   # 倒数第9次actionType
    action_last_10 = 0  # 倒数第10次actionType
    action_last_11 = 0  # 倒数第11次actionType
    action_last_12 = 0  # 倒数第12次actionType
    action_last_13 = 0  # 倒数第13次actionType
    action_last_14 = 0  # 倒数第14次actionType
    action_last_15 = 0  # 倒数第15次actionType
    action_last_16 = 0  # 倒数第16次actionType
    action_last_17 = 0  # 倒数第17次actionType
    action_last_18 = 0  # 倒数第18次actionType
    action_last_19 = 0  # 倒数第19次actionType
    action_last_20 = 0  # 倒数第20次actionType
    #actiontime_mean = np.mean(df_action['actionTime'])
    actiontime_last_1 = 0   # 倒数第1次actionTime
    actiontime_last_2 = 0   # 倒数第2次actionTime
    actiontime_last_3 = 0   # 倒数第3次actionTime
    actiontime_last_4 = 0   # 倒数第4次actionTime
    actiontime_last_5 = 0   # 倒数第5次actionTime
    actiontime_last_6 = 0   # 倒数第6次actionTime
    actiontime_last_7 = 0   # 倒数第7次actionTime
    actiontime_last_8 = 0   # 倒数第8次actionTime
    actiontime_last_9 = 0   # 倒数第9次actionTime
    actiontime_last_10 = 0  # 倒数第10次actionTime
    actiontime_last_11 = 0  # 倒数第11次actionTime
    actiontime_last_12 = 0  # 倒数第12次actionTime
    actiontime_last_13 = 0  # 倒数第13次actionTime
    actiontime_last_14 = 0  # 倒数第14次actionTime
    actiontime_last_15 = 0  # 倒数第15次actionTime
    actiontime_last_16 = 0  # 倒数第16次actionTime
    actiontime_last_17 = 0  # 倒数第17次actionTime
    actiontime_last_18 = 0  # 倒数第18次actionTime
    actiontime_last_19 = 0  # 倒数第19次actionTime
    actiontime_last_20 = 0  # 倒数第20次actionTime
    actiontypeprop_1 = 0  # actionType1占比
    actiontypeprop_2 = 0  # actionType2占比
    actiontypeprop_3 = 0  # actionType3占比
    actiontypeprop_4 = 0  # actionType4占比
    actiontypeprop_5 = 0  # actionType5占比
    actiontypeprop_6 = 0  # actionType6占比
    actiontypeprop_7 = 0  # actionType7占比
    actiontypeprop_8 = 0  # actionType8占比
    actiontypeprop_9 = 0  # actionType9占比
    timespanthred = 100
    actiontimespancount_1_5 = 0  # actionType1-5的时间差小于timespanthred的数量
    actiontimespancount_5_6 = 0  # actionType5-6的时间差小于timespanthred的数量
    actiontimespancount_6_7 = 0  # actionType6-7的时间差小于timespanthred的数量
    actiontimespancount_7_8 = 0  # actionType7-8的时间差小于timespanthred的数量
    actiontimespancount_8_9 = 0  # actionType8-9的时间差小于timespanthred的数量
    actionratio_24_59 = 1.0      # actionType2-4与5-9之间的比值
    actiontype_lasttime_1 = 0    # actionType1最后一次出现的时间
    actiontype_lasttime_5 = 0    # actionType5最后一次出现的时间
    actiontype_lasttime_6 = 0    # actionType6最后一次出现的时间
    actiontype_lasttime_7 = 0    # actionType7最后一次出现的时间
    actiontype_lasttime_8 = 0    # actionType8最后一次出现的时间
    actiontype_lasttime_9 = 0    # actionType9最后一次出现的时间
    actiontype_lasttime_24 = 0   # actionType2-4最后一次出现的时间
    actiontimespanlast_1_5 = 0   # 最后一次actionType1与5之间的间隔
    actiontimespanlast_5_6 = 0   # 最后一次actionType5与6之间的间隔
    actiontimespanlast_6_7 = 0   # 最后一次actionType6与7之间的间隔
    actiontimespanlast_7_8 = 0   # 最后一次actionType7与8之间的间隔
    actiontimespanlast_5_7 = 0   # 最后一次actionType5与7之间的间隔
    actiontimespanlast_5_8 = 0   # 最后一次actionType5与8之间的间隔
    action59seqentialratio = 0      # actionType5-9的连续程度
    actiontypeproplast20_1 = 0  # 最后20个action中，actionType1占比
    actiontypeproplast20_2 = 0  # 最后20个action中，actionType2占比
    actiontypeproplast20_3 = 0  # 最后20个action中，actionType3占比
    actiontypeproplast20_4 = 0  # 最后20个action中，actionType4占比
    actiontypeproplast20_5 = 0  # 最后20个action中，actionType5占比
    actiontypeproplast20_6 = 0  # 最后20个action中，actionType6占比
    actiontypeproplast20_7 = 0  # 最后20个action中，actionType7占比
    actiontypeproplast20_8 = 0  # 最后20个action中，actionType8占比
    actiontypeproplast20_9 = 0  # 最后20个action中，actionType9占比
    actiontime_1 = 0            # 第一个actionTime（用户第一次使用app的时间）
    if sum_action >= 1:
        actiontime_1 = df_action_of_userid['actionTime'].iat[0]
        action_last_1 = df_action_of_userid['actionType'].iat[-1]
        actiontime_last_1 = df_action_of_userid['actionTime'].iat[-1]
        time_local = time.localtime(actiontime_last_1)
        actiontime_last_1_year = time_local.tm_year
        actiontime_last_1_month = time_local.tm_mon
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 1]) > 0:
            actiontype_lasttime_1 = df_action_of_userid[df_action_of_userid['actionType'] == 1].iloc[-1]['actionTime']
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 5]) > 0:
            actiontype_lasttime_5 = df_action_of_userid[df_action_of_userid['actionType'] == 5].iloc[-1]['actionTime']
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 6]) > 0:
            actiontype_lasttime_6 = df_action_of_userid[df_action_of_userid['actionType'] == 6].iloc[-1]['actionTime']
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 7]) > 0:
            actiontype_lasttime_7 = df_action_of_userid[df_action_of_userid['actionType'] == 7].iloc[-1]['actionTime']
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 8]) > 0:
            actiontype_lasttime_8 = df_action_of_userid[df_action_of_userid['actionType'] == 8].iloc[-1]['actionTime']
        if len(df_action_of_userid[df_action_of_userid['actionType'] == 9]) > 0:
            actiontype_lasttime_9 = df_action_of_userid[df_action_of_userid['actionType'] == 9].iloc[-1]['actionTime']
        if len(df_action_of_userid[(df_action_of_userid['actionType'] >= 2) & (df_action_of_userid['actionType'] <= 4)]) > 0:
            actiontype_lasttime_24 = df_action_of_userid[(df_action_of_userid['actionType'] >= 2) & (df_action_of_userid['actionType'] <= 4)].iloc[-1]['actionTime']
    if sum_action >= 2:
        action_last_2 = df_action_of_userid['actionType'].iat[-2]
        actiontime_last_2 = df_action_of_userid['actionTime'].iat[-2]
    if sum_action >= 3:
        action_last_3 = df_action_of_userid['actionType'].iat[-3]
        actiontime_last_3 = df_action_of_userid['actionTime'].iat[-3]
        actiontimespanlast_1_5 = get2ActionTimeSpanLast(df_action_of_userid, 1, 5)
        actiontimespanlast_5_6 = get2ActionTimeSpanLast(df_action_of_userid, 5, 6)
        actiontimespanlast_6_7 = get2ActionTimeSpanLast(df_action_of_userid, 6, 7)
        actiontimespanlast_7_8 = get2ActionTimeSpanLast(df_action_of_userid, 7, 8)
        actiontimespanlast_5_7 = get2ActionTimeSpanLast(df_action_of_userid, 5, 7)
        actiontimespanlast_5_8 = get2ActionTimeSpanLast(df_action_of_userid, 5, 8)
        action59seqentialratio = calc_seqentialratio(df_action_of_userid)
    if sum_action >= 4:
        action_last_4 = df_action_of_userid['actionType'].iat[-4]
        actiontime_last_4 = df_action_of_userid['actionTime'].iat[-4]
    if sum_action >= 5:
        action_last_5 = df_action_of_userid['actionType'].iat[-5]
        actiontime_last_5 = df_action_of_userid['actionTime'].iat[-5]
    if sum_action >= 6:
        action_last_6 = df_action_of_userid['actionType'].iat[-6]
        actiontime_last_6 = df_action_of_userid['actionTime'].iat[-6]
    if sum_action >= 7:
        action_last_7 = df_action_of_userid['actionType'].iat[-7]
        actiontime_last_7 = df_action_of_userid['actionTime'].iat[-7]
    if sum_action >= 8:
        action_last_8 = df_action_of_userid['actionType'].iat[-8]
        actiontime_last_8 = df_action_of_userid['actionTime'].iat[-8]
    if sum_action >= 9:
        action_last_9 = df_action_of_userid['actionType'].iat[-9]
        actiontime_last_9 = df_action_of_userid['actionTime'].iat[-9]
    if sum_action >= 10:
        action_last_10 = df_action_of_userid['actionType'].iat[-10]
        actiontime_last_10 = df_action_of_userid['actionTime'].iat[-10]
    if sum_action >= 11:
        action_last_11 = df_action_of_userid['actionType'].iat[-11]
        actiontime_last_11 = df_action_of_userid['actionTime'].iat[-11]
    if sum_action >= 12:
        action_last_12 = df_action_of_userid['actionType'].iat[-12]
        actiontime_last_12 = df_action_of_userid['actionTime'].iat[-12]
    if sum_action >= 13:
        action_last_13 = df_action_of_userid['actionType'].iat[-13]
        actiontime_last_13 = df_action_of_userid['actionTime'].iat[-13]
    if sum_action >= 14:
        action_last_14 = df_action_of_userid['actionType'].iat[-14]
        actiontime_last_14 = df_action_of_userid['actionTime'].iat[-14]
    if sum_action >= 15:
        action_last_15 = df_action_of_userid['actionType'].iat[-15]
        actiontime_last_15 = df_action_of_userid['actionTime'].iat[-15]
    if sum_action >= 16:
        action_last_16 = df_action_of_userid['actionType'].iat[-16]
        actiontime_last_16 = df_action_of_userid['actionTime'].iat[-16]
    if sum_action >= 17:
        action_last_17 = df_action_of_userid['actionType'].iat[-17]
        actiontime_last_17 = df_action_of_userid['actionTime'].iat[-17]
    if sum_action >= 18:
        action_last_18 = df_action_of_userid['actionType'].iat[-18]
        actiontime_last_18 = df_action_of_userid['actionTime'].iat[-18]
    if sum_action >= 19:
        action_last_19 = df_action_of_userid['actionType'].iat[-19]
        actiontime_last_19 = df_action_of_userid['actionTime'].iat[-19]
    if sum_action >= 20:
        action_last_20 = df_action_of_userid['actionType'].iat[-20]
        actiontime_last_20 = df_action_of_userid['actionTime'].iat[-20]
        actiontypeproplast20_1 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==1)
        actiontypeproplast20_2 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==2)
        actiontypeproplast20_3 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==3)
        actiontypeproplast20_4 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==4)
        actiontypeproplast20_5 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==5)
        actiontypeproplast20_6 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==6)
        actiontypeproplast20_7 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==7)
        actiontypeproplast20_8 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==8)
        actiontypeproplast20_9 = np.sum(df_action_of_userid.iloc[-20:]['actionType']==9)
    actiontypeprop_1 = np.sum(df_action_of_userid['actionType']==1) / (sum_action+1.0)
    actiontypeprop_2 = np.sum(df_action_of_userid['actionType']==2) / (sum_action+1.0)
    actiontypeprop_3 = np.sum(df_action_of_userid['actionType']==3) / (sum_action+1.0)
    actiontypeprop_4 = np.sum(df_action_of_userid['actionType']==4) / (sum_action+1.0)
    actiontypeprop_5 = np.sum(df_action_of_userid['actionType']==5) / (sum_action+1.0)
    actiontypeprop_6 = np.sum(df_action_of_userid['actionType']==6) / (sum_action+1.0)
    actiontypeprop_7 = np.sum(df_action_of_userid['actionType']==7) / (sum_action+1.0)
    actiontypeprop_8 = np.sum(df_action_of_userid['actionType']==8) / (sum_action+1.0)
    actiontypeprop_9 = np.sum(df_action_of_userid['actionType']==9) / (sum_action+1.0)
    actiontimespancount_1_5 = getActionTimeSpan(df_action_of_userid, 1, 5, timespanthred)
    actiontimespancount_5_6 = getActionTimeSpan(df_action_of_userid, 5, 6, timespanthred)
    actiontimespancount_6_7 = getActionTimeSpan(df_action_of_userid, 6, 7, timespanthred)
    actiontimespancount_7_8 = getActionTimeSpan(df_action_of_userid, 5, 8, timespanthred)
    actiontimespancount_8_9 = getActionTimeSpan(df_action_of_userid, 4, 9, timespanthred)
    sum_action_24 = np.sum((df_action_of_userid['actionType'] >= 2) & (df_action_of_userid['actionType'] <= 4))
    sum_action_59 = np.sum((df_action_of_userid['actionType'] >= 5) & (df_action_of_userid['actionType'] <= 9))
    actionratio_24_59 = (sum_action_24 + 1.0) / (sum_action_59 + 1.0)
    return actiontime_last_1_year, actiontime_last_1_month, action_last_1, action_last_2, action_last_3, action_last_4, \
           action_last_5, action_last_6, action_last_7, action_last_8, action_last_9, action_last_10, action_last_11, \
           action_last_12, action_last_13, action_last_14, action_last_15, action_last_16, action_last_17, action_last_18, \
           action_last_19, action_last_20, actiontime_last_1, actiontime_last_2, actiontime_last_3, actiontime_last_4, \
           actiontime_last_5, actiontime_last_6, actiontime_last_7, actiontime_last_8, actiontime_last_9, actiontime_last_10, \
           actiontime_last_11, actiontime_last_12, actiontime_last_13, actiontime_last_14, actiontime_last_15, actiontime_last_16, \
           actiontime_last_17, actiontime_last_18, actiontime_last_19, actiontime_last_20, actiontypeprop_1, actiontypeprop_2, \
           actiontypeprop_3, actiontypeprop_4, actiontypeprop_5, actiontypeprop_6, actiontypeprop_7, actiontypeprop_8, actiontypeprop_9, \
           actiontimespancount_1_5, actiontimespancount_5_6, actiontimespancount_6_7, actiontimespancount_7_8, actiontimespancount_8_9, \
           actionratio_24_59, actiontype_lasttime_1, actiontype_lasttime_5, actiontype_lasttime_6, actiontype_lasttime_7, actiontype_lasttime_8, \
           actiontype_lasttime_9, actiontype_lasttime_24, actiontimespanlast_1_5, actiontimespanlast_5_6, actiontimespanlast_6_7, \
           actiontimespanlast_7_8, actiontimespanlast_5_7, actiontimespanlast_5_8, action59seqentialratio, actiontypeproplast20_1, \
           actiontypeproplast20_2, actiontypeproplast20_3, actiontypeproplast20_4, actiontypeproplast20_5, actiontypeproplast20_6, \
           actiontypeproplast20_7, actiontypeproplast20_8, actiontypeproplast20_9, actiontime_1


def build_action_history_features3(df, action, history):
    features = pd.DataFrame({'userid': df['userid']})

    action_grouped = dict(list(action.groupby('userid')))

    features['statistic'] = features.apply(lambda row: getTagsFromActionByUserid(row['userid'], action_grouped), axis=1)
    features['actiontime_last_1_year'] = features['statistic'].map(lambda x: x[0])
    features['actiontime_last_1_month'] = features['statistic'].map(lambda x: x[1])

    features['action_last_1'] = features['statistic'].map(lambda x: x[2])
    features['action_last_2'] = features['statistic'].map(lambda x: x[3])
    features['action_last_3'] = features['statistic'].map(lambda x: x[4])
    features['action_last_4'] = features['statistic'].map(lambda x: x[5])
    features['action_last_5'] = features['statistic'].map(lambda x: x[6])
    features['action_last_6'] = features['statistic'].map(lambda x: x[7])
    features['action_last_7'] = features['statistic'].map(lambda x: x[8])
    features['action_last_8'] = features['statistic'].map(lambda x: x[9])
    features['action_last_9'] = features['statistic'].map(lambda x: x[10])
    features['action_last_10'] = features['statistic'].map(lambda x: x[11])
    features['action_last_11'] = features['statistic'].map(lambda x: x[12])
    features['action_last_12'] = features['statistic'].map(lambda x: x[13])
    features['action_last_13'] = features['statistic'].map(lambda x: x[14])
    features['action_last_14'] = features['statistic'].map(lambda x: x[15])
    features['action_last_15'] = features['statistic'].map(lambda x: x[16])
    features['action_last_16'] = features['statistic'].map(lambda x: x[17])
    features['action_last_17'] = features['statistic'].map(lambda x: x[18])
    features['action_last_18'] = features['statistic'].map(lambda x: x[19])
    features['action_last_19'] = features['statistic'].map(lambda x: x[20])
    features['action_last_20'] = features['statistic'].map(lambda x: x[21])

    features['actiontime_last_1'] = features['statistic'].map(lambda x: x[22])
    features['actiontime_last_2'] = features['statistic'].map(lambda x: x[23])
    features['actiontime_last_3'] = features['statistic'].map(lambda x: x[24])
    features['actiontime_last_4'] = features['statistic'].map(lambda x: x[25])
    features['actiontime_last_5'] = features['statistic'].map(lambda x: x[26])
    features['actiontime_last_6'] = features['statistic'].map(lambda x: x[27])
    features['actiontime_last_7'] = features['statistic'].map(lambda x: x[28])
    features['actiontime_last_8'] = features['statistic'].map(lambda x: x[29])
    features['actiontime_last_9'] = features['statistic'].map(lambda x: x[30])
    features['actiontime_last_10'] = features['statistic'].map(lambda x: x[31])
    features['actiontime_last_11'] = features['statistic'].map(lambda x: x[32])
    features['actiontime_last_12'] = features['statistic'].map(lambda x: x[33])
    features['actiontime_last_13'] = features['statistic'].map(lambda x: x[34])
    features['actiontime_last_14'] = features['statistic'].map(lambda x: x[35])
    features['actiontime_last_15'] = features['statistic'].map(lambda x: x[36])
    features['actiontime_last_16'] = features['statistic'].map(lambda x: x[37])
    features['actiontime_last_17'] = features['statistic'].map(lambda x: x[38])
    features['actiontime_last_18'] = features['statistic'].map(lambda x: x[39])
    features['actiontime_last_19'] = features['statistic'].map(lambda x: x[40])
    features['actiontime_last_20'] = features['statistic'].map(lambda x: x[41])

    features['actiontypeprop_1'] = features['statistic'].map(lambda x: x[42])
    features['actiontypeprop_2'] = features['statistic'].map(lambda x: x[43])
    features['actiontypeprop_3'] = features['statistic'].map(lambda x: x[44])
    features['actiontypeprop_4'] = features['statistic'].map(lambda x: x[45])
    features['actiontypeprop_5'] = features['statistic'].map(lambda x: x[46])
    features['actiontypeprop_6'] = features['statistic'].map(lambda x: x[47])
    features['actiontypeprop_7'] = features['statistic'].map(lambda x: x[48])
    features['actiontypeprop_8'] = features['statistic'].map(lambda x: x[49])
    features['actiontypeprop_9'] = features['statistic'].map(lambda x: x[50])

    features['actiontimespancount_1_5'] = features['statistic'].map(lambda x: x[51])
    features['actiontimespancount_5_6'] = features['statistic'].map(lambda x: x[52])
    features['actiontimespancount_6_7'] = features['statistic'].map(lambda x: x[53])
    features['actiontimespancount_7_8'] = features['statistic'].map(lambda x: x[54])
    features['actiontimespancount_8_9'] = features['statistic'].map(lambda x: x[55])
    features['actionratio_24_59'] = features['statistic'].map(lambda x: x[56])

    features['actiontype_lasttime_1'] = features['statistic'].map(lambda x: x[57])
    features['actiontype_lasttime_5'] = features['statistic'].map(lambda x: x[58])
    features['actiontype_lasttime_6'] = features['statistic'].map(lambda x: x[59])
    features['actiontype_lasttime_7'] = features['statistic'].map(lambda x: x[60])
    features['actiontype_lasttime_8'] = features['statistic'].map(lambda x: x[61])
    features['actiontype_lasttime_9'] = features['statistic'].map(lambda x: x[62])
    features['actiontype_lasttime_24'] = features['statistic'].map(lambda x: x[63])

    features['actiontimespanlast_1_5'] = features['statistic'].map(lambda x: x[64])
    features['actiontimespanlast_5_6'] = features['statistic'].map(lambda x: x[65])
    features['actiontimespanlast_6_7'] = features['statistic'].map(lambda x: x[66])
    features['actiontimespanlast_7_8'] = features['statistic'].map(lambda x: x[67])
    features['actiontimespanlast_5_7'] = features['statistic'].map(lambda x: x[68])
    features['actiontimespanlast_5_8'] = features['statistic'].map(lambda x: x[69])

    features['action59seqentialratio'] = features['statistic'].map(lambda x: x[70])

    features['actiontypeproplast20_1'] = features['statistic'].map(lambda x: x[71])
    features['actiontypeproplast20_2'] = features['statistic'].map(lambda x: x[72])
    features['actiontypeproplast20_3'] = features['statistic'].map(lambda x: x[73])
    features['actiontypeproplast20_4'] = features['statistic'].map(lambda x: x[74])
    features['actiontypeproplast20_5'] = features['statistic'].map(lambda x: x[75])
    features['actiontypeproplast20_6'] = features['statistic'].map(lambda x: x[76])
    features['actiontypeproplast20_7'] = features['statistic'].map(lambda x: x[77])
    features['actiontypeproplast20_8'] = features['statistic'].map(lambda x: x[78])
    features['actiontypeproplast20_9'] = features['statistic'].map(lambda x: x[79])
    features['actiontime_1'] = features['statistic'].map(lambda x: x[80])

    del features['statistic']
    return features


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
        if len(timespan_list) >= 3:
            return np.min(timespan_list), np.max(timespan_list), np.mean(timespan_list), np.std(timespan_list), timespan_list[-3], timespan_list[-2]
        elif len(timespan_list) == 2:
            return np.min(timespan_list), np.max(timespan_list), np.mean(timespan_list), np.std(timespan_list), -999, timespan_list[-2]
        else:
            return np.min(timespan_list), np.max(timespan_list), np.mean(timespan_list), np.std(timespan_list), -999, -999
    else:
        return -999, -999, -999, -999, -999, -999


def build_action_history_features4(df, action, history):
    features = pd.DataFrame({'userid': df['userid']})
    action_grouped = dict(list(action.groupby('userid')))

    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 4, 5), axis=1)
    features['action_type_45_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_45_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    features['action_type_45_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    features['action_type_45_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    features['action_type_45_time_delta_last2'] = features['diff_action_type_time_delta'].map(lambda x: x[4])
    features['action_type_45_time_delta_last3'] = features['diff_action_type_time_delta'].map(lambda x: x[5])
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 5, 6), axis=1)
    features['action_type_56_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_56_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    features['action_type_56_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    features['action_type_56_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    features['action_type_56_time_delta_last2'] = features['diff_action_type_time_delta'].map(lambda x: x[4])
    features['action_type_56_time_delta_last3'] = features['diff_action_type_time_delta'].map(lambda x: x[5])
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 6, 7), axis=1)
    features['action_type_67_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_67_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    features['action_type_67_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    features['action_type_67_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    # features['action_type_67_time_delta_last2'] = features['diff_action_type_time_delta'].map(lambda x: x[4])
    # features['action_type_67_time_delta_last3'] = features['diff_action_type_time_delta'].map(lambda x: x[5])

    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 5, 8), axis=1)
    features['action_type_58_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_58_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    features['action_type_58_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    features['action_type_58_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 6, 8), axis=1)
    features['action_type_68_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_68_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    features['action_type_68_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    features['action_type_68_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])

    del features['diff_action_type_time_delta']
    return features


def main():

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

    action_train['actionTime'] = pd.to_datetime(action_train['actionTime'])
    action_test['actionTime'] = pd.to_datetime(action_test['actionTime'])
    orderHistory_train['orderTime'] = pd.to_datetime(orderHistory_train['orderTime'])
    orderHistory_test['orderTime'] = pd.to_datetime(orderHistory_test['orderTime'])

    action_train.sort_values(by='actionTime', inplace=True)
    action_test.sort_values(by='actionTime', inplace=True)
    orderHistory_train.sort_values(by='orderTime', inplace=True)
    orderHistory_test.sort_values(by='orderTime', inplace=True)

    feature_name = 'action_history_features'
    if not data_utils.is_feature_created(feature_name):
        print('build train action history features')
        train_features = build_action_history_features(train, action_train, orderHistory_train)
        print('build test action history features')
        test_features = build_action_history_features(test, action_test, orderHistory_test)

        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'action_history_features2'
    if not data_utils.is_feature_created(feature_name):
        print('build train action history features2')
        train_features = build_action_history_features2(train, action_train, orderHistory_train)
        print('build test action history features2')
        test_features = build_action_history_features2(test, action_test, orderHistory_test)

        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    action_train = pd.read_csv(Configure.base_path + 'train/action_train.csv')
    action_test = pd.read_csv(Configure.base_path + 'test/action_test.csv')

    feature_name = 'action_history_features3'
    if not data_utils.is_feature_created(feature_name):
        print('build train action history features3')
        train_features = build_action_history_features3(train, action_train, orderHistory_train)
        print('build test action history features3')
        test_features = build_action_history_features3(test, action_test, orderHistory_test)

        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'action_history_features4'
    # if not data_utils.is_feature_created(feature_name):
    print('build train action history features4')
    train_features = build_action_history_features4(train, action_train, orderHistory_train)
    print('build test action history features4')
    test_features = build_action_history_features4(test, action_test, orderHistory_test)

    print('save ', feature_name)
    data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 结合 action、 history 和 comment 提取历史特征 ==========")
    main()
