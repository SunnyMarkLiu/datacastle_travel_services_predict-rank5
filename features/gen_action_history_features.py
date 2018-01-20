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

import time
import datetime
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from conf.configure import Configure
from utils import data_utils

default_start_order_time = int(time.mktime(datetime.datetime(2016, 11, 01).timetuple()))
default_end_order_time = int(time.mktime(datetime.datetime(2017, 9, 12).timetuple()))

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

    print('getTagsFromActionByUserid...')
    features['statistic'] = features.apply(lambda row: getTagsFromActionByUserid(row['userid'], action_grouped), axis=1)
    print('split statistic')
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


def last_time_x_actiontype(uid, action_grouped, last):
    """ 最后几次操作的时间乘以操作类型 """
    df = action_grouped[uid]
    if df.shape[0] >= -last:
        return df['actionTime'].iat[last] * df['actionType'].iat[last]
    else:
        return -1

def last_paymoney_orderhistory_statistic(uid, action_grouped, history_grouped, flag):
    action_df = action_grouped[uid]
    if flag == 0:
        # 查找 action为 paymoney 记录
        df = action_df[action_df['actionType'] == 9]
        if df.shape[0] == 0:
            last_order_time = action_df['actionTime'].iat[0]
        else:
            last_order_time = df['actionTime'].iat[-1]
    else:
        # 查找交易记录
        history_df = history_grouped[uid]
        last_order_time = history_df['orderTime'].iat[-1]

    df = action_df[action_df['actionTime'] > last_order_time]
    last_order_sum_all = df.shape[0]
    last_order_actiontimespanlast_1_5 = get2ActionTimeSpanLast(df, 1, 5)
    last_order_actiontimespanlast_5_6 = get2ActionTimeSpanLast(df, 5, 6)
    last_order_actiontimespanlast_6_7 = get2ActionTimeSpanLast(df, 6, 7)
    last_order_actiontimespanlast_7_8 = get2ActionTimeSpanLast(df, 7, 8)
    last_order_actiontimespanlast_5_7 = get2ActionTimeSpanLast(df, 5, 7)
    last_order_actiontimespanlast_5_8 = get2ActionTimeSpanLast(df, 5, 8)
    last_order_action59seqentialratio = calc_seqentialratio(df)

    timespanthred = 100
    last_order_actiontimespancount_1_5 = getActionTimeSpan(df, 1, 5, timespanthred)
    last_order_actiontimespancount_5_6 = getActionTimeSpan(df, 5, 6, timespanthred)
    last_order_actiontimespancount_5_7 = getActionTimeSpan(df, 5, 7, timespanthred)
    last_order_actiontimespancount_5_8 = getActionTimeSpan(df, 5, 8, timespanthred)
    last_order_actiontimespancount_6_7 = getActionTimeSpan(df, 6, 7, timespanthred)
    last_order_actiontimespancount_6_8 = getActionTimeSpan(df, 6, 8, timespanthred)

    return last_order_sum_all, last_order_actiontimespanlast_1_5, last_order_actiontimespanlast_5_6, \
           last_order_actiontimespanlast_6_7, last_order_actiontimespanlast_7_8, last_order_actiontimespanlast_5_7, \
           last_order_actiontimespanlast_5_8, last_order_action59seqentialratio, last_order_actiontimespancount_1_5, \
           last_order_actiontimespancount_5_6, last_order_actiontimespancount_5_7, last_order_actiontimespancount_5_8, \
           last_order_actiontimespancount_6_7, last_order_actiontimespancount_6_8


def calc_action_score(uid, action_grouped):
    action_df = action_grouped[uid]
    user_score = sum(action_df['time_weight'] * action_df['action_weight'])
    return user_score


def build_action_history_features5(df, action, history):
    features = pd.DataFrame({'userid': df['userid']})
    df_ids = history['userid'].unique()
    action_grouped = dict(list(action.groupby('userid')))
    history['orderTime'] = history.orderTime.values.astype(np.int64) // 10 ** 9
    history_grouped = dict(list(history.groupby('userid')))

    # # 是否有交易历史
    # features['has_history_flag'] = features['userid'].map(lambda uid: uid in df_ids).astype(int)

    features['last_-1_x_actiontype'] = features.apply(lambda row: last_time_x_actiontype(row['userid'], action_grouped, -1), axis=1)
    features['last_-2_x_actiontype'] = features.apply(lambda row: last_time_x_actiontype(row['userid'], action_grouped, -2), axis=1)
    features['last_-3_x_actiontype'] = features.apply(lambda row: last_time_x_actiontype(row['userid'], action_grouped, -3), axis=1)
    features['last_time_x_actiontypr_mean'] = np.mean(features[['last_-1_x_actiontype', 'last_-2_x_actiontype', 'last_-3_x_actiontype']], axis=1)
    del features['last_-2_x_actiontype']
    del features['last_-3_x_actiontype']

    # print('距离上一次 actiontype 为 pay money 或有交易历史开始到现在的统计信息')
    # features['last_paymoney_orderhistory_statistic'] = features.apply(
    #     lambda row: last_paymoney_orderhistory_statistic(row['userid'], action_grouped, history_grouped, row['has_history_flag']),
    #     axis=1
    # )
    # features['last_order_sum_all'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[0])
    # features['last_order_actiontimespanlast_1_5'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[1])
    # features['last_order_actiontimespanlast_5_6'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[2])
    # features['last_order_actiontimespanlast_6_7'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[3])
    # features['last_order_actiontimespanlast_7_8'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[4])
    # features['last_order_actiontimespanlast_5_7'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[5])
    # features['last_order_actiontimespanlast_5_8'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[6])
    # features['last_order_action59seqentialratio'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[7])
    # features['last_order_actiontimespancount_1_5'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[8])
    # features['last_order_actiontimespancount_5_6'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[9])
    # features['last_order_actiontimespancount_5_7'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[10])
    # features['last_order_actiontimespancount_5_8'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[11])
    # features['last_order_actiontimespancount_6_7'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[12])
    # features['last_order_actiontimespancount_6_8'] = features['last_paymoney_orderhistory_statistic'].map(lambda x: x[13])
    # del features['last_paymoney_orderhistory_statistic']
    #
    # print('action type 变化求出 score')
    # end_time = action['actionTime'].max()
    # actionType = pd.get_dummies(action['actionType'], prefix='actionType')
    # tr_x = pd.concat([action, actionType], axis=1, join='inner')  # axis=1 是行
    # del tr_x['actionType']
    # del tr_x['actionTime']
    # tr_x = tr_x.groupby('userid', as_index=False).sum()
    # vals = []
    # for i in range(1, 10):
    #     vals.append(tr_x['actionType_%s' % i].sum())
    # vals = list(
    #     map(lambda x: round(-math.log((1.0 * (x - min(vals) + 100) / (max(vals) - min(vals) + 100 * len(vals)))), 4),
    #         vals))
    # acttype2weight = {(idx + 1): weight for idx, weight in enumerate(vals)}
    # action['time_weight'] = action['actionTime'].apply(lambda x: 0.5 ** int((end_time - x) / (30 * 24 * 3600)))
    # action['action_weight'] = action['actionType'].apply(lambda x: acttype2weight[x])
    # action_grouped = dict(list(action.groupby('userid')))
    # print('calc action score')
    # features['action_score'] = features.apply(lambda row: calc_action_score(row['userid'], action_grouped), axis=1)

    # del features['has_history_flag']
    return features


def build_action_history_features6(features3):
    features = pd.DataFrame({'userid': features3['userid']})
    features['timespan_action1tolast'] = features3['actiontime_last_1'] - features3['actiontype_lasttime_1']
    features['timespan_action5tolast'] = features3['actiontime_last_1'] - features3['actiontype_lasttime_5']
    features['timespan_action6tolast'] = features3['actiontime_last_1'] - features3['actiontype_lasttime_6']
    features['timespan_action7tolast'] = features3['actiontime_last_1'] - features3['actiontype_lasttime_7']
    features['timespan_action8tolast'] = features3['actiontime_last_1'] - features3['actiontype_lasttime_8']
    features['timespan_action9tolast'] = features3['actiontime_last_1'] - features3['actiontype_lasttime_9']
    features['timespan_action24tolast'] = features3['actiontime_last_1'] - features3['actiontype_lasttime_24']
    features['timespan_last_1'] = (features3['actiontime_last_1'] - features3['actiontime_last_2'])
    features['timespan_last_2'] = (features3['actiontime_last_2'] - features3['actiontime_last_3'])
    features['timespan_last_3'] = (features3['actiontime_last_3'] - features3['actiontime_last_4'])
    features['timespan_last_4'] = (features3['actiontime_last_4'] - features3['actiontime_last_5'])
    features['timespan_last_5'] = (features3['actiontime_last_5'] - features3['actiontime_last_6'])
    features['timespan_last_6'] = (features3['actiontime_last_6'] - features3['actiontime_last_7'])
    features['timespan_last_7'] = (features3['actiontime_last_7'] - features3['actiontime_last_8'])
    features['timespan_last_8'] = (features3['actiontime_last_8'] - features3['actiontime_last_9'])
    features['timespan_last_9'] = (features3['actiontime_last_9'] - features3['actiontime_last_10'])
    features['timespanmean_last_3'] = np.mean(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3']],
                                              axis=1)
    features['timespanmin_last_3'] = np.min(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3']], axis=1)
    features['timespanmax_last_3'] = np.max(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3']], axis=1)
    features['timespanstd_last_3'] = np.std(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3']], axis=1)
    features['timespanmean_last_4'] = np.mean(
        features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4']], axis=1)
    features['timespanmin_last_4'] = np.min(
        features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4']], axis=1)
    features['timespanmax_last_4'] = np.max(
        features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4']], axis=1)
    features['timespanstd_last_4'] = np.std(
        features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4']], axis=1)
    features['timespanmean_last_5'] = np.mean(
        features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5']],
        axis=1)
    features['timespanmin_last_5'] = np.min(
        features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5']],
        axis=1)
    features['timespanmax_last_5'] = np.max(
        features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5']],
        axis=1)
    features['timespanstd_last_5'] = np.std(
        features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3', 'timespan_last_4', 'timespan_last_5']],
        axis=1)
    features['timespanmean_last_6'] = np.mean(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                        'timespan_last_4', 'timespan_last_5', 'timespan_last_6']],
                                              axis=1)
    features['timespanmin_last_6'] = np.min(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6']], axis=1)
    features['timespanmax_last_6'] = np.max(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6']], axis=1)
    features['timespanstd_last_6'] = np.std(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6']], axis=1)
    features['timespanmean_last_7'] = np.mean(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                        'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                        'timespan_last_7']], axis=1)
    features['timespanmin_last_7'] = np.min(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                      'timespan_last_7']], axis=1)
    features['timespanmax_last_7'] = np.max(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                      'timespan_last_7']], axis=1)
    features['timespanstd_last_7'] = np.std(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                      'timespan_last_7']], axis=1)
    features['timespanmean_last_8'] = np.mean(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                        'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                        'timespan_last_7', 'timespan_last_8']], axis=1)
    features['timespanmin_last_8'] = np.min(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                      'timespan_last_7', 'timespan_last_8']], axis=1)
    features['timespanmax_last_8'] = np.max(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                      'timespan_last_7', 'timespan_last_8']], axis=1)
    features['timespanstd_last_8'] = np.std(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                      'timespan_last_7', 'timespan_last_8']], axis=1)
    features['timespanmean_last_9'] = np.mean(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                        'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                        'timespan_last_7', 'timespan_last_8', 'timespan_last_9']],
                                              axis=1)
    features['timespanmin_last_9'] = np.min(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                      'timespan_last_7', 'timespan_last_8', 'timespan_last_9']], axis=1)
    features['timespanmax_last_9'] = np.max(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                      'timespan_last_7', 'timespan_last_8', 'timespan_last_9']], axis=1)
    features['timespanstd_last_9'] = np.std(features[['timespan_last_1', 'timespan_last_2', 'timespan_last_3',
                                                      'timespan_last_4', 'timespan_last_5', 'timespan_last_6',
                                                      'timespan_last_7', 'timespan_last_8', 'timespan_last_9']], axis=1)
    features['timespan_total'] = features3['actiontime_last_1'] - features3['actiontime_1']

    features['actiontypeproplast20_mean'] = np.mean(features3[['actiontypeproplast20_1', 'actiontypeproplast20_2', 'actiontypeproplast20_3',
                                                              'actiontypeproplast20_4', 'actiontypeproplast20_5', 'actiontypeproplast20_6',
                                                              'actiontypeproplast20_7', 'actiontypeproplast20_8', 'actiontypeproplast20_9']], axis=1)
    features['actiontypeproplast20_std'] = np.std(features3[['actiontypeproplast20_1', 'actiontypeproplast20_2', 'actiontypeproplast20_3',
                                                              'actiontypeproplast20_4', 'actiontypeproplast20_5', 'actiontypeproplast20_6',
                                                              'actiontypeproplast20_7', 'actiontypeproplast20_8', 'actiontypeproplast20_9']], axis=1)
    features['actiontime_lasttime_mean'] =  np.mean(features3[['actiontype_lasttime_1', 'actiontype_lasttime_5', 'actiontype_lasttime_6',
                                                          'actiontype_lasttime_7', 'actiontype_lasttime_8', 'actiontype_lasttime_9']], axis=1)
    features['actiontime_lasttime_std'] =  np.std(features3[['actiontype_lasttime_1', 'actiontype_lasttime_5', 'actiontype_lasttime_6',
                                                          'actiontype_lasttime_7', 'actiontype_lasttime_8', 'actiontype_lasttime_9']], axis=1)
    features['actiontypeprop_mean'] = np.mean(features3[['actiontypeprop_1', 'actiontypeprop_2', 'actiontypeprop_3',
                                                        'actiontypeprop_4', 'actiontypeprop_5', 'actiontypeprop_6',
                                                        'actiontypeprop_7', 'actiontypeprop_8', 'actiontypeprop_9']], axis=1)
    features['actiontypeprop_std'] = np.std(features3[['actiontypeprop_1', 'actiontypeprop_2', 'actiontypeprop_3',
                                                        'actiontypeprop_4', 'actiontypeprop_5', 'actiontypeprop_6',
                                                        'actiontypeprop_7', 'actiontypeprop_8', 'actiontypeprop_9']], axis=1)

    return features


def several_days_had_action(uid, action_grouped, days_gap):
    action_df = action_grouped[uid]

    days = 1
    last_day = 0
    action_days_from_nows = action_df['days_from_now'].values
    last_action_day = action_days_from_nows[-1]

    for i in range(1, days_gap+1):
        days += sum((action_df['days_from_now'] == last_action_day).astype(int))
        last_action_day += 1

    return days


def order_history_delta_statistic(uid, history_grouped, flag):
    if flag == 0:
        return -1, -1, -1, -1

    df = history_grouped[uid]
    order_times = df['orderTime'].values.tolist()
    time_deltas = []
    if len(order_times) == 1:
        return -1, -1, -1, -1
    for i in range(len(order_times) - 1):
        time_deltas.append(order_times[i + 1] - order_times[i])
    return np.mean(time_deltas), np.max(time_deltas), np.min(time_deltas), time_deltas[-1]


def last_order_history_time(uid, history_grouped, flag):
    if flag == 0:
        return 0
    df = history_grouped[uid]
    return df['orderTime'].values.tolist()[-1]


def last_month_order_now_action_count(uid, action_grouped):
    """ 最近一个月距离现在的 action 操作的次数 """
    a_df = action_grouped[uid]

    sub_action_df = a_df[a_df['days_from_now'] < 30]
    if sub_action_df.shape[0] == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    actionTypes = sub_action_df['actionType'].tolist()
    return len(actionTypes), actionTypes.count('browse_product'), actionTypes.count('browse_product2') \
        , actionTypes.count('browse_product3') , actionTypes.count('fillin_form5'), \
           actionTypes.count('fillin_form6'), actionTypes.count('fillin_form7'), actionTypes.count('open_app'), \
           actionTypes.count('pay_money'), actionTypes.count('submit_order')


def build_action_history_features7(df, action, history):
    features = pd.DataFrame({'userid': df['userid']})
    action_grouped = dict(list(action.groupby('userid')))

    # # 连续操作的天数(如两天内都有操作 app)
    # features['several_1days_had_action'] = features.apply(lambda row: several_days_had_action(row['userid'], action_grouped, 1), axis=1)
    # features['several_1days_had_action_lg_150'] = features['several_1days_had_action'].map(lambda x: int(x > 150))
    # features['several_1days_had_action'] = np.log1p(features['several_1days_had_action'])
    #
    # features['several_2days_had_action'] = features.apply(lambda row: several_days_had_action(row['userid'], action_grouped, 2), axis=1)
    # # features['several_2days_had_action_lg_220'] = features['several_2days_had_action'].map(lambda x: int(x > 220))
    # features['several_2days_had_action'] = np.log1p(features['several_2days_had_action'])


    df_ids = history['userid'].unique()
    # history['orderTime'] = history.orderTime.values.astype(np.int64) // 10 ** 9
    history_grouped = dict(list(history.groupby('userid')))

    # # 是否有交易历史
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in df_ids).astype(int)

    # 上一次交易到现在的时间差和交易的平均时间差的统计特征
    features['order_history_delta_statistic'] = features.apply(lambda row: order_history_delta_statistic(row['userid'], history_grouped, row['has_history_flag']), axis=1)
    features['order_history_delta_mean'] = features['order_history_delta_statistic'].map(lambda x: x[0])
    features['order_history_delta_max'] = features['order_history_delta_statistic'].map(lambda x: x[1])
    features['order_history_delta_min'] = features['order_history_delta_statistic'].map(lambda x: x[2])
    features['order_history_last_delta'] = features['order_history_delta_statistic'].map(lambda x: x[3])
    del features['order_history_delta_statistic']
    features['last_order_history_time'] = features.apply(lambda row: last_order_history_time(row['userid'], history_grouped, row['has_history_flag']), axis=1)
    features['order_history_last_delta_minus_deltamean'] = features['order_history_last_delta'] - features['order_history_delta_mean']
    features['order_history_last_delta_minus_deltamax'] = features['order_history_last_delta'] - features['order_history_delta_max']
    features['order_history_last_delta_minus_deltamin'] = features['order_history_last_delta'] - features['order_history_delta_min']
    del features['order_history_last_delta']

    features['last_month_order_now_action_count'] = features.apply(lambda row: last_month_order_now_action_count(row['userid'], action_grouped), axis=1)
    features['last_month_order_now_action_total_count'] = features['last_month_order_now_action_count'].map(lambda x: x[0])
    features['last_month_order_now_action_browse_product_count'] = features['last_month_order_now_action_count'].map(lambda x: x[1])
    features['last_month_order_now_action_browse_product2_count'] = features['last_month_order_now_action_count'].map(lambda x: x[2])
    features['last_month_order_now_action_browse_product3_count'] = features['last_month_order_now_action_count'].map(lambda x: x[3])
    features['last_month_order_now_action_fillin_form5_count'] = features['last_month_order_now_action_count'].map(lambda x: x[4])
    features['last_month_order_now_action_fillin_form6_count'] = features['last_month_order_now_action_count'].map(lambda x: x[5])
    features['last_month_order_now_action_fillin_form7_count'] = features['last_month_order_now_action_count'].map(lambda x: x[6])
    features['last_month_order_now_action_open_app_count'] = features['last_month_order_now_action_count'].map(lambda x: x[7])
    features['last_month_order_now_action_pay_money_count'] = features['last_month_order_now_action_count'].map(lambda x: x[8])
    features['last_month_order_now_action_submit_order_count'] = features['last_month_order_now_action_count'].map(lambda x: x[9])
    del features['last_month_order_now_action_count']
    # 是否有支付操作和提交订单操作
    features['last_month_order_now_has_paied_money'] = features['last_month_order_now_action_pay_money_count'].map(lambda x: int(x > 0))
    features['last_month_order_now_has_submited_order'] = features['last_month_order_now_action_submit_order_count'].map(lambda x: int(x > 0))

    del features['has_history_flag']
    return features


def get_baseline_features():
    train_features = pd.read_csv('./data_train.csv')
    test_features = pd.read_csv('./data_train.csv')

    used_features = ['userid',
                     'hasprovince',
                     'histord_ratio1_0',
                     # 'histord_sum_cont1',
                     # 'histord_sum_cont2',
                     # 'histord_sum_cont3',
                     # 'histord_sum_cont4',
                     # 'histord_sum_cont5',
                     'histord_time_last_1',
                     # 'histord_time_last_1_month',
                     # 'histord_time_last_1_year',
                     'histord_time_last_2',
                     'histord_time_last_2_month',
                     'histord_time_last_2_year',
                     'histord_time_last_3',
                     'histord_time_last_3_month',
                     'histord_time_last_3_year',
                     'timespan_action_lastord',
                     # 'timespan_lastord_1_2',
                     # 'timespan_lastord_2_3',
                     ]

    return train_features[used_features], test_features[used_features]


def diff_action_type_time_delta_after_newyear(uid, action_grouped, actiontypeA, actiontypeB, flag):
    if flag == 0:
        return -999, -999, -999, -999, -999, -999

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


def build_action_history_features8(df, action, history):
    features = pd.DataFrame({'userid': df['userid']})
    action_grouped = dict(list(action.groupby('userid')))
    action_ids = action['userid'].unique()

    # 是否有交易历史
    features['after_newyear_has_action_flag'] = features['userid'].map(lambda uid: uid in action_ids).astype(int)

    # type 1 与 5-9 的时间差统计特征
    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta_after_newyear(row['userid'], action_grouped, 1, 5, row['after_newyear_has_action_flag']), axis=1)
    features['action_type_15_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_15_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    features['action_type_15_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    features['action_type_15_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    # very bad features
    # features['action_type_15_time_delta_last2'] = features['diff_action_type_time_delta'].map(lambda x: x[4])
    # features['action_type_15_time_delta_last3'] = features['diff_action_type_time_delta'].map(lambda x: x[5])

    features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta_after_newyear(row['userid'], action_grouped, 1, 6, row['after_newyear_has_action_flag']), axis=1)
    features['action_type_16_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    features['action_type_16_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    features['action_type_16_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    features['action_type_16_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    # bad features
    # features['action_type_16_time_delta_last2'] = features['diff_action_type_time_delta'].map(lambda x: x[4])
    # features['action_type_16_time_delta_last3'] = features['diff_action_type_time_delta'].map(lambda x: x[5])

    # bad features
    # features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 1, 7), axis=1)
    # features['action_type_17_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    # features['action_type_17_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    # features['action_type_17_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    # features['action_type_17_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    # features['action_type_17_time_delta_last2'] = features['diff_action_type_time_delta'].map(lambda x: x[4])
    # features['action_type_17_time_delta_last3'] = features['diff_action_type_time_delta'].map(lambda x: x[5])

    # bad features
    # features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 1, 8), axis=1)
    # features['action_type_18_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    # features['action_type_18_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    # features['action_type_18_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    # features['action_type_18_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    # features['action_type_18_time_delta_last2'] = features['diff_action_type_time_delta'].map(lambda x: x[4])
    # features['action_type_18_time_delta_last3'] = features['diff_action_type_time_delta'].map(lambda x: x[5])

    # bad features
    # features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 1, 9), axis=1)
    # features['action_type_19_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    # features['action_type_19_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    # features['action_type_19_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    # features['action_type_19_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    # features['action_type_19_time_delta_last2'] = features['diff_action_type_time_delta'].map(lambda x: x[4])
    # features['action_type_19_time_delta_last3'] = features['diff_action_type_time_delta'].map(lambda x: x[5])

    # bad features
    # features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 2, 5), axis=1)
    # features['action_type_25_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    # features['action_type_25_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    # features['action_type_25_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    # features['action_type_25_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])
    # features['diff_action_type_time_delta'] = features.apply(lambda row: diff_action_type_time_delta(row['userid'], action_grouped, 3, 5), axis=1)
    # features['action_type_35_time_delta_min'] = features['diff_action_type_time_delta'].map(lambda x: x[0])
    # features['action_type_35_time_delta_max'] = features['diff_action_type_time_delta'].map(lambda x: x[1])
    # features['action_type_35_time_delta_mean'] = features['diff_action_type_time_delta'].map(lambda x: x[2])
    # features['action_type_35_time_delta_std'] = features['diff_action_type_time_delta'].map(lambda x: x[3])

    del features['diff_action_type_time_delta']
    del features['after_newyear_has_action_flag']
    return features


def action_time_seq_fft(uid, action_grouped):
    df = action_grouped[uid]
    sequence = df['actionTime'].values.tolist()
    result = fft(sequence, len(sequence) + 2)
    real = result.real
    return real[0], real[1], real[2]


def action_type_seq_fft(uid, action_grouped):
    df = action_grouped[uid]
    sequence = df['actionType'].values.tolist()
    result = fft(sequence, len(sequence) + 2)
    real = result.real
    return real[0], real[1], real[2]


def build_action_history_features9(df, action):
    features = pd.DataFrame({'userid': df['userid']})
    action_grouped = dict(list(action.groupby('userid')))

    features['action_time_seq_fft'] = features.apply(lambda row: action_time_seq_fft(row['userid'], action_grouped), axis=1)
    features['actiontime_seq_fft_real_0'] = features['action_time_seq_fft'].map(lambda x: x[0])
    features['actiontime_seq_fft_real_1'] = features['action_time_seq_fft'].map(lambda x: x[1])
    features['actiontime_seq_fft_real_2'] = features['action_time_seq_fft'].map(lambda x: x[2])
    del features['action_time_seq_fft']

    # features['action_type_seq_fft'] = features.apply(lambda row: action_type_seq_fft(row['userid'], action_grouped), axis=1)
    # features['actiontype_seq_fft_real_0'] = features['action_type_seq_fft'].map(lambda x: x[0])
    # features['actiontype_seq_fft_real_1'] = features['action_type_seq_fft'].map(lambda x: x[1])
    # features['actiontype_seq_fft_real_2'] = features['action_type_seq_fft'].map(lambda x: x[2])
    # del features['action_type_seq_fft']

    return features


def goodorder_vs_actiontype_ratio(uid, action_grouped, history_grouped, flag, action_type):
    """ 精品订单量与 action 操作数量的比值 """
    if flag == 0:
        return 0

    history_df = history_grouped[uid]
    action_df = action_grouped[uid]

    action_type_df = action_df[action_df['actionType'] == action_type]

    if action_type_df.shape[0] == 0:
        return 0
    else:
        return 1.0 * sum(history_df['orderType']) / action_type_df.shape[0]


def order_vs_actiontype_ratio(uid, action_grouped, history_grouped, flag, action_type):
    """ 精品订单量与 action 操作数量的比值 """
    if flag == 0:
        return 0

    history_df = history_grouped[uid]
    action_df = action_grouped[uid]

    action_type_df = action_df[action_df['actionType'] == action_type]

    if action_type_df.shape[0] == 0:
        return 0
    else:
        return 1.0 * history_df.shape[0] / action_type_df.shape[0]


def last_order_timestamp(uid, history_grouped, flag):
    if flag == 0:
        return default_start_order_time, default_end_order_time - default_start_order_time

    df = history_grouped[uid]
    last_order_time = df['orderTime'].iat[-1]
    return last_order_time, default_end_order_time - last_order_time


def last_order_actiontime_statistic(uid, action_grouped, history_grouped, flag):
    if flag == 0:
        last_order_time = default_start_order_time
    else:
        last_order_time = history_grouped[uid]['orderTime'].iat[-1]

    action_df = action_grouped[uid]
    action_df = action_df[action_df['actionTime'] < last_order_time]
    action_times = action_df['actionTime'].values
    action_type1_times = action_df[action_df['actionType'] == 1]['actionTime'].values
    action_type234_times = action_df[(action_df['actionType'] == 2) | (action_df['actionType'] == 3) | (action_df['actionType'] == 4)]['actionTime'].values
    action_type5_times = action_df[action_df['actionType'] == 5]['actionTime'].values
    action_type6_times = action_df[action_df['actionType'] == 6]['actionTime'].values
    action_type7_times = action_df[action_df['actionType'] == 7]['actionTime'].values
    action_type8_times = action_df[action_df['actionType'] == 8]['actionTime'].values

    return np.mean(action_times), np.std(action_times), \
           np.mean(action_type1_times), np.std(action_type1_times), \
           np.mean(action_type234_times), np.std(action_type234_times), \
           np.mean(action_type5_times), np.std(action_type5_times), \
           np.mean(action_type6_times), np.std(action_type6_times), \
           np.mean(action_type7_times), np.std(action_type7_times), \
           np.mean(action_type8_times), np.std(action_type8_times)
    
    
def build_action_history_features10(df, action, history):
    features = pd.DataFrame({'userid': df['userid']})
    action_grouped = dict(list(action.groupby('userid')))
    history['orderTime'] = history.orderTime.values.astype(np.int64) // 10 ** 9
    history_grouped = dict(list(history.groupby('userid')))

    history_uids = history['userid'].unique()
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in history_uids).astype(int)

    print('用户精品/总订单与浏览量比值')
    features['goodorder_vs_actiontype_1_ratio'] = features.apply(lambda row: goodorder_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 1), axis=1)
    features['goodorder_vs_actiontype_2_ratio'] = features.apply(lambda row: goodorder_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 2), axis=1)
    features['goodorder_vs_actiontype_3_ratio'] = features.apply(lambda row: goodorder_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 3), axis=1)
    features['goodorder_vs_actiontype_4_ratio'] = features.apply(lambda row: goodorder_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 4), axis=1)
    features['goodorder_vs_actiontype_5_ratio'] = features.apply(lambda row: goodorder_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 5), axis=1)
    features['goodorder_vs_actiontype_6_ratio'] = features.apply(lambda row: goodorder_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 6), axis=1)
    features['goodorder_vs_actiontype_7_ratio'] = features.apply(lambda row: goodorder_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 7), axis=1)
    features['goodorder_vs_actiontype_8_ratio'] = features.apply(lambda row: goodorder_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 8), axis=1)
    features['goodorder_vs_actiontype_9_ratio'] = features.apply(lambda row: goodorder_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 9), axis=1)

    features['order_vs_actiontype_1_ratio'] = features.apply(lambda row: order_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 1), axis=1)
    features['order_vs_actiontype_2_ratio'] = features.apply(lambda row: order_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 2), axis=1)
    features['order_vs_actiontype_3_ratio'] = features.apply(lambda row: order_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 3), axis=1)
    features['order_vs_actiontype_4_ratio'] = features.apply(lambda row: order_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 4), axis=1)
    features['order_vs_actiontype_5_ratio'] = features.apply(lambda row: order_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 5), axis=1)
    features['order_vs_actiontype_6_ratio'] = features.apply(lambda row: order_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 6), axis=1)
    features['order_vs_actiontype_7_ratio'] = features.apply(lambda row: order_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 7), axis=1)
    features['order_vs_actiontype_8_ratio'] = features.apply(lambda row: order_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 8), axis=1)
    features['order_vs_actiontype_9_ratio'] = features.apply(lambda row: order_vs_actiontype_ratio(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 9), axis=1)

    print('距离上一次 order 到现在的 action type 的时间差的统计特征')
    features['last_order_timestamp_state'] = features.apply(lambda row: last_order_timestamp(row['userid'], history_grouped, row['has_history_flag']), axis=1)
    features['last_order_timestamp'] = features['last_order_timestamp_state'].map(lambda x: x[0])
    features['last_order_timestamp_from_now_delta'] = features['last_order_timestamp_state'].map(lambda x: x[1])
    features['last_order_timestamp_from_now_delta_vs_last_ratio'] = features['last_order_timestamp_from_now_delta'].astype(float) / features['last_order_timestamp']
    del features['last_order_timestamp_state']

    # features['last_order_actiontime_statistic'] = features.apply(lambda row: last_order_actiontime_statistic(row['userid'], action_grouped, history_grouped, row['has_history_flag']), axis=1)
    # features['last_order_action_time_mean'] = features['last_order_actiontime_statistic'].map(lambda x: x[0])
    # features['last_order_action_time_std'] = features['last_order_actiontime_statistic'].map(lambda x: x[1])
    # # features['last_order_action1_time_mean'] = features['last_order_actiontime_statistic'].map(lambda x: x[2])
    # # features['last_order_action1_time_std'] = features['last_order_actiontime_statistic'].map(lambda x: x[3])
    # features['last_order_action234_time_mean'] = features['last_order_actiontime_statistic'].map(lambda x: x[4])
    # features['last_order_action234_time_std'] = features['last_order_actiontime_statistic'].map(lambda x: x[5])
    # features['last_order_action5_time_mean'] = features['last_order_actiontime_statistic'].map(lambda x: x[6])
    # # features['last_order_action5_time_std'] = features['last_order_actiontime_statistic'].map(lambda x: x[7])
    # features['last_order_action6_time_mean'] = features['last_order_actiontime_statistic'].map(lambda x: x[8])
    # # features['last_order_action6_time_std'] = features['last_order_actiontime_statistic'].map(lambda x: x[9])
    # features['last_order_action7_time_mean'] = features['last_order_actiontime_statistic'].map(lambda x: x[10])
    # # features['last_order_action7_time_std'] = features['last_order_actiontime_statistic'].map(lambda x: x[11])
    # features['last_order_action8_time_mean'] = features['last_order_actiontime_statistic'].map(lambda x: x[12])
    # # features['last_order_action8_time_std'] = features['last_order_actiontime_statistic'].map(lambda x: x[13])
    # del features['last_order_actiontime_statistic']

    del features['has_history_flag']
    return features


def last_order_first_actiontype_time(uid, action_grouped, history_grouped, flag, action_type):
    """ 最后一次 order 之后第一次 actiontype 的时间 """
    if flag == 0:
        return -1

    action_df = action_grouped[uid]
    action_df = action_df[action_df['actionTime'] > history_grouped[uid]['orderTime'].iat[-1]]
    if action_df.shape[0] > 0:
        action_df = action_df[action_df['actionType'] == action_type]
        if action_df.shape[0] > 0:
            return action_df['actionTime'].iat[0]
        return -1
    return -1


def last_openapp_browse_count(uid, action_grouped):
    """ 最后一次点击APP开始浏览量 """
    action_df = action_grouped[uid]
    action_df = action_df[action_df['actionType'] == 1]
    if action_df.shape[0] == 0:
        return 0

    last_open_time = action_df['actionTime'].iat[-1]
    action_df = action_df[action_df['actionTime'] > last_open_time]
    return action_df[(action_df['actionType'] == 2) | (action_df['actionType'] == 3) | (action_df['actionType'] == 4)].shape[0]


def last_openapp_fillform_count(uid, action_grouped):
    """ 最后一次点击APP开始填写表单量 """
    action_df = action_grouped[uid]
    action_df = action_df[action_df['actionType'] == 1]
    if action_df.shape[0] == 0:
        return 0

    last_open_time = action_df['actionTime'].iat[-1]
    action_df = action_df[action_df['actionTime'] > last_open_time]
    return action_df[(action_df['actionType'] == 5) | (action_df['actionType'] == 6) | (action_df['actionType'] == 7)].shape[0]


def build_action_history_features11(df, action, history):
    features = pd.DataFrame({'userid': df['userid']})
    action_grouped = dict(list(action.groupby('userid')))
    history['orderTime'] = history.orderTime.values.astype(np.int64) // 10 ** 9
    history_grouped = dict(list(history.groupby('userid')))

    history_uids = history['userid'].unique()
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in history_uids).astype(int)

    # # 最后一次 order 之后第一次 actiontype 的时间
    # # features['last_order_first_actiontype1_time'] = features.apply(lambda row: last_order_first_actiontype_time(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 1), axis=1)
    # # features['last_order_first_actiontype2_time'] = features.apply(lambda row: last_order_first_actiontype_time(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 2), axis=1)
    # features['last_order_first_actiontype3_time'] = features.apply(lambda row: last_order_first_actiontype_time(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 3), axis=1)
    # features['last_order_first_actiontype4_time'] = features.apply(lambda row: last_order_first_actiontype_time(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 4), axis=1)
    # # features['last_order_first_actiontype5_time'] = features.apply(lambda row: last_order_first_actiontype_time(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 5), axis=1)
    # # features['last_order_first_actiontype6_time'] = features.apply(lambda row: last_order_first_actiontype_time(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 6), axis=1)
    # features['last_order_first_actiontype7_time'] = features.apply(lambda row: last_order_first_actiontype_time(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 7), axis=1)
    # features['last_order_first_actiontype8_time'] = features.apply(lambda row: last_order_first_actiontype_time(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 8), axis=1)
    # # features['last_order_first_actiontype9_time'] = features.apply(lambda row: last_order_first_actiontype_time(row['userid'], action_grouped, history_grouped, row['has_history_flag'], 9), axis=1)

    # 最后一次点击APP开始浏览量
    features['last_openapp_browse_count'] = features.apply(lambda row: last_openapp_browse_count(row['userid'], action_grouped), axis=1)
    # features['last_openapp_fillform_count'] = features.apply(lambda row: last_openapp_fillform_count(row['userid'], action_grouped), axis=1)

    del features['has_history_flag']
    return features


def main():

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
    if not data_utils.is_feature_created(feature_name):
        print('build train action history features4')
        train_features = build_action_history_features4(train, action_train, orderHistory_train)
        print('build test action history features4')
        test_features = build_action_history_features4(test, action_test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    action_train.sort_values(by='actionTime', inplace=True)
    action_test.sort_values(by='actionTime', inplace=True)

    feature_name = 'action_history_features5'
    if not data_utils.is_feature_created(feature_name):
        print('build train action history features5')
        train_features = build_action_history_features5(train, action_train, orderHistory_train)
        print('build test action history features5')
        test_features = build_action_history_features5(test, action_test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'action_history_features6'
    if not data_utils.is_feature_created(feature_name):
        train_features3, test_features3 = data_utils.load_features('action_history_features3')
        print('build train action history features6')
        train_features = build_action_history_features6(train_features3)
        print('build test action history features6')
        test_features = build_action_history_features6(test_features3)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'action_history_features7'
    if not data_utils.is_feature_created(feature_name):
        action_train = pd.read_csv(Configure.cleaned_path + 'cleaned_action_train.csv')
        action_test = pd.read_csv(Configure.cleaned_path + 'cleaned_action_test.csv')
        action_train['actionTime'] = pd.to_datetime(action_train['actionTime'])
        action_test['actionTime'] = pd.to_datetime(action_test['actionTime'])
        action_train.sort_values(by='actionTime', inplace=True)
        action_test.sort_values(by='actionTime', inplace=True)
        print('build train action history features7')
        train_features = build_action_history_features7(train, action_train, orderHistory_train)
        print('build test action history features7')
        test_features = build_action_history_features7(test, action_test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'action_history_features8'
    if not data_utils.is_feature_created(feature_name):
        print('build train action history features8')
        train_features = build_action_history_features8(train, action_train, orderHistory_train)
        print('build test action history features8')
        test_features = build_action_history_features8(test, action_test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'action_history_features9'
    if not data_utils.is_feature_created(feature_name):
        print('build trainaction_history_features9')
        train_features = build_action_history_features9(train, action_train)
        print('build test action_history_features9')
        test_features = build_action_history_features9(test, action_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'baseline_features'
    if not data_utils.is_feature_created(feature_name):
        train_features, test_features = get_baseline_features()
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    # action 和 history 相结合构造特征
    action_train = pd.read_csv(Configure.base_path + 'train/action_train.csv')
    action_test = pd.read_csv(Configure.base_path + 'test/action_test.csv')

    feature_name = 'action_history_features10'
    if not data_utils.is_feature_created(feature_name):
        print('build train action history features10')
        train_features = build_action_history_features10(train, action_train, orderHistory_train)
        print('build test action history features10')
        test_features = build_action_history_features10(test, action_test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'action_history_features11'
    if not data_utils.is_feature_created(feature_name):
        print('build train action history features11')
        train_features = build_action_history_features11(train, action_train, orderHistory_train)
        print('build test action history features11')
        test_features = build_action_history_features11(test, action_test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 结合 action、 history 和 comment 提取历史特征 ==========")
    main()
