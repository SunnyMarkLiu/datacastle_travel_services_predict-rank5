#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-21 下午3:01
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
from conf.configure import Configure
from utils import data_utils

default_start_order_time = int(time.mktime(datetime.datetime(2016, 11, 01).timetuple()))
default_end_order_time = int(time.mktime(datetime.datetime(2017, 9, 12).timetuple()))


def order_type_sum(uid, history_grouped, flag):
    """ 最近交易的信息 """
    if flag == 0:
        return 0
    history_df = history_grouped[uid]
    return sum(history_df['orderType'])


def gen_history_features(df, history):
    features = pd.DataFrame({'userid': df['userid']})

    df_ids = history['userid'].unique()
    history['orderTime'] = history.orderTime.values.astype(np.int64) // 10 ** 9
    history_grouped = dict(list(history.groupby('userid')))

    # 给trade表打标签，若id在login表中，则打标签为1，否则为0
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in df_ids).astype(int)

    """ history_order_type_sum 和 history_order_type_sum_lg0 后期再加上"""
    features['history_order_type_sum'] = features.apply(
        lambda row: order_type_sum(row['userid'], history_grouped, row['has_history_flag']), axis=1)
    features['history_order_type_sum_lg0'] = features['history_order_type_sum'].map(lambda x: int(x > 0))
    del features['history_order_type_sum']

    del features['has_history_flag']
    return features


def gen_actiontype_sequence(uid, action_grouped):
    """ 用户订单历史结果构成的序列 """
    df = action_grouped[uid]
    sequence = ' '.join(df['actionType'].astype(str).values.tolist())
    return sequence


def last_day_action_statistic(uid, action_grouped):
    action_df = action_grouped[uid]

    # 最后一天
    last_action_day = action_df['days_from_now'].iat[-1]
    action_df = action_df[action_df['days_from_now'] == last_action_day]

    last_day_action_count = action_df.shape[0] * 1.0
    actiontypes = action_df['actionType'].values.tolist()
    actiontypes1_count = actiontypes.count(1)
    actiontypes2_count = actiontypes.count(2)
    actiontypes3_count = actiontypes.count(3)
    actiontypes4_count = actiontypes.count(4)
    actiontypes5_count = actiontypes.count(5)
    actiontypes6_count = actiontypes.count(6)
    actiontypes7_count = actiontypes.count(7)
    actiontypes8_count = actiontypes.count(8)
    actiontypes9_count = actiontypes.count(9)

    actiontypes1_ratio = 1.0 * actiontypes1_count / last_day_action_count if last_day_action_count != 0 else 0
    actiontypes2_ratio = 1.0 * actiontypes2_count / last_day_action_count if last_day_action_count != 0 else 0
    actiontypes3_ratio = 1.0 * actiontypes3_count / last_day_action_count if last_day_action_count != 0 else 0
    actiontypes4_ratio = 1.0 * actiontypes4_count / last_day_action_count if last_day_action_count != 0 else 0
    actiontypes5_ratio = 1.0 * actiontypes5_count / last_day_action_count if last_day_action_count != 0 else 0
    actiontypes6_ratio = 1.0 * actiontypes6_count / last_day_action_count if last_day_action_count != 0 else 0
    actiontypes7_ratio = 1.0 * actiontypes7_count / last_day_action_count if last_day_action_count != 0 else 0
    actiontypes8_ratio = 1.0 * actiontypes8_count / last_day_action_count if last_day_action_count != 0 else 0
    actiontypes9_ratio = 1.0 * actiontypes9_count / last_day_action_count if last_day_action_count != 0 else 0

    actiontimes = action_df['actionTime'].values
    actiontimes_mean = np.mean(actiontimes)
    actiontimes_max = np.max(actiontimes)
    actiontimes_min = np.min(actiontimes)
    actiontimes_std = np.std(actiontimes)
    actiontimes_extreme_delta = actiontimes_max - actiontimes_min
    actiontimes_std_mean = (actiontimes_std + 1.0) / (actiontimes_mean + 1)

    return actiontypes1_ratio, actiontypes2_ratio, actiontypes3_ratio, actiontypes4_ratio, actiontypes5_ratio, \
           actiontypes6_ratio, actiontypes7_ratio, actiontypes8_ratio, actiontypes9_ratio, last_day_action_count, \
           actiontimes_mean, actiontimes_max, actiontimes_min, actiontimes_std, actiontimes_extreme_delta, actiontimes_std_mean


def continues_action_info(uid, action_grouped):
    """ 连续操作 app 的特征 """
    action_df = action_grouped[uid]

    continues_action_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    action_types = action_df['actionType'].values
    for i in range(len(action_types) - 1):
        delta = action_types[i + 1] - action_types[i]
        if delta == 0:  # 出现重复操作同一个 action type
            continues_action_count[action_types[i + 1]] = continues_action_count.get(action_types[i + 1]) + 1

    action_1_continues_ratio = 1.0 * (continues_action_count[1] + 1) / (action_df.shape[0]+2)
    action_2_continues_ratio = 1.0 * (continues_action_count[2] + 1) / (action_df.shape[0]+2)
    action_3_continues_ratio = 1.0 * (continues_action_count[3] + 1) / (action_df.shape[0]+2)
    action_4_continues_ratio = 1.0 * (continues_action_count[4] + 1) / (action_df.shape[0]+2)
    action_5_continues_ratio = 1.0 * (continues_action_count[5] + 1) / (action_df.shape[0]+2)
    action_6_continues_ratio = 1.0 * (continues_action_count[6] + 1) / (action_df.shape[0]+2)
    action_7_continues_ratio = 1.0 * (continues_action_count[7] + 1) / (action_df.shape[0]+2)
    action_8_continues_ratio = 1.0 * (continues_action_count[8] + 1) / (action_df.shape[0]+2)
    action_9_continues_ratio = 1.0 * (continues_action_count[9] + 1) / (action_df.shape[0]+2)

    return action_1_continues_ratio, action_2_continues_ratio, action_3_continues_ratio, action_4_continues_ratio, \
           action_5_continues_ratio, action_6_continues_ratio, action_7_continues_ratio, action_8_continues_ratio, \
           action_9_continues_ratio


def nextaction_current_action_timedelta_statistic(uid, action_grouped):
    """ action type 后的第一个操作的时间差的统计特征， 9 × 4 个特征 """
    action_df = action_grouped[uid]

    action_type1_deltas = []
    action_type2_deltas = []
    action_type3_deltas = []
    action_type4_deltas = []
    action_type5_deltas = []
    action_type6_deltas = []
    action_type7_deltas = []
    action_type8_deltas = []
    action_type9_deltas = []

    actionTypes = action_df['actionType'].values
    actionTimes = action_df['actionTime'].values

    for i in range(len(actionTypes) - 1):
        if actionTypes[i] == 1:
            action_type1_deltas.append(actionTimes[i+1] - actionTypes[i])
        if actionTypes[i] == 2:
            action_type2_deltas.append(actionTimes[i+1] - actionTypes[i])
        if actionTypes[i] == 3:
            action_type3_deltas.append(actionTimes[i + 1] - actionTypes[i])
        if actionTypes[i] == 4:
            action_type4_deltas.append(actionTimes[i + 1] - actionTypes[i])
        if actionTypes[i] == 5:
            action_type5_deltas.append(actionTimes[i + 1] - actionTypes[i])
        if actionTypes[i] == 6:
            action_type6_deltas.append(actionTimes[i + 1] - actionTypes[i])
        if actionTypes[i] == 7:
            action_type7_deltas.append(actionTimes[i + 1] - actionTypes[i])
        if actionTypes[i] == 8:
            action_type8_deltas.append(actionTimes[i + 1] - actionTypes[i])
        if actionTypes[i] == 9:
            action_type9_deltas.append(actionTimes[i + 1] - actionTypes[i])

    action_type1_deltas_mean = np.mean(action_type1_deltas) if len(action_type1_deltas) > 0 else -999
    action_type1_deltas_min = np.min(action_type1_deltas) if len(action_type1_deltas) > 0 else -999
    action_type1_deltas_max = np.max(action_type1_deltas) if len(action_type1_deltas) > 0 else -999
    action_type1_deltas_std = np.std(action_type1_deltas) if len(action_type1_deltas) > 0 else -999

    action_type2_deltas_mean = np.mean(action_type2_deltas) if len(action_type2_deltas) > 0 else -999
    action_type2_deltas_min = np.min(action_type2_deltas) if len(action_type2_deltas) > 0 else -999
    action_type2_deltas_max = np.max(action_type2_deltas) if len(action_type2_deltas) > 0 else -999
    action_type2_deltas_std = np.std(action_type2_deltas) if len(action_type2_deltas) > 0 else -999

    action_type3_deltas_mean = np.mean(action_type3_deltas) if len(action_type3_deltas) > 0 else -999
    action_type3_deltas_min = np.min(action_type3_deltas) if len(action_type3_deltas) > 0 else -999
    action_type3_deltas_max = np.max(action_type3_deltas) if len(action_type3_deltas) > 0 else -999
    action_type3_deltas_std = np.std(action_type3_deltas) if len(action_type3_deltas) > 0 else -999

    action_type4_deltas_mean = np.mean(action_type4_deltas) if len(action_type4_deltas) > 0 else -999
    action_type4_deltas_min = np.min(action_type4_deltas) if len(action_type4_deltas) > 0 else -999
    action_type4_deltas_max = np.max(action_type4_deltas) if len(action_type4_deltas) > 0 else -999
    action_type4_deltas_std = np.std(action_type4_deltas) if len(action_type4_deltas) > 0 else -999

    action_type5_deltas_mean = np.mean(action_type5_deltas) if len(action_type5_deltas) > 0 else -999
    action_type5_deltas_min = np.min(action_type5_deltas) if len(action_type5_deltas) > 0 else -999
    action_type5_deltas_max = np.max(action_type5_deltas) if len(action_type5_deltas) > 0 else -999
    action_type5_deltas_std = np.std(action_type5_deltas) if len(action_type5_deltas) > 0 else -999

    action_type6_deltas_mean = np.mean(action_type6_deltas) if len(action_type6_deltas) > 0 else -999
    action_type6_deltas_min = np.min(action_type6_deltas) if len(action_type6_deltas) > 0 else -999
    action_type6_deltas_max = np.max(action_type6_deltas) if len(action_type6_deltas) > 0 else -999
    action_type6_deltas_std = np.std(action_type6_deltas) if len(action_type6_deltas) > 0 else -999

    action_type7_deltas_mean = np.mean(action_type7_deltas) if len(action_type7_deltas) > 0 else -999
    action_type7_deltas_min = np.min(action_type7_deltas) if len(action_type7_deltas) > 0 else -999
    action_type7_deltas_max = np.max(action_type7_deltas) if len(action_type7_deltas) > 0 else -999
    action_type7_deltas_std = np.std(action_type7_deltas) if len(action_type7_deltas) > 0 else -999

    action_type8_deltas_mean = np.mean(action_type8_deltas) if len(action_type8_deltas) > 0 else -999
    action_type8_deltas_min = np.min(action_type8_deltas) if len(action_type8_deltas) > 0 else -999
    action_type8_deltas_max = np.max(action_type8_deltas) if len(action_type8_deltas) > 0 else -999
    action_type8_deltas_std = np.std(action_type8_deltas) if len(action_type8_deltas) > 0 else -999

    action_type9_deltas_mean = np.mean(action_type9_deltas) if len(action_type9_deltas) > 0 else -999
    action_type9_deltas_min = np.min(action_type9_deltas) if len(action_type9_deltas) > 0 else -999
    action_type9_deltas_max = np.max(action_type9_deltas) if len(action_type9_deltas) > 0 else -999
    action_type9_deltas_std = np.std(action_type9_deltas) if len(action_type9_deltas) > 0 else -999

    return action_type1_deltas_mean, action_type1_deltas_min, action_type1_deltas_max, action_type1_deltas_std, \
           action_type2_deltas_mean, action_type2_deltas_min, action_type2_deltas_max, action_type2_deltas_std, \
           action_type3_deltas_mean, action_type3_deltas_min, action_type3_deltas_max, action_type3_deltas_std, \
           action_type4_deltas_mean, action_type4_deltas_min, action_type4_deltas_max, action_type4_deltas_std, \
           action_type5_deltas_mean, action_type5_deltas_min, action_type5_deltas_max, action_type5_deltas_std, \
           action_type6_deltas_mean, action_type6_deltas_min, action_type6_deltas_max, action_type6_deltas_std, \
           action_type7_deltas_mean, action_type7_deltas_min, action_type7_deltas_max, action_type7_deltas_std, \
           action_type8_deltas_mean, action_type8_deltas_min, action_type8_deltas_max, action_type8_deltas_std, \
           action_type9_deltas_mean, action_type9_deltas_min, action_type9_deltas_max, action_type9_deltas_std

def gen_action_features(df, action):
    features = pd.DataFrame({'userid': df['userid']})
    action_grouped = dict(list(action.groupby('userid')))

    # features['action_sequence'] = features.apply(lambda row: gen_actiontype_sequence(row['userid'], action_grouped), axis=1)
    # action_texts = features['action_sequence'].values
    # vectorizer = TfidfVectorizer(stop_words=None)
    # dtm = vectorizer.fit_transform(action_texts)
    # # vocab = np.array(vectorizer.get_feature_names())
    # # clf = decomposition.NMF(n_components=20, random_state=1)
    # # doctopic = clf.fit_transform(dtm)
    # # doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
    # doctopic = pd.DataFrame(dtm.toarray(), columns=['topic_model_{}'.format(i) for i in range(9)])
    # features = pd.concat([features, doctopic], axis=1)
    #
    # del features['action_sequence']

    # 最后一天 action 操作情况
    # features['last_day_action_statistic'] = features.apply(lambda row: last_day_action_statistic(row['userid'], action_grouped), axis=1)
    # features['last_day_actiontypes1_ratio'] = features['last_day_action_statistic'].map(lambda x: x[0])
    # features['last_day_actiontypes2_ratio'] = features['last_day_action_statistic'].map(lambda x: x[1])
    # features['last_day_actiontypes3_ratio'] = features['last_day_action_statistic'].map(lambda x: x[2])
    # features['last_day_actiontypes4_ratio'] = features['last_day_action_statistic'].map(lambda x: x[3])
    # features['last_day_actiontypes5_ratio'] = features['last_day_action_statistic'].map(lambda x: x[4])
    # features['last_day_actiontypes6_ratio'] = features['last_day_action_statistic'].map(lambda x: x[5])
    # features['last_day_actiontypes7_ratio'] = features['last_day_action_statistic'].map(lambda x: x[6])
    # features['last_day_actiontypes8_ratio'] = features['last_day_action_statistic'].map(lambda x: x[7])
    # features['last_day_actiontypes9_ratio'] = features['last_day_action_statistic'].map(lambda x: x[8])
    # features['last_day_action_count'] = features['last_day_action_statistic'].map(lambda x: x[9])
    # features['last_day_actiontimes_mean'] = features['last_day_action_statistic'].map(lambda x: x[10])
    # features['last_day_actiontimes_max'] = features['last_day_action_statistic'].map(lambda x: x[11])
    # features['last_day_actiontimes_min'] = features['last_day_action_statistic'].map(lambda x: x[12])
    # features['last_day_actiontimes_std'] = features['last_day_action_statistic'].map(lambda x: x[13])
    # features['last_day_actiontimes_extreme_delta'] = features['last_day_action_statistic'].map(lambda x: x[14])
    # features['last_day_actiontimes_std_mean'] = features['last_day_action_statistic'].map(lambda x: x[15])
    # del features['last_day_action_statistic']

    # # 连续操作 app 的特征
    # features['continues_action_info'] = features.apply(lambda row: continues_action_info(row['userid'], action_grouped), axis=1)
    # features['action_1_continues_ratio'] = features['continues_action_info'].map(lambda x:x[0])
    # features['action_2_continues_ratio'] = features['continues_action_info'].map(lambda x:x[1])
    # features['action_3_continues_ratio'] = features['continues_action_info'].map(lambda x:x[2])
    # features['action_4_continues_ratio'] = features['continues_action_info'].map(lambda x:x[3])
    # features['action_5_continues_ratio'] = features['continues_action_info'].map(lambda x:x[4])
    # features['action_6_continues_ratio'] = features['continues_action_info'].map(lambda x:x[5])
    # features['action_7_continues_ratio'] = features['continues_action_info'].map(lambda x:x[6])
    # features['action_8_continues_ratio'] = features['continues_action_info'].map(lambda x:x[7])
    # features['action_9_continues_ratio'] = features['continues_action_info'].map(lambda x:x[8])
    # del features['continues_action_info']

    # action type 后的第一个操作的时间差的统计特征， 9 × 4
    print('nextaction_current_action_timedelta_statistic')
    features['nextaction_current_action_timedelta_statistic'] = features.apply(lambda row: nextaction_current_action_timedelta_statistic(row['userid'], action_grouped), axis=1)
    features['action_type1_deltas_mean'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[0])
    features['action_type1_deltas_min'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[1])
    features['action_type1_deltas_max'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[2])
    features['action_type1_deltas_std'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[3])

    features['action_type2_deltas_mean'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[4])
    features['action_type2_deltas_min'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[5])
    features['action_type2_deltas_max'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[6])
    features['action_type2_deltas_std'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[7])

    features['action_type3_deltas_mean'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[8])
    features['action_type3_deltas_min'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[9])
    features['action_type3_deltas_max'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[10])
    features['action_type3_deltas_std'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[11])

    features['action_type4_deltas_mean'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[12])
    features['action_type4_deltas_min'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[13])
    features['action_type4_deltas_max'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[14])
    features['action_type4_deltas_std'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[15])

    features['action_type5_deltas_mean'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[16])
    features['action_type5_deltas_min'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[17])
    features['action_type5_deltas_max'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[18])
    features['action_type5_deltas_std'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[19])

    features['action_type6_deltas_mean'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[20])
    features['action_type6_deltas_min'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[21])
    features['action_type6_deltas_max'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[22])
    features['action_type6_deltas_std'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[23])

    features['action_type7_deltas_mean'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[24])
    features['action_type7_deltas_min'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[25])
    features['action_type7_deltas_max'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[26])
    features['action_type7_deltas_std'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[27])

    features['action_type8_deltas_mean'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[28])
    features['action_type8_deltas_min'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[29])
    features['action_type8_deltas_max'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[30])
    features['action_type8_deltas_std'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[31])

    features['action_type9_deltas_mean'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[32])
    features['action_type9_deltas_min'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[33])
    features['action_type9_deltas_max'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[34])
    features['action_type9_deltas_std'] = features['nextaction_current_action_timedelta_statistic'].map(lambda x: x[35])

    del features['nextaction_current_action_timedelta_statistic']

    # 用户使用最频繁的一天使用APP距离现在的时间
    features['most_actionapp_days'] = features.apply(lambda row: most_actionapp_days(row['userid'], action_grouped), axis=1)
    features['most_actionapp_day'] = features['most_actionapp_days'].map(lambda x: x[0])
    features['most_actionapp_day_mean'] = features['most_actionapp_days'].map(lambda x: x[1])
    features['most_actionapp_day_min'] = features['most_actionapp_days'].map(lambda x: x[2])
    features['most_actionapp_day_max'] = features['most_actionapp_days'].map(lambda x: x[3])
    features['most_actionapp_day_std'] = features['most_actionapp_days'].map(lambda x: x[4])
    del features['most_actionapp_days']
    return features


def most_actionapp_days(uid, action_grouped):
    df = action_grouped[uid]
    df = df.groupby('days_from_now').count()['action_year'].reset_index().rename(columns={'action_year':'day_action_count'})
    df = df.sort_values(by='day_action_count', ascending=False).reset_index(drop=True)

    days = df['days_from_now'].values.tolist()

    return days[0], np.mean(days), np.min(days), max(days), np.std(days)


def build_time_features(action_df):
    action_df['actionTime2'] = pd.to_datetime(action_df['actionTime'], unit='s')
    # 训练集和测试集最后一天是 2017-09-11
    now = datetime.datetime(2017, 9, 12)
    action_df['days_from_now'] = action_df['actionTime2'].map(lambda order: (now - order).days)
    action_df['date'] = action_df['actionTime2'].dt.date

    action_df['action_year'] = action_df['actionTime2'].dt.year
    action_df['action_month'] = action_df['actionTime2'].dt.month
    action_df['action_day'] = action_df['actionTime2'].dt.day
    action_df['action_weekofyear'] = action_df['actionTime2'].dt.weekofyear
    action_df['action_weekday'] = action_df['actionTime2'].dt.weekday
    action_df['action_hour'] = action_df['actionTime2'].dt.hour
    action_df['action_minute'] = action_df['actionTime2'].dt.minute
    action_df['action_is_weekend'] = action_df['action_weekday'].map(lambda d: 1 if (d == 0) | (d == 6) else 0)
    action_df['action_week_hour'] = action_df['action_weekday'] * 24 + action_df['action_hour']
    del action_df['actionTime2']

    return action_df


def g_h_filter(data, x0, dx=1, g=3. / 10, h=1. / 3, dt=1., pred=None):
    x = x0
    results = []
    for z in data:
        x_est = x + (dx * dt)
        dx = dx
        if pred is not None:
            pred.append(x_est)
        residual = z - x_est
        dx = dx + h * residual / dt
        x = x_est + g * residual
        results.append(x)
    return np.array(results)


def action_tpye_kalman(uid, action_grouped):
    df = action_grouped[uid]
    sequence = df['actionType'].values.tolist()
    first = df['actionType'].iat[0]
    real = g_h_filter(sequence, first)
    real_0 = real[0]
    real_1 = real_2 = 0
    if len(real) > 1:
        real_1 = real[1]
        if len(real) > 2:
            real_2 = real[2]
    return real_0, real_1, real_2, np.std(real), np.mean(real), np.max(real), np.min(real)


def gen_action_in_hot_time(uid, action_grouped):
    action_df = action_grouped[uid]
    action_time = []
    pre_time = action_df['actionTime'].iat[0]
    action_time.append(pre_time)
    # 这里首先做的简单一点儿,首先取出来用户第一个操作频率间隔
    for i, row in action_df.iterrows():
        if row.actionTime < (pre_time + 1296000):
            pre_time = row.actionTime

        else:
            pre_time = row.actionTime
            action_time.append(row.actionTime)
    if len(action_time) > 0:
        return np.std(action_time), np.mean(action_time), np.max(action_time), np.min(action_time)
    else:
        return 0, 0, 0, 0


def gen_action_max_diff_vs_count(uid, action_grouped):
    action_df = action_grouped[uid]
    if len(action_df) > 0:
        max_count = action_df.shape[0]
        min_action_time = action_df['actionTime'].iat[0]
        max_action_time = action_df['actionTime'].iat[-1]
        time_diff = max_action_time - min_action_time
        if time_diff != 0:
            return 1.0 * time_diff / max_count
        else:
            return 0
    else:
        return 0


def gen_action_features1(df, action):
    features = pd.DataFrame({'userid': df['userid']})
    action_grouped = dict(list(action.groupby('userid')))

    features['action_tpye_kalman'] = features.apply(lambda row: action_tpye_kalman(row['userid'], action_grouped), axis=1)
    features['actiontype_seq_kalman_real_0'] = features['action_tpye_kalman'].map(lambda x: x[0])
    features['actiontype_seq_kalman_real_1'] = features['action_tpye_kalman'].map(lambda x: x[1])
    features['actiontype_seq_kalman_real_2'] = features['action_tpye_kalman'].map(lambda x: x[2])
    features['actiontype_seq_kalman_std'] = features['action_tpye_kalman'].map(lambda x: x[3])
    features['actiontype_seq_kalman_mean'] = features['action_tpye_kalman'].map(lambda x: x[4])
    features['actiontype_seq_kalman_max'] = features['action_tpye_kalman'].map(lambda x: x[5])
    features['actiontype_seq_kalman_min'] = features['action_tpye_kalman'].map(lambda x: x[6])
    del features['action_tpye_kalman']

    # features['action_time_static_in_hottime'] = features.apply(lambda row: gen_action_in_hot_time(row['userid'], action_grouped), axis=1)
    # features['actiontime_hottime_std'] = features['action_time_static_in_hottime'].map(lambda x: x[0])
    # features['actiontime_hottime_mean'] = features['action_time_static_in_hottime'].map(lambda x: x[1])
    # features['actiontime_hottime_max'] = features['action_time_static_in_hottime'].map(lambda x: x[2])
    # features['actiontime_hottime_min'] = features['action_time_static_in_hottime'].map(lambda x: x[3])
    # del features['action_time_static_in_hottime']

    # features['action_max_diff_vs_count'] = features.apply(lambda row: gen_action_max_diff_vs_count(row['userid'], action_grouped), axis=1)

    return features


def get_peak_data(uid, action_grouped):
    """ action type 操作波峰检测 """
    action_df = action_grouped[uid]
    action_size_df = action_df.groupby(['userid', 'date']).size().reset_index(name='day_click_count')

    # 波峰值,用户最频繁使用app的一天
    action_peak_time = []
    action_peak_count = []

    # 波谷值，表明用户开始逐渐增加使用app的频率
    action_low_peak_time = []
    action_low_peak_count = []

    if len(action_size_df) > 3:
        first_action_count = action_size_df['day_click_count'].iat[0]
        first_action_time = action_size_df['date'].iat[0]
        pre_action_count = first_action_count
        pre_action_time = first_action_time

        upstate = True

        # 如果第一天count大于k,则认为是一个峰值
        if first_action_count > 2:
            action_peak_time.append(first_action_time)
            action_peak_count.append(first_action_count)
        else:
            action_low_peak_time.append(first_action_time)
            action_low_peak_count.append(first_action_count)
        # 寻找峰值以及谷值
        for i, row in action_size_df.iterrows():
            current_count = row.day_click_count
            current_time = row.date
            if upstate:
                if current_count >= pre_action_count:
                    pre_action_count = current_count
                    pre_action_time = current_time
                else:
                    # 这里可以根据实际情况设置一个阈值，当大于一定的数量,再判定为峰值，这个阈值先设为0,后期可以调整
                    if np.abs(pre_action_count - current_count) > 0:
                        action_peak_time.append(pre_action_time)
                        action_peak_count.append(pre_action_count)
                        pre_action_count = current_count
                        pre_action_time = current_time
                        upstate = False
                    else:
                        pre_action_count = current_count
                        pre_action_time = current_time
            else:
                if upstate == False:
                    if current_count <= pre_action_count:
                        pre_action_count = current_count
                        pre_action_time = current_time
                    else:
                        if np.abs(current_count - pre_action_count) > 0:
                            upstate = True

                            action_low_peak_time.append(pre_action_time)
                            action_low_peak_count.append(pre_action_count)
                            pre_action_count = current_count
                            pre_action_time = current_time
        # 这里求一下峰值时间差，使用秒进行表示
        action_time_diff = []
        if len(action_peak_count) > 0:
            for i in range(len(action_peak_time) - 1):
                action_time_diff.append((action_peak_time[i + 1] - action_peak_time[i]).days * 24 * 60 * 60)
            if len(action_time_diff) > 0:
                return len(action_peak_count), np.min(action_time_diff), np.max(action_time_diff), np.std(action_time_diff)
            else:
                return 0, 0, 0, 0,
        else:
            return 0, 0, 0, 0

    else:
        return 0, 0, 0, 0, 0


def gen_action_features2(df, action):
    features = pd.DataFrame({'userid': df['userid']})
    action_grouped = dict(list(action.groupby('userid')))

    # features['action_peak_static'] = features.apply(lambda row: get_peak_data(row['userid'], action_grouped), axis=1)
    # features['action_peak_count'] = features['action_peak_static'].map(lambda x: x[0])
    # features['action_peak_min_difftime'] = features['action_peak_static'].map(lambda x: x[1])
    # features['action_peak_max_difftime'] = features['action_peak_static'].map(lambda x: x[2])
    # features['action_peak_std_difftime'] = features['action_peak_static'].map(lambda x: x[3])
    # del features['action_peak_static']

    return features


def main():
    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')
    orderHistory_train = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_test.csv', encoding='utf8')
    action_train = pd.read_csv(Configure.base_path + 'train/action_train.csv')
    action_test = pd.read_csv(Configure.base_path + 'test/action_test.csv')

    action_train = build_time_features(action_train)
    action_test = build_time_features(action_test)

    orderHistory_train['city'] = orderHistory_train['city'].astype(str)
    orderHistory_test['city'] = orderHistory_test['city'].astype(str)
    orderHistory_train['orderTime'] = pd.to_datetime(orderHistory_train['orderTime'])
    orderHistory_test['orderTime'] = pd.to_datetime(orderHistory_test['orderTime'])

    feature_name = 'advance_order_history_features'
    if not data_utils.is_feature_created(feature_name):
        print('build train advance_order_history_features')
        train_features = gen_history_features(train, orderHistory_train)
        print('build test advance_order_history_features')
        test_features = gen_history_features(test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'advance_action_features'
    if not data_utils.is_feature_created(feature_name):
        print('build train advance_action_features')
        train_features = gen_action_features(train, action_train)
        print('build test advance_action_features')
        test_features = gen_action_features(test, action_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'advance_action_features1'
    if not data_utils.is_feature_created(feature_name):
        print('build train advance_action_features1')
        train_features = gen_action_features1(train, action_train)
        print('build test advance_action_features1')
        test_features = gen_action_features1(test, action_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'advance_action_features2'
    if not data_utils.is_feature_created(feature_name):
        print('build train advance_action_features2')
        train_features = gen_action_features2(train, action_train)
        print('build test advance_action_features2')
        test_features = gen_action_features2(test, action_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    # feature_name = 'advance_action_features3'
    # if not data_utils.is_feature_created(feature_name):
    #     print('add sqg action features')
    #     train_features = pd.read_csv('train_sqg_stage_three.csv')
    #     test_features = pd.read_csv('test_sqg_stage_three.csv')
    #
    #     print('add wxr action features')
    #     with open('wxr_operate_0_train_action_features.pkl', "rb") as f:
    #         action_features_train = cPickle.load(f)
    #     with open('wxr_operate_0_test_action_features.pkl', "rb") as f:
    #         action_features_test = cPickle.load(f)
    #     train_features = pd.merge(train_features, action_features_train, on='userid', how='left')
    #     test_features = pd.merge(test_features, action_features_test, on='userid', how='left')
    #
    #     with open('wxr_operate_1_train_action_features.pkl', "rb") as f:
    #         action_features_train = cPickle.load(f)
    #     with open('wxr_operate_1_test_action_features.pkl', "rb") as f:
    #         action_features_test = cPickle.load(f)
    #     train_features = pd.merge(train_features, action_features_train, on='userid', how='left')
    #     test_features = pd.merge(test_features, action_features_test, on='userid', how='left')
    #
    #     data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== gen advance features ==========")
    main()
