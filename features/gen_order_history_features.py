#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-22 下午7:23
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
import numpy as np
import pandas as pd
from pypinyin import lazy_pinyin
from sklearn.preprocessing import LabelEncoder
from conf.configure import Configure
from utils import data_utils


def check_last_time_order_info(uid, userid_grouped, flag, check_name, last_time=1):
    """ 最近的一次交易的具体信息 check_name """
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    if df.shape[0] < last_time:
        return -1
    else:
        return df.iloc[-last_time][check_name]


def pre_days_order_count(uid, userid_grouped, flag, days):
    """ 往前 days 的 order 数量 """
    if flag == 0:
        return 0

    df = userid_grouped[uid]
    df = df.loc[df['days_from_now'] < days]
    return df.shape[0]


def pre_days_checkname_diff_count(uid, userid_grouped, flag, days, check_name):
    """ 往前 days 的 order 的不同 check_name 数量 """
    if flag == 0:
        return 0

    df = userid_grouped[uid]
    df = df.loc[df['days_from_now'] < days]
    if df.shape[0] == 0:
        return 0
    else:
        return len(df[check_name].unique())


def year_order_count(uid, userid_grouped, flag, year):
    """ 2016年的 order 的不同 check_name 数量 """
    if flag == 0:
        return 0

    df = userid_grouped[uid]
    df = df.loc[df['order_year'] == year]
    return df.shape[0]


def year_checkname_diff_count(uid, userid_grouped, flag, year, check_name):
    """ year 的 order 的不同 check_name 数量 """
    if flag == 0:
        return 0

    df = userid_grouped[uid]
    df = df.loc[df['order_year'] == year]
    if df.shape[0] == 0:
        return 0
    else:
        return len(df[check_name].unique())


def year_order_month_count(uid, userid_grouped, flag, year):
    """ 每年去了几个月份 """
    if flag == 0:
        return 0

    df = userid_grouped[uid]
    df = df.loc[df['order_year'] == year]
    if df.shape[0] == 0:
        return 0
    else:
        return len(df['order_month'].unique())


def year_order_month_most(uid, userid_grouped, flag, year):
    """ 每年一个月去的最多的次数 """
    if flag == 0:
        return 0

    df = userid_grouped[uid]
    df = df.loc[df['order_year'] == year]
    df = df.groupby(['order_month']).count()['orderTime'].reset_index()
    if df.shape[0] == 0:
        return 0
    else:
        return df['orderTime'].max()


def year_most_order_month(uid, userid_grouped, flag, year):
    """ 每年去的最多次数的月份 """
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    df = df.loc[df['order_year'] == year]
    df = df.groupby(['order_month']).count()['orderTime'].reset_index()
    if df.shape[0] == 0:
        return -1
    else:
        return df.sort_values(by='orderTime', ascending=False)['order_month'].values[0]


def year_good_order_count(uid, userid_grouped, flag, year):
    """ 每年精品订单数量 """
    if flag == 0:
        return 0

    df = userid_grouped[uid]
    df = df.loc[df['order_year'] == year]
    return sum(df['orderType'])


def last_time_checkname_ratio(uid, userid_grouped, flag, check_name):
    """ 最后一次 checkname 的占比 """
    if flag == 0:
        return 0

    df = userid_grouped[uid]
    last_check_name = df.iloc[-1][check_name]
    last_count = df[check_name].tolist().count(last_check_name)
    return 1.0 * last_count / df.shape[0]


def build_order_history_features(df, history):
    features = pd.DataFrame({'userid': df['userid']})

    df_ids = history['userid'].unique()
    userid_grouped = dict(list(history.groupby('userid')))

    #给trade表打标签，若id在login表中，则打标签为1，否则为0
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in df_ids).astype(int)

    print("基本特征")
    # build_order_history_features2 函数中提交提取，冗余
    # 最近的一次交易的 orderType
    # features['last_time_orderType'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'orderType', 1), axis=1)
    # 倒数第二个 orderType
    # features['last_2_time_orderType'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'orderType', 2), axis=1)
    # features['last_3_time_orderType'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'orderType',3), axis=1)
    # 倒数第二次距离现在的时间
    # features['last_2_time_days_from_now'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'days_from_now', 2), axis=1)
    # features['last_3_time_days_from_now'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'days_from_now', 3), axis=1)

    # 最近的一次交易的 days_from_now, order_year, order_month, order_day, order_weekofyear, order_weekday
    features['last_time_days_from_now'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'days_from_now'), axis=1)
    features['last_time_order_year'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'order_year'), axis=1)
    features['last_time_order_month'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'order_month'), axis=1)
    features['last_time_order_day'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'order_day'), axis=1)
    features['last_time_order_weekofyear'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'order_weekofyear'), axis=1)
    features['last_time_order_weekday'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'order_weekday'), axis=1)
    features['last_time_continent'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'continent'), axis=1)
    features['last_time_country'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'country'), axis=1)
    features['last_time_city'] = features.apply(lambda row: check_last_time_order_info(row['userid'], userid_grouped, row['has_history_flag'], 'city'), axis=1)

    print("计数特征")
    # 往前 90days 的计数特征
    features['pre_90days_order_count'] = features.apply(lambda row: pre_days_order_count(row['userid'], userid_grouped, row['has_history_flag'], 90), axis=1)
    features['pre_90days_order_continent_count'] = features.apply(lambda row: pre_days_checkname_diff_count(row['userid'], userid_grouped, row['has_history_flag'], 90, 'continent'), axis=1)
    features['pre_90days_order_country_count'] = features.apply(lambda row: pre_days_checkname_diff_count(row['userid'], userid_grouped, row['has_history_flag'], 90, 'country'), axis=1)
    features['pre_90days_order_city_count'] = features.apply(lambda row: pre_days_checkname_diff_count(row['userid'], userid_grouped, row['has_history_flag'], 90, 'city'), axis=1)

    features['2016_order_count'] = features.apply(lambda row: year_order_count(row['userid'], userid_grouped, row['has_history_flag'], 2016), axis=1)
    features['2017_order_count'] = features.apply(lambda row: year_order_count(row['userid'], userid_grouped, row['has_history_flag'], 2017), axis=1)
    # features['order_count_diff'] = features['2016_order_count'] - features['2017_order_count']
    # features['2016_order_continent_count'] = features.apply(lambda row: year_checkname_diff_count(row['userid'], userid_grouped, row['has_history_flag'], 2016, 'continent'), axis=1)
    # features['2016_order_country_count'] = features.apply(lambda row: year_checkname_diff_count(row['userid'], userid_grouped, row['has_history_flag'], 2016, 'country'), axis=1)
    # features['2016_order_city_count'] = features.apply(lambda row: year_checkname_diff_count(row['userid'], userid_grouped, row['has_history_flag'], 2016, 'city'), axis=1)
    features['2017_order_continent_count'] = features.apply(lambda row: year_checkname_diff_count(row['userid'], userid_grouped, row['has_history_flag'], 2017, 'continent'), axis=1)
    features['2017_order_country_count'] = features.apply(lambda row: year_checkname_diff_count(row['userid'], userid_grouped, row['has_history_flag'], 2017, 'country'), axis=1)
    features['2017_order_city_count'] = features.apply(lambda row: year_checkname_diff_count(row['userid'], userid_grouped, row['has_history_flag'], 2017, 'city'), axis=1)
    # 是否 2016 年和 2017 年都有 order
    features['both_year_has_order'] = features.apply(lambda row: (row['2016_order_count'] > 0) & (row['2017_order_count'] > 0), axis=1).astype(int)
    # 每年去了几个月份
    features['2016_order_month_count'] = features.apply(lambda row: year_order_month_count(row['userid'], userid_grouped, row['has_history_flag'], 2016), axis=1)
    features['2017_order_month_count'] = features.apply(lambda row: year_order_month_count(row['userid'], userid_grouped, row['has_history_flag'], 2017), axis=1)
    # 每年一个月去的最多的次数
    # features['2016_order_month_most'] = features.apply(lambda row: year_order_month_most(row['userid'], userid_grouped, row['has_history_flag'], 2016), axis=1)
    # features['2017_most_order_month'] = features.apply(lambda row: year_order_month_most(row['userid'], userid_grouped, row['has_history_flag'], 2017), axis=1)
    # 每年去的最多的月份
    # features['2016_most_order_month'] = features.apply(lambda row: year_most_order_month(row['userid'], userid_grouped, row['has_history_flag'], 2016), axis=1)
    # features['2017_most_order_month'] = features.apply(lambda row: year_most_order_month(row['userid'], userid_grouped, row['has_history_flag'], 2017), axis=1)

    print('比率特征')
    # 用户总订单数、精品订单数、精品订单比例
    features['2016_good_order_count'] = features.apply(lambda row: year_good_order_count(row['userid'], userid_grouped, row['has_history_flag'], 2016), axis=1)
    features['2016_good_order_ratio'] = features.apply(lambda row: row['2016_good_order_count'] / row['2016_order_count'] if row['2016_order_count'] != 0 else 0, axis=1)

    features['2017_good_order_count'] = features.apply(lambda row: year_good_order_count(row['userid'], userid_grouped, row['has_history_flag'], 2017), axis=1)
    features['2017_good_order_ratio'] = features.apply(lambda row: row['2017_good_order_count'] / row['2017_order_count'] if row['2017_order_count'] != 0 else 0, axis=1)

    features['total_order_count'] = features['2016_order_count'] + features['2017_order_count']
    features['total_good_order_count'] = features['2016_good_order_count'] + features['2017_good_order_count']
    features['total_good_order_ratio'] = features.apply(lambda row: row['total_good_order_count'] / row['total_order_count'] if row['total_order_count'] != 0 else 0, axis=1)
    # has_good_order 强特！！
    features['has_good_order'] = (features['total_good_order_ratio'] > 0).astype(int)
    features.drop(['2016_good_order_count', '2017_good_order_count', 'total_order_count', 'total_good_order_count'], axis=1, inplace=True)

    # cv 变差一点点，不到1个万分点
    # print('最后一次 order 的 check_name 的占比') #（未测试）
    # features['last_time_order_year_ratio'] = features.apply(lambda row: last_time_checkname_ratio(row['userid'], userid_grouped, row['has_history_flag'], 'order_year'), axis=1)
    # features['last_time_order_month_ratio'] = features.apply(lambda row: last_time_checkname_ratio(row['userid'], userid_grouped, row['has_history_flag'], 'order_month'), axis=1)
    # features['last_time_order_day_ratio'] = features.apply(lambda row: last_time_checkname_ratio(row['userid'], userid_grouped, row['has_history_flag'], 'order_day'), axis=1)
    # features['last_time_order_weekofyear_ratio'] = features.apply(lambda row: last_time_checkname_ratio(row['userid'], userid_grouped, row['has_history_flag'], 'order_weekofyear'), axis=1)
    # features['last_time_order_weekday_ratio'] = features.apply(lambda row: last_time_checkname_ratio(row['userid'], userid_grouped, row['has_history_flag'], 'order_weekday'), axis=1)
    # features['last_time_continent_ratio'] = features.apply(lambda row: last_time_checkname_ratio(row['userid'], userid_grouped, row['has_history_flag'], 'continent'), axis=1)
    # features['last_time_country_ratio'] = features.apply(lambda row: last_time_checkname_ratio(row['userid'], userid_grouped, row['has_history_flag'], 'country'), axis=1)
    # features['last_time_city_ratio'] = features.apply(lambda row: last_time_checkname_ratio(row['userid'], userid_grouped, row['has_history_flag'], 'city'), axis=1)

    return features


def build_time_category_encode(history):
    history['orderTime'] = pd.to_datetime(history['orderTime'], unit='s')

    # 训练集和测试集最后一天是 2017-09-11
    now = datetime.datetime(2017, 9, 12)
    history['days_from_now'] = history['orderTime'].map(lambda order: (now - order).days)
    history['order_year'] = history['orderTime'].dt.year
    history['order_month'] = history['orderTime'].dt.month
    history['order_year_month'] = history['order_year'] * 100 + history['order_month']
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

    return history


def father_son_order_statistic(uid, userid_grouped, flag):
    if flag == 0:
        return -1, -1

    df = userid_grouped[uid]
    if len(set(df['orderTime'])) < df.shape[0]: # 存在子父订单
        start = -1
        count = 0
        for i in range(df.shape[0] - 2):
            if df['orderTime'].iat[i] == df['orderTime'].iat[i+1]:
                if count == 0:
                    start = i
                count += 1
        df = df.iloc[start: start+count]
        if df.shape[0] == 0:
            return -1, -1
        else:
            order_type0_count = df[df['orderType'] == 0].shape[0]
            order_type1_count = df[df['orderType'] == 0].shape[0]
            order_type0_ratio = 1.0* order_type0_count / df.shape[0]
            order_type1_ratio = 1.0* order_type1_count / df.shape[0]
            return order_type0_ratio, order_type1_ratio
    else:
        return -1, -1

def year_first_last_order_history_type(uid, history_grouped, flag, year=2017):
    """  每年第一次和最后一次订单的 ordertype """
    if flag == 0:
        return -1, -1

    df = history_grouped[uid]
    df = df[df['order_year'] == year]

    if df.shape[0] < 1:
        return -1, -1

    first1_ordertype = df['orderType'].iat[0]
    last1_ordertype = df['orderType'].iat[-1]
    return first1_ordertype, last1_ordertype


def build_order_history_features2(df, history):
    features = pd.DataFrame({'userid': df['userid']})

    history_uids = history['userid'].unique()
    history_grouped = dict(list(history.groupby('userid')))

    #给trade表打标签，若id在login表中，则打标签为1，否则为0
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in history_uids).astype(int)

    # # 子父订单统计特征
    # features['father_son_order_statistic'] = features.apply(lambda row: father_son_order_statistic(row['userid'], history_grouped, row['has_history_flag']), axis=1)
    # # features['has_father_son_order'] = features['father_son_order_statistic'].map(lambda x: x[0])
    # # features['father_son_order_order_type0_count'] = features['father_son_order_statistic'].map(lambda x: x[1])
    # # features['father_son_order_order_type1_count'] = features['father_son_order_statistic'].map(lambda x: x[2])
    # features['father_son_order_order_type0_ratio'] = features['father_son_order_statistic'].map(lambda x: x[0])
    # features['father_son_order_order_type1_ratio'] = features['father_son_order_statistic'].map(lambda x: x[1])
    # del features['father_son_order_statistic']

    print('强特:2016_2017_first_last_ordertype')
    features['2017_first_last_order_history_type'] = features.apply(lambda row: year_first_last_order_history_type(row['userid'], history_grouped, row['has_history_flag'], year=2017), axis=1)
    features['2017_first_order_history_type'] = features['2017_first_last_order_history_type'].map(lambda x: x[0])
    features['2017_last_order_history_type'] = features['2017_first_last_order_history_type'].map(lambda x: x[1])
    features['2016_first_last_order_history_type'] = features.apply(lambda row: year_first_last_order_history_type(row['userid'], history_grouped, row['has_history_flag'], year=2016), axis=1)
    features['2016_first_order_history_type'] = features['2016_first_last_order_history_type'].map(lambda x: x[0])
    features['2016_last_order_history_type'] = features['2016_first_last_order_history_type'].map(lambda x: x[1])

    features['2016_2017_first_last_ordertype'] = ((features['2016_first_order_history_type'] == 1) | (features['2017_first_order_history_type'] == 1) |
                                                  (features['2016_last_order_history_type'] == 1) | (features['2017_last_order_history_type'] == 1)).astype(int)

    features.drop(['2017_first_last_order_history_type', '2017_first_order_history_type', '2017_last_order_history_type',
                   '2016_first_last_order_history_type', '2016_first_order_history_type', '2016_last_order_history_type'], axis=1, inplace=True)

    print('每年每个月份订单的统计')
    df = history.groupby(by=['userid', 'order_year_month']).count().reset_index()[['userid', 'order_year_month', 'orderid']].rename(columns={'orderid': 'year_month_order_count'})
    df = df.pivot('userid', 'order_year_month', 'year_month_order_count').reset_index().fillna(0)
    df.columns = df.columns.astype(str)
    df.drop(['201709', '201708', '201707', '201701', '201705'], axis=1, inplace=True)
    features = features.merge(df, on='userid', how='left')
    del features['has_history_flag']
    return features


def history_city_hot_statistic(uid, history_grouped, flag, hot_df, column):
    if flag == 0:
        return -1, -1, -1, -1

    df = history_grouped[uid]
    citys = df[column].values
    hots = []
    for c in citys:
        hots.append(hot_df[hot_df[column] == c]['hot'].values[0])

    hots = np.array(hots)
    return np.mean(hots), np.max(hots), np.min(hots), np.std(hots)


def last_order_location_hot(uid, history_grouped, flag, hot_df, column):
    if flag == 0:
        return -1

    df = history_grouped[uid]
    last_hot = hot_df[hot_df[column] == df[column].iat[-1]]['hot'].values[0]
    return last_hot


def build_order_history_features3(df, orderHistory, history):
    """ 热度分析 """
    city_hot = orderHistory.groupby(['city']).count()['userid'].reset_index().rename(columns={'userid': 'hot'})
    city_hot['hot'] = city_hot['hot'].astype(float) / sum(city_hot['hot'])
    country_hot = orderHistory.groupby(['country']).count()['userid'].reset_index().rename(columns={'userid': 'hot'})
    country_hot['hot'] = country_hot['hot'].astype(float) / sum(country_hot['hot'])
    continent_hot = orderHistory.groupby(['continent']).count()['userid'].reset_index().rename(columns={'userid': 'hot'})
    continent_hot['hot'] = continent_hot['hot'].astype(float) / sum(continent_hot['hot'])

    features = pd.DataFrame({'userid': df['userid']})
    history_uids = history['userid'].unique()
    history_grouped = dict(list(history.groupby('userid')))
    #给trade表打标签，若id在login表中，则打标签为1，否则为0
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in history_uids).astype(int)

    # features['history_city_hot_statistic'] = features.apply(lambda row: history_city_hot_statistic(row['userid'], history_grouped, row['has_history_flag'], city_hot, 'city'), axis=1)
    # features['history_city_hot_mean'] = features['history_city_hot_statistic'].map(lambda x:x[0])
    # features['history_city_hot_max'] = features['history_city_hot_statistic'].map(lambda x:x[1])
    # features['history_city_hot_min'] = features['history_city_hot_statistic'].map(lambda x:x[2])
    # features['history_city_hot_std'] = features['history_city_hot_statistic'].map(lambda x:x[3])
    # del features['history_city_hot_statistic']
    # features['history_country_hot_statistic'] = features.apply(lambda row: history_city_hot_statistic(row['userid'], history_grouped, row['has_history_flag'], country_hot, 'country'), axis=1)
    # features['history_country_hot_mean'] = features['history_country_hot_statistic'].map(lambda x:x[0])
    # features['history_country_hot_max'] = features['history_country_hot_statistic'].map(lambda x:x[1])
    # features['history_country_hot_min'] = features['history_country_hot_statistic'].map(lambda x:x[2])
    # features['history_country_hot_std'] = features['history_country_hot_statistic'].map(lambda x:x[3])
    # del features['history_country_hot_statistic']
    # features['history_continent_hot_statistic'] = features.apply(lambda row: history_city_hot_statistic(row['userid'], history_grouped, row['has_history_flag'], continent_hot, 'continent'), axis=1)
    # features['history_continent_hot_mean'] = features['history_continent_hot_statistic'].map(lambda x:x[0])
    # features['history_continent_hot_max'] = features['history_continent_hot_statistic'].map(lambda x:x[1])
    # features['history_continent_hot_min'] = features['history_continent_hot_statistic'].map(lambda x:x[2])
    # features['history_continent_hot_std'] = features['history_continent_hot_statistic'].map(lambda x:x[3])
    # del features['history_continent_hot_statistic']

    # 只有 last_order_city_hot 线上A榜提升
    features['last_order_city_hot'] = features.apply(lambda row: last_order_location_hot(row['userid'], history_grouped, row['has_history_flag'], city_hot, 'city'), axis=1)
    # features['last_order_country_hot'] = features.apply(lambda row: last_order_location_hot(row['userid'], history_grouped, row['has_history_flag'], country_hot, 'country'), axis=1)
    # features['last_order_continent_hot'] = features.apply(lambda row: last_order_location_hot(row['userid'], history_grouped, row['has_history_flag'], continent_hot, 'continent'), axis=1)

    del features['has_history_flag']
    return features


def multi_order_has_good_order(uid, history_grouped, flag):
    """ 多次订单并且有精品的老用户 """
    if flag == 0:
        return 0

    df = history_grouped[uid]
    if (df.shape[0] > 1) and sum(df['orderType']) > 0:
        return 1

    return 0


def build_order_history_features4(df, history):
    features = pd.DataFrame({'userid': df['userid']})

    history_uids = history['userid'].unique()
    history_grouped = dict(list(history.groupby('userid')))

    #给trade表打标签，若id在login表中，则打标签为1，否则为0
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in history_uids).astype(int)

    # 多次订单并且有精品的老用户
    features['multi_order_has_good_order'] = features.apply(lambda row: multi_order_has_good_order(row['userid'], history_grouped, row['has_history_flag']), axis=1)

    # 多次订单并且都没有精品订单
    del features['has_history_flag']
    return features


def main():
    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')
    orderHistory_train = pd.read_csv(Configure.base_path + 'train/orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.base_path + 'test/orderHistory_test.csv', encoding='utf8')
    orderHistory_train = build_time_category_encode(orderHistory_train)
    orderHistory_test = build_time_category_encode(orderHistory_test)
    orderHistory_train.to_csv(Configure.cleaned_path + 'cleaned_orderHistory_train.csv', index=False,
                              columns=orderHistory_train.columns)
    orderHistory_test.to_csv(Configure.cleaned_path + 'cleaned_orderHistory_test.csv', index=False,
                             columns=orderHistory_test.columns)

    feature_name = 'user_order_history_features'
    if not data_utils.is_feature_created(feature_name):
        print('build train user_order_history_features')
        train_features = build_order_history_features(train, orderHistory_train)
        print('build test user_order_history_features')
        test_features = build_order_history_features(test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'user_order_history_features2'
    if not data_utils.is_feature_created(feature_name):
        print('build train user_order_history_features2')
        train_features = build_order_history_features2(train, orderHistory_train)
        print('build test user_order_history_features2')
        test_features = build_order_history_features2(test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'user_order_history_features3'
    if not data_utils.is_feature_created(feature_name):
        orderHistory = pd.concat([orderHistory_train, orderHistory_test])
        print('build train user_order_history_features3')
        train_features = build_order_history_features3(train, orderHistory, orderHistory_train)
        print('build test user_order_history_features3')
        test_features = build_order_history_features3(test, orderHistory, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'user_order_history_features4'
    if not data_utils.is_feature_created(feature_name):
        print('build train user_order_history_features4')
        train_features = build_order_history_features4(train, orderHistory_train)
        print('build test user_order_history_features4')
        test_features = build_order_history_features4(test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 构造用户历史订单特征 ==========")
    main()
