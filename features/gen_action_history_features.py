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
            return 0, 0, 0, 0, 0, 0, 0, 0

        actionTypes = sub_action_df['actionType'].tolist()
        return len(actionTypes), actionTypes.count('browse_product'), actionTypes.count('fillin_form5'), \
               actionTypes.count('fillin_form6'), actionTypes.count('fillin_form7'), actionTypes.count('open_app'), \
               actionTypes.count('pay_money'), actionTypes.count('submit_order')

    h_df = history_grouped[uid]
    if a_df.shape[0] == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0
    else:
        last_order_time = h_df.iloc[-1]['orderTime']
        sub_action_df = a_df[a_df['actionTime'] > last_order_time]
        if sub_action_df.shape[0] == 0:
            return 0, 0, 0, 0, 0, 0, 0, 0

        actionTypes = sub_action_df['actionType'].tolist()
        return len(actionTypes), actionTypes.count('browse_product'), actionTypes.count('fillin_form5'), \
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
    features['last_time_order_now_action_fillin_form5_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[2])
    features['last_time_order_now_action_fillin_form6_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[3])
    features['last_time_order_now_action_fillin_form7_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[4])
    features['last_time_order_now_action_open_app_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[5])
    features['last_time_order_now_action_pay_money_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[6])
    features['last_time_order_now_action_submit_order_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[7])
    del features['last_time_order_now_action_info_count']
    # 是否有支付操作和提交订单操作
    features['last_time_order_now_has_paied_money'] = features['last_time_order_now_action_pay_money_count'].map(lambda x: int(x > 0))
    features['last_time_order_now_has_submited_order'] = features['last_time_order_now_action_submit_order_count'].map(lambda x: int(x > 0))

    print('距离最近的 action type 的时间距离')
    features['last_action_open_app_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'open_app'), axis=1)
    features['last_action_browse_product_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'browse_product'), axis=1)
    features['last_action_fillin_form5_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'fillin_form5'), axis=1)
    features['last_action_fillin_form6_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'fillin_form6'), axis=1)
    features['last_action_fillin_form7_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'fillin_form7'), axis=1)
    features['last_action_submit_order_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'submit_order'), axis=1)
    features['last_action_pay_money_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'pay_money'), axis=1)

    print('距离最近的倒数第二次 action type 的时间距离')
    features['last2_action_open_app_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'open_app', 2), axis=1)
    features['last2_action_browse_product_time_delta'] = features.apply(lambda row: last_action_type_time_delta(row['userid'], action_grouped, 'browse_product', 2), axis=1)
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
    features['last_open_app_action_count'] = features.apply(lambda row: last_actiontype_action_count(row['userid'], action_grouped, 'open_app'), axis=1)

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
    # if not data_utils.is_feature_created(feature_name):
    print('build train action history features2')
    train_features = build_action_history_features2(train, action_train, orderHistory_train)
    print('build test action history features2')
    test_features = build_action_history_features2(test, action_test, orderHistory_test)

    print('save ', feature_name)
    data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 结合 action、 history 和 comment 提取历史特征 ==========")
    main()
