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

    #给trade表打标签，若id在login表中，则打标签为1，否则为0
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in df_ids).astype(int)

    features['history_order_type_sum'] = features.apply(lambda row: order_type_sum(row['userid'], history_grouped, row['has_history_flag']), axis=1)
    features['history_order_type_sum_lg0'] = features['history_order_type_sum'].map(lambda x: int(x>0))
    del features['history_order_type_sum']

    del features['has_history_flag']
    return features


def main():
    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')
    orderHistory_train = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_test.csv', encoding='utf8')

    orderHistory_train['city'] = orderHistory_train['city'].astype(str)
    orderHistory_test['city'] = orderHistory_test['city'].astype(str)
    orderHistory_train['orderTime'] = pd.to_datetime(orderHistory_train['orderTime'])
    orderHistory_test['orderTime'] = pd.to_datetime(orderHistory_test['orderTime'])

    feature_name = 'order_history_features'
    if data_utils.is_feature_created(feature_name):
        print('build train order_history_features')
        train_features = gen_history_features(train, orderHistory_train)
        print('build test order_history_features')
        test_features = gen_history_features(test, orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== gen history features2 ==========")
    main()
