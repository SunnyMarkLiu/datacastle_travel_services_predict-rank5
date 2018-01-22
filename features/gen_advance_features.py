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
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
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

    """ history_order_type_sum 和 history_order_type_sum_lg0 后期再加上"""
    # features['history_order_type_sum'] = features.apply(lambda row: order_type_sum(row['userid'], history_grouped, row['has_history_flag']), axis=1)
    # features['history_order_type_sum_lg0'] = features['history_order_type_sum'].map(lambda x: int(x > 0))
    # del features['history_order_type_sum']

    del features['has_history_flag']
    return features


def gen_actiontype_sequence(uid, action_grouped):
    """ 用户订单历史结果构成的序列 """
    df = action_grouped[uid]
    sequence = ' '.join(df['actionType'].astype(str).values.tolist())
    return sequence


def gen_action_features(df, action):
    features = pd.DataFrame({'userid': df['userid']})
    action_grouped = dict(list(action.groupby('userid')))

    features['action_sequence'] = features.apply(lambda row: gen_actiontype_sequence(row['userid'], action_grouped), axis=1)
    action_texts = features['action_sequence'].values
    vectorizer = TfidfVectorizer(stop_words=None)
    dtm = vectorizer.fit_transform(action_texts)
    # vocab = np.array(vectorizer.get_feature_names())
    # clf = decomposition.NMF(n_components=20, random_state=1)
    # doctopic = clf.fit_transform(dtm)
    # doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
    doctopic = pd.DataFrame(dtm.toarray(), columns=['topic_model_{}'.format(i) for i in range(9)])
    features = pd.concat([features, doctopic], axis=1)

    del features['action_sequence']

    return features


def main():
    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')
    orderHistory_train = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_test.csv', encoding='utf8')
    action_train = pd.read_csv(Configure.cleaned_path + 'cleaned_action_train.csv')
    action_test = pd.read_csv(Configure.cleaned_path + 'cleaned_action_test.csv')

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


if __name__ == "__main__":
    print("========== gen advance features ==========")
    main()
