#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-14 下午8:23
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings
warnings.filterwarnings('ignore')

import cPickle
import numpy as np
import pandas as pd
from conf.configure import Configure
from utils import data_utils


def main():
    feature_name = 'wxr_features'
    if data_utils.is_feature_created(feature_name):
        return

    print('add comment score features')
    with open('wxr_train_comment_features.pkl', "rb") as f:
        user_comment_train = cPickle.load(f)
    with open('wxr_test_comment_features.pkl', "rb") as f:
        user_comment_test = cPickle.load(f)

    user_comment_train.fillna(-1, inplace=True)
    user_comment_test.fillna(-1, inplace=True)

    train_features = user_comment_train
    test_features = user_comment_test

    # print('add user_info features')
    # with open('wxr_train_user_info_features.pkl', "rb") as f:
    #     train_user_info = cPickle.load(f)
    # with open('wxr_test_user_info_features.pkl', "rb") as f:
    #     test_user_info = cPickle.load(f)
    # train_user_info.drop(['gender', 'province', 'age'], axis=1, inplace=True)
    # test_user_info.drop(['gender', 'province', 'age'], axis=1, inplace=True)
    #
    # train_features = train_features.merge(train_user_info, on='userid', how='left')
    # test_features = test_features.merge(test_user_info, on='userid', how='left')

    print('add history features')
    with open('wxr_operate_4_train_order_history_features.pkl', "rb") as f:
        history_features_train = cPickle.load(f)
    with open('wxr_operate_4_test_order_history_features.pkl', "rb") as f:
        history_features_test = cPickle.load(f)

    use_features = ['userid','avg_days_between_order', 'days_ratio_since_last_order','city_num', 'country_num', 'continent_num',
                    'city_rich', 'city_avg_rich', 'country_rich', 'country_avg_rich', 'histord_time_last_1_year',
                    'histord_time_last_1_month', 'histord_sum_cont1', 'histord_sum_cont2', 'histord_sum_cont3',
                    'histord_sum_cont4', 'histord_sum_cont5', 'timespan_lastord_1_2', 'timespan_lastord_2_3']
    history_features_train = history_features_train[use_features]
    history_features_test = history_features_test[use_features]
    train_features = train_features.merge(history_features_train, on='userid', how='left')
    test_features = test_features.merge(history_features_test, on='userid', how='left')

    print('add action features')
    with open('wxr_operate_3_train_action_features.pkl', "rb") as f:
        action_features_train = cPickle.load(f)
    with open('wxr_operate_3_test_action_features.pkl', "rb") as f:
        action_features_test = cPickle.load(f)
    use_features = ['userid', 'avg_browse_num_after_last_order', 'browse_num_after_last_order',
                    'operate_num_after_last_order', 'avg_operate_num_after_last_order', 'open_num_after_last_order',
                    'action_1_num_after_last_order', 'action_2_num_after_last_order', 'action_3_num_after_last_order',
                    'action_4_num_after_last_order', 'action_5_num_after_last_order', 'action_6_num_after_last_order',
                    'action_7_num_after_last_order', 'action_8_num_after_last_order', 'action_9_num_after_last_order']
    action_features_train = action_features_train[use_features]
    action_features_test = action_features_test[use_features]
    train_features = train_features.merge(action_features_train, on='userid', how='left')
    test_features = test_features.merge(action_features_test, on='userid', how='left')

    print('add someother features')
    some_other_train = pd.read_csv('some_other_train_features.csv')
    some_other_test  = pd.read_csv('some_other_test_features.csv')
    train_features = train_features.merge(some_other_train, on='userid', how='left')
    test_features = test_features.merge(some_other_test, on='userid', how='left')

    print('save ', feature_name)
    data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== merge wxr features ==========")
    main()
