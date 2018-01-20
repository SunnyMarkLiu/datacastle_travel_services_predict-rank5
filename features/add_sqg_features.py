#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-18 下午3:51
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


def main():
    feature_name = 'sqg_features'
    # if data_utils.is_feature_created(feature_name):
    #     return

    print('add stage_one_features')
    train_features = pd.read_csv('train_sqg_stage_one_features.csv')
    test_features = pd.read_csv('test_sqg_stage_one_features.csv')

    used_features = ['userid', 'rate_all_good',
                     'big_than_mean', 'rate_user_click',
                     'click_1_rate', 'click_2_rate', 'click_3_rate', 'click_4_rate',
                     'click_5_rate', 'click_6_rate', 'click_7_rate', 'click_8_rate',
                     'click_9_rate', 'less_than_4_rate', 'more_than_5_rate',
                     'click_1_num', 'click_2_num', 'click_3_num', 'click_4_num',
                     'click_5_num', 'click_6_num', 'click_7_num', 'click_8_num',
                     'click_9_num', 'less_than_4_num', 'more_than_6_num',
                     'action_time_min', 'action_time_max', 'diff_time_num_click',
                     'max_diff_days', 'rate_diff_num_time_in_max',
                     'diff_max_x', 'diff_median_x', 'diff_min_x', 'diff_max_y', 'diff_median_y', 'diff_min_y',
                     'rate_orderNum_in_clickNum', 'rate_goodNum_in_clickNum', 'lessthan4_Num_minus_more_than_6_num',
                     'order_time_max', 'order_time_min', 'order_time_median']

    train_features = train_features[used_features]
    test_features = test_features[used_features]

    print('save ', feature_name)
    data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== merge guo-ge features ==========")
    main()
