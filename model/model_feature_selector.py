#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-19 上午10:23
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
from utils import data_utils, xgb_feature_selector
from sklearn.model_selection import train_test_split


def xgboost_select_features(train, selected_size, save_tmp_features_path):
    train_df, _ = train_test_split(train, test_size=0.7, random_state=42, shuffle=True, stratify=train['orderType'])
    selector = xgb_feature_selector.XgboostGreedyFeatureSelector(train_df.drop(['orderType'], axis=1),
                                                                 train_df['orderType'])

    xgb_params = {
        'eta': 0.05,
        'colsample_bytree': 0.8,
        'max_depth': 4,
        'subsample': 0.9,
        'lambda': 2.0,
        'scale_pos_weight': 1,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'updater': 'grow_gpu',
        'gpu_id': 0,
        'nthread': -1,
        'silent': 1,
        'booster': 'gbtree'
    }
    base_features = ['actiontimespanlast_5_6', 'last_-1_x_actiontype', 'country_avg_rich', 'timespan_action6tolast',
                     'timespanmin_last_3', 'actionratio_24_59', 'actiontimespancount_5_6',
                     'action_type_56_time_delta_std',
                     'timespan_action24tolast']

    best_subset_features = selector.select_best_subset_features(xgb_params=xgb_params, cv_nfold=4,
                                                                selected_feature_size=int(
                                                                    train_df.shape[1] * selected_size),
                                                                num_boost_round=1000, base_features=base_features,
                                                                save_tmp_features_path=save_tmp_features_path,
                                                                early_stopping_rounds=50, maximize=True,
                                                                stratified=True, shuffle=True)
    return best_subset_features
