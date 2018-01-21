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

from utils import xgb_feature_selector
from sklearn.model_selection import train_test_split


def xgboost_select_features(train, selected_size, save_tmp_features_path, base_features, decrease_auc_threshold):
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
        'gpu_id': 2,
        'nthread': -1,
        'silent': 1,
        'booster': 'gbtree'
    }

    best_subset_features = selector.select_best_subset_features(xgb_params=xgb_params, cv_nfold=3,
                                                                selected_feature_size=int(train_df.shape[1] * selected_size),
                                                                num_boost_round=1000, base_features=base_features,
                                                                thread_size=10,
                                                                save_tmp_features_path=save_tmp_features_path,
                                                                early_stopping_rounds=50,
                                                                stratified=True, shuffle=True,
                                                                decrease_auc_threshold=decrease_auc_threshold)
    return best_subset_features


def xgboost_remove_features(train, remove_ratio, save_removed_features_path, decrease_auc_threshold):
    train_df, _ = train_test_split(train, test_size=0.7, random_state=42, shuffle=True, stratify=train['orderType'])
    selector = xgb_feature_selector.XgboostGreedyFeatureSelector(train_df.drop(['orderType'], axis=1),
                                                                 train_df['orderType'])
    print('train data:', train_df.shape[0], ' order_type1 vs order_type0:',
          1.0 * sum(train_df['orderType']) / (train_df.shape[0] - sum(train_df['orderType'])))

    xgb_params = {
        'eta': 0.1,
        'min_child_weight': 20,
        'colsample_bytree': 0.5,
        'max_depth': 3,
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

    removed_features = selector.remove_worst_features(xgb_params=xgb_params, cv_nfold=3, remove_ratio=remove_ratio,
                                                      num_boost_round=1000, early_stopping_rounds=50, thread_size=10,
                                                      save_removed_features_path=save_removed_features_path,
                                                      stratified=True, shuffle=True,
                                                      decrease_auc_threshold=decrease_auc_threshold)
    return removed_features
