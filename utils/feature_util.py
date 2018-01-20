#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-14 下午9:04
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
# remove warnings
import warnings
warnings.filterwarnings('ignore')
import dataframe_util


def feature_check_before_modeling(train, test, check_feature_names):
    """
    It might save you some headache to check your train and test feature
    distributions before modeling. Usually in kaggle competitions train 
    and test sets are iid. If there is huge differenc between train and 
    test set than probably you have a bug in your feature extraction pipeline.
    """
    print '======== train data missing info ========'
    print dataframe_util.contains_null(train)
    print '======== test data missing info ========'
    print dataframe_util.contains_null(test)

    feature_stats = pd.DataFrame({'feature': check_feature_names})
    feature_stats.loc[:, 'train_mean'] = np.nanmean(train[check_feature_names].values, axis=0).round(4)
    feature_stats.loc[:, 'test_mean'] = np.nanmean(test[check_feature_names].values, axis=0).round(4)
    feature_stats.loc[:, 'train_std'] = np.nanstd(train[check_feature_names].values, axis=0).round(4)
    feature_stats.loc[:, 'test_std'] = np.nanstd(test[check_feature_names].values, axis=0).round(4)
    feature_stats.loc[:, 'train_nan_mean_ratio'] = np.mean(np.isnan(train[check_feature_names].values), axis=0).round(3)
    feature_stats.loc[:, 'test_nan_mean_ratio'] = np.mean(np.isnan(test[check_feature_names].values), axis=0).round(3)
    feature_stats.loc[:, 'train_test_mean_diff'] = np.abs(
        feature_stats['train_mean'] - feature_stats['test_mean']) / np.abs(
        feature_stats['train_std'] + feature_stats['test_std']) * 2
    feature_stats.loc[:, 'train_test_nan_mean_ratio_diff'] = np.abs(feature_stats['train_nan_mean_ratio'] - feature_stats['test_nan_mean_ratio'])

    print '======== train test mean difference (ignore Nans) ========'
    feature_stats = feature_stats.sort_values(by='train_test_mean_diff')
    print feature_stats[['feature', 'train_test_mean_diff']].tail()

    print '======== train test mean difference (ignore Nans) ========'
    feature_stats = feature_stats.sort_values(by='train_test_nan_mean_ratio_diff')
    print feature_stats[['feature', 'train_nan_mean_ratio', 'test_nan_mean_ratio', 'train_test_nan_mean_ratio_diff']].tail()

    return feature_stats
