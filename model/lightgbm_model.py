#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-8 上午11:17
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

import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from get_datasets import load_train_test
from IPython.display import display


def evaluate_score(predict, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, predict, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score


def main():
    print("load train test datasets")
    train, test = load_train_test()

    y_train_all = train['orderType']
    id_test = test['userid']
    del train['orderType']

    df_columns = train.columns.values
    print('===> feature count: {}'.format(len(df_columns)))
    # print('feature check before modeling...')
    # feature_util.feature_check_before_modeling(train, test, df_columns)

    scale_pos_weight = (np.sum(y_train_all == 0) / np.sum(y_train_all == 1)) - 1
    # print('scale_pos_weight = ', scale_pos_weight)

    d_train = lgbm.Dataset(train, label=y_train_all)

    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'nthread': -1,
        'subsample': 0.75,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'min_split_gain': 0.4,

        'num_leaves': 2 ** 6,
        'learning_rate': 0.015,
        'max_depth': 10,

        'reg_alpha': 0.1,
        'reg_lambda': 0.1,

        'scale_pos_weight': 1,
        'early_stopping_round': 20,
        'metric': 'auc',
        'verbose': 0
    }

    cv_results = lgbm.cv(lgbm_params,
                         d_train,
                         num_boost_round=20000,
                         nfold=5,
                         early_stopping_rounds=200,
                         verbose_eval=20)

    best_num_boost_rounds = len(cv_results['auc-mean'])
    print('best_num_boost_rounds =', best_num_boost_rounds)
    # train model
    print('training on total training data...')
    model = lgbm.train(lgbm_params, d_train, num_boost_round=best_num_boost_rounds)

    print('predict submit...')
    y_pred = model.predict(test)
    df_sub = pd.DataFrame({'userid': id_test, 'orderType': y_pred})
    submission_path = '../result/{}_submission_{}.csv'.format('lightgbm',
                                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                               time.localtime(time.time())))
    df_sub.to_csv(submission_path, index=False, columns=['userid', 'orderType'])
    print('-------- predict and valid check  ------')
    print('test  count mean: {:.6f} , std: {:.6f}'.format(np.mean(df_sub['orderType']), np.std(df_sub['orderType'])))
    print('done.')


if __name__ == "__main__":
    print("========== lightgbm model ==========")
    main()
