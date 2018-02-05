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
from sklearn.metrics import auc, roc_curve
from get_datasets import load_datasets


def evaluate_score(predict, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, predict, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score


def main():
    print("load train test datasets")
    train, test = load_datasets()

    y_train_all = train['orderType']
    id_test = test['userid']
    del train['orderType']

    df_columns = train.columns.values
    print('===> feature count: {}'.format(len(df_columns)))
    # print('feature check before modeling...')
    # feature_util.feature_check_before_modeling(train, test, df_columns)

    dtrain = lgbm.Dataset(train, label=y_train_all)

    lgbm_params = {'bagging_fraction': 0.7,
                   'bagging_seed': 10,
                   'boosting_type': 'gbdt',
                   'feature_fraction': 0.9,
                   'feature_fraction_seed': 10,
                   'lambda_l1': 0.5,
                   'lambda_l2': 0.5,
                   'learning_rate': 0.01,
                   'max_bin': 255,
                   'metric': 'auc',
                   'min_child_weight': 1,
                   'min_split_gain': 0,
                   'min_sum_hessian_in_leaf': 0.1,
                   'num_leaves': 64,
                   'num_thread': -1,
                   'objective': 'binary',
                   'verbose': 0}

    cv_results = lgbm.cv(lgbm_params,
                         dtrain,
                         nfold=5,
                         stratified=True,
                         num_boost_round=20000,
                         early_stopping_rounds=100,
                         verbose_eval=50
                         )

    best_num_boost_rounds = len(cv_results['auc-mean'])
    print('best_num_boost_rounds =', best_num_boost_rounds)
    mean_test_auc = np.mean(cv_results['auc-mean'][best_num_boost_rounds - 6: best_num_boost_rounds - 1])
    print('mean_test_auc = {:.7f}\n'.format(mean_test_auc))

    # train model
    print('training on total training data...')
    model = lgbm.train(lgbm_params, dtrain, num_boost_round=best_num_boost_rounds)

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
