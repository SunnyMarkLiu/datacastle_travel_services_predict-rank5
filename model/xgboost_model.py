#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-12-10 下午12:12
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from get_datasets import load_train_test
from utils import xgb_utils
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

    xgb_params = {
        'eta': 0.01,
        'min_child_weight': 20,
        'colsample_bytree': 0.5,
        'max_depth': 8,
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

    X_train, X_valid, y_train, y_valid = train_test_split(train, y_train_all, test_size=0.25, random_state=42)
    print('train: {}, valid: {}, test: {}'.format(X_train.shape[0], X_valid.shape[0], test.shape[0]))

    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    dvalid = xgb.DMatrix(X_valid, y_valid, feature_names=df_columns)
    dtest = xgb.DMatrix(test, feature_names=df_columns)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(dict(xgb_params),
                      dtrain,
                      evals=watchlist,
                      verbose_eval=50,
                      early_stopping_rounds=100,
                      num_boost_round=4000)

    importances = xgb_utils.get_xgb_importance(model)
    importances.to_csv('../features/features_importances.csv', index=False, columns=['feature', 'importance'])
    feature_percentile = 0.95
    use_columns = importances[importances['importance'] > importances.importance.quantile(1 - feature_percentile)]['feature']
    print('使用 {}% 的特征后，特征的维度：{}'.format(feature_percentile, len(use_columns)))
    train = train[use_columns]
    test = test[use_columns]

    X_train, X_valid, y_train, y_valid = train_test_split(train, y_train_all, test_size=0.25, random_state=42)
    print('train: {}, valid: {}, test: {}'.format(X_train.shape[0], X_valid.shape[0], test.shape[0]))
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=use_columns)
    dvalid = xgb.DMatrix(X_valid, y_valid, feature_names=use_columns)
    dtest = xgb.DMatrix(test, feature_names=use_columns)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(dict(xgb_params),
                      dtrain,
                      evals=watchlist,
                      verbose_eval=50,
                      early_stopping_rounds=100,
                      num_boost_round=4000)

    # predict train
    predict_train = model.predict(dtrain)
    train_auc = evaluate_score(predict_train, y_train)

    # predict validate
    predict_valid = model.predict(dvalid)
    valid_auc = evaluate_score(predict_valid, y_valid)

    print('train auc = {:.7f} , valid auc = {:.7f}\n'.format(train_auc, valid_auc))

    print('---> cv train to choose best_num_boost_round')
    dtrain_all = xgb.DMatrix(train.values, y_train_all, feature_names=use_columns)

    cv_result = xgb.cv(dict(xgb_params),
                       dtrain_all,
                       num_boost_round=4000,
                       early_stopping_rounds=100,
                       verbose_eval=100,
                       show_stdv=False,
                       )
    best_num_boost_rounds = len(cv_result)
    mean_train_logloss = cv_result.loc[best_num_boost_rounds-11 : best_num_boost_rounds-1, 'train-auc-mean'].mean()
    mean_test_logloss = cv_result.loc[best_num_boost_rounds-11 : best_num_boost_rounds-1, 'test-auc-mean'].mean()
    print('best_num_boost_rounds = {}'.format(best_num_boost_rounds))

    # num_boost_round = int(best_num_boost_rounds * 1.1)
    # print('num_boost_round = ', num_boost_round)

    print('mean_train_auc = {:.7f} , mean_test_auc = {:.7f}\n'.format(mean_train_logloss, mean_test_logloss))
    print('---> training on total dataset to predict test and submit')
    model = xgb.train(dict(xgb_params),
                      dtrain_all,
                      num_boost_round=best_num_boost_rounds)
    print('---> predict and submit')
    print('---> predict submit')
    y_pred = model.predict(dtest)
    df_sub = pd.DataFrame({'userid': id_test, 'orderType': y_pred})
    submission_path = '../result/{}_submission_{}.csv'.format('xgboost',
                                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                               time.localtime(time.time())))
    df_sub.to_csv(submission_path, index=False, columns=['userid', 'orderType'])
    print('-------- predict and valid check  ------')
    print('test  count mean: {:.6f} , std: {:.6f}'.format(np.mean(df_sub['orderType']), np.std(df_sub['orderType'])))
    print('done.')


if __name__ == '__main__':
    print('========== xgboost 模型训练 ==========')
    main()
