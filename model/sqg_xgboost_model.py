#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-12-10 下午12:12
"""
from __future__ import absolute_import, division, print_function

import os
import sys

reload(sys)
sys.setdefaultencoding('gbk')

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import auc, roc_curve
from utils import xgb_utils
from conf.configure import Configure


def evaluate_score(predict, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, predict, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score


def main():
    print("load train test datasets")
    train = pd.read_csv(Configure.base_path + 'sun_qian_guo/train.csv')
    test = pd.read_csv(Configure.base_path + 'sun_qian_guo/test.csv')

    submit_df = pd.DataFrame({'userid': test['userid']})
    y_train_all = train['orderType']
    train.drop(['orderType'], axis=1, inplace=True)

    train.columns = ['feature_{}'.format(i) for i in range(train.shape[1])]
    test.columns = ['feature_{}'.format(i) for i in range(test.shape[1])]

    df_columns = train.columns.values
    print('train: {}, test: {}, feature count: {}, orderType 1:0 = {:.5f}'.format(
        train.shape[0], test.shape[0], len(df_columns), 1.0*sum(y_train_all) / len(y_train_all)))

    # print('feature check before modeling...')
    # feature_util.feature_check_before_modeling(train, test, df_columns)

    xgb_params = {
        'eta': 0.01,
        'min_child_weight': 20,
        'colsample_bytree': 0.5,
        'max_depth': 10,
        'subsample': 0.9,
        'lambda': 2.0,
        'scale_pos_weight': 1,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'updater': 'grow_gpu',
        'gpu_id':2,
        'nthread': -1,
        'silent': 1,
        'booster': 'gbtree',
    }

    print('---> cv train to choose best_num_boost_round')
    dtrain_all = xgb.DMatrix(train.values, y_train_all, feature_names=df_columns)
    dtest = xgb.DMatrix(test, feature_names=df_columns)

    # 4-折 valid 为 10077 和 测试集大小一致
    nfold = 5
    cv_result = xgb.cv(dict(xgb_params),
                       dtrain_all,
                       nfold=nfold,
                       stratified=True,
                       num_boost_round=400,
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
    print('---> training on total dataset')

    model = xgb.train(dict(xgb_params),
                      dtrain_all,
                      num_boost_round=best_num_boost_rounds)

    print('save feature importances')
    importances = xgb_utils.get_xgb_importance(model, df_columns)
    importances.to_csv('../features/features_importances.csv', index=False, columns=['feature', 'importance'])
    importances = xgb_utils.get_xgb_importance_by_weights(model, df_columns)
    importances.to_csv('../features/features_weights.csv', index=False, columns=['feature', 'weights'])
    importances = xgb_utils.get_xgb_importance_by_gains(model, df_columns)
    importances.to_csv('../features/features_gains.csv', index=False, columns=['feature', 'gains'])
    importances = xgb_utils.get_xgb_importance_by_covers(model, df_columns)
    importances.to_csv('../features/features_covers.csv', index=False, columns=['feature', 'covers'])

    print('---> predict test')
    y_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)

    submit_df['orderType'] = y_pred
    submission_path = '../result/{}_submission_{}.csv'.format('xgboost',
                                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                               time.localtime(time.time())))

    # # 规则设置
    # submit_df['orderType'] = submit_df.apply(lambda row: 1 if row['history_order_type_sum_lg0'] == 1 else row['orderType'], axis=1)
    # # 概率阈值需要调
    # # df_sub['orderType'] = df_sub.apply(lambda row: 0 if ((row['history_order_type_sum_lg0'] == 0) and (row['orderType'] < 0.001)) else row['orderType'], axis=1)
    # submit_df = submit_df[['userid', 'orderType']]

    submit_df.to_csv(submission_path, index=False, columns=['userid', 'orderType'])
    print('-------- predict and valid check  ------')
    print('test  count mean: {:.6f} , std: {:.6f}'.format(np.mean(submit_df['orderType']), np.std(submit_df['orderType'])))
    print('done.')


if __name__ == '__main__':
    print('========== xgboost 模型训练 ==========')
    main()
