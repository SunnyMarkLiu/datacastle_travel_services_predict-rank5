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

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from get_datasets import load_datasets
from utils import xgb_utils


def evaluate_score(predict, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, predict, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score


def main():
    print("load train test datasets")
    train, test = load_datasets()

    rm_features = ['open_app_pay_money_ratio', 'browse_product_pay_money_ratio', 'browse_product2_pay_money_ratio', 'fillin_form5_pay_money_ratio',
                   'fillin_form6_pay_money_ratio', 'fillin_form7_pay_money_ratio', 'submit_order_pay_money_ratio',
                   '2016_year_pay_money_count', '2017_year_pay_money_count',
                   'last_time_order_year', 'last_time_order_day', 'last_time_order_weekday', 'last_time_continent', 'last_time_country',
                   'last_time_order_now_has_submited_order',]

    train.drop(rm_features, axis=1, inplace=True)
    test.drop(rm_features, axis=1, inplace=True)

    train, _ = train_test_split(train, test_size=0.7, random_state=42, shuffle=True, stratify=train['orderType'])
    print('train:', train.shape)

    submit_df = pd.DataFrame({'userid': test['userid']})

    y_train_all = train['orderType']
    train.drop(['orderType'], axis=1, inplace=True)
    df_columns = train.columns.values

    xgb_params = {
        'eta': 0.1,
        'min_child_weight': 20,
        'colsample_bytree': 0.5,
        'max_depth': 10,
        'subsample': 0.9,
        'lambda': 2.0,
        'scale_pos_weight': 1,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'updater': 'grow_gpu',
        'gpu_id':0,
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
    print('---> training on total dataset')

    model = xgb.train(dict(xgb_params),
                      dtrain_all,
                      num_boost_round=best_num_boost_rounds)

    print('save feature importances')
    importances = xgb_utils.get_xgb_importance(model, df_columns)
    importances.to_csv('../features/features_importances.csv', index=False, columns=['feature', 'importance'])

    print('---> predict test')
    y_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)

    submit_df['orderType'] = y_pred
    print('-------- predict and valid check  ------')
    print('test  count mean: {:.6f} , std: {:.6f}'.format(np.mean(submit_df['orderType']), np.std(submit_df['orderType'])))
    print('done.')


if __name__ == '__main__':
    print('========== xgboost 模型训练 ==========')
    main()
