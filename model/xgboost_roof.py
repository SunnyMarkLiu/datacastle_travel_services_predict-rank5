#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-8 上午9:45
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from get_datasets import load_train_test


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

    X_train = train
    X_test = test
    df_columns = train.columns.values
    print('===> feature count: {}'.format(len(df_columns)))

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

    roof_flod = 7
    kf = KFold(n_splits=roof_flod, shuffle=True, random_state=42)

    pred_train_full = np.zeros(train.shape[0])
    pred_test_full = 0
    cv_scores = []

    dtest = xgb.DMatrix(test, feature_names=df_columns)

    for i, (dev_index, val_index) in enumerate(kf.split(X_train)):
        print('========== perform fold {}, train size: {}, validate size: {} =========='.format(i, len(dev_index), len(val_index)))
        dev_X, val_X = X_train.ix[dev_index], X_train.ix[val_index]
        dev_y, val_y = y_train_all[dev_index], y_train_all[val_index]
        ddev = xgb.DMatrix(dev_X, dev_y, feature_names=df_columns)
        dval = xgb.DMatrix(val_X, val_y, feature_names=df_columns)

        model = xgb.train(dict(xgb_params), ddev,
                          evals=[(ddev, 'train'), (dval, 'valid')],
                          verbose_eval=50,
                          early_stopping_rounds=100,
                          num_boost_round=4000)

        pred_valid = model.predict(dval)
        pred_test = model.predict(dtest)

        # predict train
        predict_train = model.predict(ddev)
        train_auc = evaluate_score(predict_train, dev_y)

        # predict validate
        predict_valid = model.predict(dval)
        valid_auc = evaluate_score(predict_valid, val_y)
        print('========== train_auc = {}, valid_auc = {} =========='.format(train_auc, valid_auc))
        cv_scores.append(valid_auc)

        # run-out-of-fold predict
        pred_train_full[val_index] = pred_valid
        pred_test_full += pred_test

    print('Mean cv auc:', np.mean(cv_scores))
    pred_test_full = pred_test_full / float(roof_flod)

    # saving test predictions for ensemble #
    test_pred_df = pd.DataFrame({'userid': id_test})
    test_pred_df['orderType'] = pred_test_full
    test_pred_df.to_csv("./ensemble/xgboost_roof{}_predict_test.csv".format(roof_flod), index=False, columns=['userid', 'orderType'])


if __name__ == "__main__":
    print("========== xgboost run out of fold ==========")
    main()
