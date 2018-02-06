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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_curve
from model.get_datasets import load_datasets
from optparse import OptionParser


# 构建模型输入
def pre_train():
    train_all, test = load_datasets()
    # train_all.fillna(-1,inplace=True)
    # test.fillna(-1,inplace=True)

    y_train_all = train_all['orderType']
    id_train = train_all['userid']
    train_all.drop(['orderType'], axis=1, inplace=True)

    id_test = test['userid']
    # test.drop(['userid'], axis=1, inplace=True)

    print("train_all: ({}), test: ({})".format(train_all.shape, test.shape))
    return train_all, y_train_all, id_train, test, id_test


def evaluate_score(predict, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, predict, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score


def main(options):
    print("load train test datasets")
    train_all, y_train_all, id_train, test, id_test = pre_train()

    model_params = {
        'eta': options.eta,
        'min_child_weight': options.min_child_weight,
        'colsample_bytree': options.colsample_bytree,
        'max_depth': options.max_depth,
        'subsample': options.subsample,
        'seed': options.seed,
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

    roof_flod = options.roof_flod
    print(type(roof_flod))
    kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=42)

    pred_train_full = np.zeros(train_all.shape[0])
    pred_test_full = 0
    cv_scores = []

    dtest = xgb.DMatrix(test)

    predict_feature = 'xgb_predict_roof_fold{}_eta{}_min_child_weight{}_colsample_bytree{}_max_depth{}_subsample{}_seed{}'.format(
        options.roof_flod, options.eta, options.min_child_weight, options.colsample_bytree, options.max_depth,
        options.subsample, options.seed
    )

    print('params info:', predict_feature)

    for i, (dev_index, val_index) in enumerate(kf.split(train_all, y_train_all)):
        print('========== perform fold {}, train size: {}, validate size: {} =========='.format(i, len(dev_index),
                                                                                                len(val_index)))
        train_x, val_x = train_all.ix[dev_index], train_all.ix[val_index]
        train_y, val_y = y_train_all[dev_index], y_train_all[val_index]
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dval = xgb.DMatrix(val_x, label=val_y)

        model = xgb.train(model_params, dtrain,
                          evals=[(dtrain, 'train'), (dval, 'valid')],
                          verbose_eval=200,
                          early_stopping_rounds=100,
                          num_boost_round=4000)

        # predict validate
        predict_valid = model.predict(dval, ntree_limit=model.best_ntree_limit)
        valid_auc = evaluate_score(predict_valid, val_y)
        # predict test
        predict_test = model.predict(dtest, ntree_limit=model.best_ntree_limit)

        print('valid_auc = {}'.format(valid_auc))
        cv_scores.append(valid_auc)

        # run-out-of-fold predict
        pred_train_full[val_index] = predict_valid
        pred_test_full += predict_test

    mean_cv_scores = np.mean(cv_scores)
    print('Mean cv auc:', mean_cv_scores)

    print("saving train predictions for ensemble")
    train_pred_df = pd.DataFrame({'userid': id_train})
    train_pred_df[predict_feature] = pred_train_full
    train_pred_df.to_csv("./ensemble/train/lq_xgb_roof{}_predict_train_cv{}_{}.csv".format(roof_flod, mean_cv_scores, predict_feature),
                         index=False, columns=['userid', predict_feature])

    print("saving test predictions for ensemble")
    pred_test_full = pred_test_full / float(roof_flod)
    test_pred_df = pd.DataFrame({'userid': id_test})
    test_pred_df[predict_feature] = pred_test_full
    test_pred_df.to_csv("./ensemble/test/lq_xgb_roof{}_predict_test_cv{}_{}.csv".format(roof_flod, mean_cv_scores, predict_feature),
                        index=False, columns=['userid', predict_feature])


if __name__ == "__main__":
    print("========== lq xgboost run out of fold ==========")
    parser = OptionParser()

    parser.add_option(
        "-f", "--roof_flod",
        dest="roof_flod",
        default=5,
        type='int'
    )
    parser.add_option(
        "-e", "--eta",
        dest="eta",
        default=0.01,
        type='float'
    )
    parser.add_option(
        "-w", "--min_child_weight",
        dest="min_child_weight",
        default=20,
        type='int'
    )
    parser.add_option(
        "-c", "--colsample_bytree",
        dest="colsample_bytree",
        default=0.5,
        type='float'
    )
    parser.add_option(
        "-d", "--max_depth",
        dest="max_depth",
        default=10,
        type='int'
    )
    parser.add_option(
        "-p", "--subsample",
        dest="subsample",
        default=0.9,
        type='float'
    )
    parser.add_option(
        "-s", "--seed",
        dest="seed",
        default=0,
        type='int'
    )
    options, _ = parser.parse_args()
    main(options)
