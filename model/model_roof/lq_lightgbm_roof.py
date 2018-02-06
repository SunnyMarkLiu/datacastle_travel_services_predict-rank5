# _*_ coding: utf-8 _*_
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import lightgbm as lgb
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
    train_all.drop(['orderType', 'userid'], axis=1, inplace=True)

    id_test = test['userid']
    test.drop(['userid'], axis=1, inplace=True)

    print("train_all: ({}), test: ({})".format(train_all.shape, test.shape))
    return train_all, y_train_all, id_train, test, id_test


# 评估函数
def evaluate_score(predict, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, predict, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score


def main(options):
    print("load train test datasets")
    train_all, y_train_all, id_train, test, id_test = pre_train()

    model_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': options.learning_rate,
        'num_leaves': options.num_leaves,
        'min_child_weight': options.min_child_weight,
        'feature_fraction': options.feature_fraction,
        'bagging_fraction': options.bagging_fraction,
        'lambda_l1': options.lambda_l1,
        'lambda_l2': options.lambda_l2,
        'bagging_seed': 10,
        'feature_fraction_seed': 10,
        'nthread': -1,
        'verbose': 0
    }

    predict_feature = 'lgbm_roof_fold{}_lr{}_numleaves{}_minchildweight{}_featurefraction{}_baggingfraction{}_l1-{}_l2-{}seed{}'.format(
        options.roof_flod, options.learning_rate, options.num_leaves, options.min_child_weight, options.feature_fraction,
        options.bagging_fraction, options.lambda_l1, options.lambda_l2, options.seed
    )

    roof_flod = options.roof_flod
    kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=10)

    pred_train_full = np.zeros(train_all.shape[0])
    pred_test_full = 0
    cv_scores = []

    for i, (dev_index, val_index) in enumerate(kf.split(train_all, y_train_all)):
        print('========== perform fold {}, train size: {}, validate size: {} =========='.format(i, len(dev_index),
                                                                                                len(val_index)))
        train_x, val_x = train_all.ix[dev_index], train_all.ix[val_index]
        train_y, val_y = y_train_all[dev_index], y_train_all[val_index]
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(val_x, val_y, reference=lgb_train)

        model = lgb.train(model_params, lgb_train, num_boost_round=5000, valid_sets=[lgb_train, lgb_eval],
                          valid_names=['train', 'eval'], early_stopping_rounds=100, verbose_eval=200)

        # predict validate
        predict_valid = model.predict(val_x, num_iteration=model.best_iteration)
        valid_auc = evaluate_score(predict_valid, val_y)
        # predict test
        predict_test = model.predict(test, num_iteration=model.best_iteration)

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
    train_pred_df.to_csv("./ensemble/train/lq_lgbm_roof{}_predict_train_cv{}_{}.csv".format(roof_flod, mean_cv_scores, predict_feature),
                         index=False, columns=['userid', predict_feature])

    print("saving test predictions for ensemble")
    pred_test_full = pred_test_full / float(roof_flod)
    test_pred_df = pd.DataFrame({'userid': id_test})
    test_pred_df[predict_feature] = pred_test_full
    test_pred_df.to_csv("./ensemble/test/lq_lgbm_roof{}_predict_test_cv{}_{}.csv".format(roof_flod, mean_cv_scores, predict_feature),
                        index=False, columns=['userid', predict_feature])


if __name__ == "__main__":
    print("========== lq lightgbm run out of fold ==========")
    parser = OptionParser()

    parser.add_option(
        "--fl", "--roof_flod",
        dest="roof_flod",
        default=5,
        type='int'
    )
    parser.add_option(
        "--lr", "--learning_rate",
        dest="learning_rate",
        default=0.01,
        type='float'
    )
    parser.add_option(
        "--mw", "--min_child_weight",
        dest="min_child_weight",
        default=1,
        type='int'
    )
    parser.add_option(
        "--ff", "--feature_fraction",
        dest="feature_fraction",
        default=0.9,
        type='float'
    )
    parser.add_option(
        "--nl", "--num_leaves",
        dest="num_leaves",
        default=64,
        type='int'
    )
    parser.add_option(
        "--bf", "--bagging_fraction",
        dest="bagging_fraction",
        default=0.7,
        type='float'
    )
    parser.add_option(
        "--l1", "--lambda_l1",
        dest="lambda_l1",
        default=0.5,
        type='float'
    )
    parser.add_option(
        "--l2", "--lambda_l2",
        dest="lambda_l2",
        default=0.5,
        type='float'
    )
    parser.add_option(
        "--sd", "--seed",
        dest="seed",
        default=0,
        type='int'
    )
    ops, _ = parser.parse_args()
    main(ops)
