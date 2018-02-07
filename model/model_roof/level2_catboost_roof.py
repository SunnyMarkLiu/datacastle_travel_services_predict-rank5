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

import cPickle
import numpy as np
import pandas as pd
import catboost as cat
from catboost import Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_curve
from optparse import OptionParser


# 构建模型输入
def pre_train():
    with open('../ensemble/level1_train.pkl', "rb") as f:
        train = cPickle.load(f)
    with open('../ensemble/level1_test.pkl', "rb") as f:
        test = cPickle.load(f)

    y_train_all = train['orderType']
    id_train = train['userid']
    train.drop(['orderType', 'userid'], axis=1, inplace=True)

    id_test = test['userid']
    test.drop(['userid'], axis=1, inplace=True)

    train = train[test.columns.values]

    print("train_all: ({}), test: ({})".format(train.shape, test.shape))
    return train, y_train_all, id_train, test, id_test


def evaluate_score(predict, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, predict, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score


def main(options):
    print("load train test datasets")
    train_all, y_train_all, id_train, test, id_test = pre_train()

    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': options.learning_rate,
        'l2_leaf_reg': options.l2_leaf_reg,  # L2 regularization coefficient.
        'subsample': options.subsample,
        'depth': options.depth,  # Depth of the tree
        'border_count': 255,  # The number of splits for numerical features
        'thread_count': 6,
        'train_dir': 'catboost_train_logs',
        'bootstrap_type': 'Bernoulli',
        'use_best_model': True,
        'random_seed': options.seed
    }

    roof_flod = options.roof_flod
    kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=options.seed)

    pred_train_full = np.zeros(train_all.shape[0])
    pred_test_full = 0
    cv_scores = []

    predict_feature = 'catboost_predict_roof_fold{}_lr{}_l2_leaf_reg{}_subsample{}_depth{}_seed{}'.format(
        options.roof_flod, options.learning_rate, options.l2_leaf_reg, options.subsample, options.depth, options.seed
    )

    print('params info:', predict_feature)

    for i, (dev_index, val_index) in enumerate(kf.split(train_all, y_train_all)):
        print('========== perform fold {}, train size: {}, validate size: {} =========='.format(i, len(dev_index),
                                                                                                len(val_index)))
        train_x, val_x = train_all.ix[dev_index], train_all.ix[val_index]
        train_y, val_y = y_train_all[dev_index], y_train_all[val_index]

        model = cat.train(pool=Pool(train_x, train_y), params=cat_params, iterations=460, eval_set=(val_x, val_y), verbose=False)

        # predict validate
        predict_valid = model.predict(val_x.values, prediction_type='Probability')[:, 1]
        valid_auc = evaluate_score(predict_valid, val_y)
        # predict test
        predict_test = model.predict(test.values, prediction_type='Probability')[:, 1]

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
    train_pred_df.to_csv("./ensemble/level2/level2_catboost_predict_train_cv{}.csv".format(mean_cv_scores),
                         index=False, columns=['userid', predict_feature])

    print("saving test predictions for ensemble")
    pred_test_full = pred_test_full / float(roof_flod)
    test_pred_df = pd.DataFrame({'userid': id_test})
    test_pred_df[predict_feature] = pred_test_full
    test_pred_df.to_csv("./ensemble/level2/level2_catboost_predict_test_cv{}.csv".format(mean_cv_scores),
                        index=False, columns=['userid', predict_feature])


if __name__ == "__main__":
    print("========== level-2 catboost run out of fold ==========")
    parser = OptionParser()

    parser.add_option(
        "-f", "--roof_flod",
        dest="roof_flod",
        default=5,
        type='int'
    )
    parser.add_option(
        "-r", "--learning_rate",
        dest="learning_rate",
        default=0.1,
        type='float'
    )
    parser.add_option(
        "-l", "--l2_leaf_reg",
        dest="l2_leaf_reg",
        default=5,
        type='int'
    )
    parser.add_option(
        "-s", "--subsample",
        dest="subsample",
        default=0.9,
        type='float'
    )
    parser.add_option(
        "-d", "--depth",
        dest="depth",
        default=8,
        type='int'
    )
    parser.add_option(
        "-e", "--seed",
        dest="seed",
        default=0,
        type='int'
    )
    ops, _ = parser.parse_args()
    main(ops)
