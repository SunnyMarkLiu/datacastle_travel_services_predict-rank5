# _*_ coding: utf-8 _*_
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from model.get_datasets import load_datasets
from optparse import OptionParser


# 构建模型输入
def pre_train():
    train_all, test = load_datasets()
    train_all.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    y_train_all = train_all['orderType']
    id_train = train_all['userid']
    train_all.drop(['orderType'], axis=1, inplace=True)

    id_test = test['userid']
    #test.drop(['userid'], axis=1, inplace=True)

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

    predict_feature = 'rf_predict_roof_fold{}_n_estimators{}_min_samples_leaf{}_min_samples_split{}_seed{}'.format(
        options.roof_flod, options.n_estimators, options.min_samples_leaf, options.min_samples_split, options.seed)

    print('params info:', predict_feature)

    model_params = {'max_depth': None,
                    'max_features': 'auto',
                    'min_samples_leaf': options.min_samples_leaf,
                    'min_samples_split': options.min_samples_split,
                    'n_estimators': options.n_estimators}

    roof_flod = options.roof_flod
    kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=options.seed)

    pred_train_full = np.zeros(train_all.shape[0])
    pred_test_full = 0
    cv_scores = []

    for i, (dev_index, val_index) in enumerate(kf.split(train_all, y_train_all)):
        print('========== perform fold {}, train size: {}, validate size: {} =========='.format(i, len(dev_index),
                                                                                                len(val_index)))
        train_x, val_x = train_all.ix[dev_index], train_all.ix[val_index]
        train_y, val_y = y_train_all[dev_index], y_train_all[val_index]

        model = RandomForestClassifier(n_estimators=model_params['n_estimators'], random_state=options.seed, n_jobs=-1,
                                       oob_score=True, max_depth=model_params['max_depth'],
                                       max_features=model_params['max_features'],
                                       min_samples_leaf=model_params['min_samples_leaf'],
                                       min_samples_split=model_params['min_samples_split'])
        model.fit(train_x, train_y)

        # predict train
        predict_train = model.predict_proba(train_x)[:, 1]
        train_auc = evaluate_score(predict_train, train_y)
        # predict validate
        predict_valid = model.predict_proba(val_x)[:, 1]
        valid_auc = evaluate_score(predict_valid, val_y)
        # predict test
        predict_test = model.predict_proba(test)[:, 1]

        print('train_auc = {}, valid_auc = {}'.format(train_auc, valid_auc))
        cv_scores.append(valid_auc)

        # run-out-of-fold predict
        pred_train_full[val_index] = predict_valid
        pred_test_full += predict_test

    mean_cv_scores = np.mean(cv_scores)
    print('Mean cv auc:', np.mean(cv_scores))

    print("saving train predictions for ensemble")
    train_pred_df = pd.DataFrame({'userid': id_train})
    train_pred_df[predict_feature] = pred_train_full
    train_pred_df.to_csv("./ensemble/train/rf_roof{}_predict_train_cv{}_{}.csv".format(roof_flod, mean_cv_scores, predict_feature), index=False,
                         columns=['userid', predict_feature])

    print("saving test predictions for ensemble")
    pred_test_full = pred_test_full / float(roof_flod)
    test_pred_df = pd.DataFrame({'userid': id_test})
    test_pred_df[predict_feature] = pred_test_full
    test_pred_df.to_csv("./ensemble/test/rf_roof{}_predict_test_cv{}_{}.csv".format(roof_flod, mean_cv_scores, predict_feature), index=False,
                        columns=['userid', predict_feature])

if __name__ == "__main__":
    print("========== rf run out of fold ==========")
    parser = OptionParser()

    parser.add_option(
        "-f", "--roof_flod",
        dest="roof_flod",
        default=5,
        type='int'
    )
    parser.add_option(
        "-n", "--n_estimators",
        dest="n_estimators",
        default=1300,
        type='int'
    )
    parser.add_option(
        "-l", "--min_samples_leaf",
        dest="min_samples_leaf",
        default=1,
        type='int'
    )
    parser.add_option(
        "-t", "--min_samples_split",
        dest="min_samples_split",
        default=50,
        type='int'
    )
    parser.add_option(
        "-s", "--seed",
        dest="seed",
        default=10,
        type='int'
    )
    options, _ = parser.parse_args()
    main(options)
