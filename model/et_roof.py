# _*_ coding: utf-8 _*_
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import auc, roc_curve
from model.get_datasets import load_datasets


# 构建模型输入
def pre_train():
    train_all, test = load_datasets()
    train_all.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

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


# 交叉验证
def model_cv(x_train, y_train):
    best_param = {}
    print('===cv: max_depth,max_features,min_samples_leaf,min_samples_split===')
    rf_model = ExtraTreesClassifier(n_estimators=1000, random_state=10, n_jobs=-1)
    param_grid = {
        'max_depth': [None, 10, 50, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [1, 25, 50],
        'min_samples_split': [2, 10, 50]
    }

    CV_rfr = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=4, scoring='roc_auc', n_jobs=-1, verbose=1)
    CV_rfr.fit(x_train, y_train)
    best_param['max_depth'] = CV_rfr.best_params_['max_depth']
    best_param['max_features'] = CV_rfr.best_params_['max_features']
    best_param['min_samples_leaf'] = CV_rfr.best_params_['min_samples_leaf']
    best_param['min_samples_split'] = CV_rfr.best_params_['min_samples_split']

    print('===cv: n_estimators===')
    rf_model = ExtraTreesClassifier(n_estimators=1000, random_state=10, n_jobs=-1,
                                    max_depth=best_param['max_depth'],
                                    max_features=best_param['max_features'],
                                    min_samples_leaf=best_param['min_samples_leaf'],
                                    min_samples_split=best_param['min_samples_split'])
    param_grid = {
        'n_estimators': range(1000, 4001, 500)
    }
    CV_rfr = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=4, scoring='roc_auc', n_jobs=-1, verbose=1)
    CV_rfr.fit(x_train, y_train)
    best_param['n_estimators'] = CV_rfr.best_params_['n_estimators']

    print("Below are the bset parameters:")
    for key in best_param:
        print("{} : {}".format(key, best_param[key]))
    del rf_model, param_grid
    return best_param


def main():
    print("load train test datasets")
    train_all, y_train_all, id_train, test, id_test = pre_train()

    model_params = {'max_depth': None,
                    'max_features': 'auto',
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'n_estimators': 2000}
    # print("model cv")
    # model_params = model_cv(train_all, y_train_all)

    roof_flod = 5
    kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=10)

    pred_train_full = np.zeros(train_all.shape[0])
    pred_test_full = 0
    cv_scores = []

    for i, (dev_index, val_index) in enumerate(kf.split(train_all, y_train_all)):
        print('========== perform fold {}, train size: {}, validate size: {} =========='.format(i, len(dev_index),
                                                                                                len(val_index)))
        train_x, val_x = train_all.ix[dev_index], train_all.ix[val_index]
        train_y, val_y = y_train_all[dev_index], y_train_all[val_index]

        model = ExtraTreesClassifier(n_estimators=model_params['n_estimators'], random_state=10, n_jobs=-1,
                                     max_depth=model_params['max_depth'],
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

    print('Mean cv auc:', np.mean(cv_scores))

    print("saving train predictions for ensemble")
    train_pred_df = pd.DataFrame({'userid': id_train})
    train_pred_df['et_orderType'] = pred_train_full
    train_pred_df.to_csv("./ensemble/et_roof{}_predict_train.csv".format(roof_flod), index=False,
                         columns=['userid', 'et_orderType'])

    print("saving test predictions for ensemble")
    pred_test_full = pred_test_full / float(roof_flod)
    test_pred_df = pd.DataFrame({'userid': id_test})
    test_pred_df['et_orderType'] = pred_test_full
    test_pred_df.to_csv("./ensemble/et_roof{}_predict_test.csv".format(roof_flod), index=False,
                        columns=['userid', 'et_orderType'])


if __name__ == "__main__":
    print("========== et run out of fold ==========")
    main()
