#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-8 上午11:40
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
import catboost as cat
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
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

    df_columns = train.columns.values
    print('===> feature count: {}'.format(len(df_columns)))

    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': 0.1,
        'l2_leaf_reg': 5,  # L2 regularization coefficient.
        'subsample': 0.9,
        'depth': 8,  # Depth of the tree
        'border_count': 255,  # The number of splits for numerical features
        'thread_count': 6,
        'train_dir': 'catboost_train_logs',
        'bootstrap_type': 'Bernoulli',
        'use_best_model': True,
        'random_seed': 42
    }

    pool = Pool(train, y_train_all)

    cv_results = cat.train(pool=pool, params=cat_params,
                        num_boost_round=4000, nfold=5,
                        seed=42, stratified=True)

    # categorical_features_indices = np.where(train.dtypes != np.float)[0]
    # X_train, X_valid, y_train, y_valid = train_test_split(train, y_train_all, test_size=0.25, random_state=42)
    # print('train: {}, valid: {}, test: {}'.format(X_train.shape[0], X_valid.shape[0], test.shape[0]))
    # model = CatBoostClassifier(learning_rate=0.1,
    #                            thread_count=16,
    #                            random_seed=42,
    #                            use_best_model=True,
    #                            depth=6,
    #                            train_dir='./catboost_train_logs/',
    #                            calc_feature_importance=True,
    #                            leaf_estimation_method='Gradient',
    #                            logging_level='Verbose')
    #
    # model.fit(X=X_train.values,
    #           y=y_train.values,
    #           cat_features=categorical_features_indices,
    #           eval_set=[X_valid.values, y_valid.values],
    #           verbose=True,
    #        )
    #
    # # predict train
    # predict_train = model.predict_proba(X_train.values)[:, 1]
    # print(predict_train)
    # train_auc = evaluate_score(predict_train, y_train)
    #
    # # predict validate
    # predict_valid = model.predict_proba(X_valid.values)[:, 1]
    # valid_auc = evaluate_score(predict_valid, y_valid)
    #
    # print('train auc = {:.7f} , valid auc = {:.7f}\n'.format(train_auc, valid_auc))
    #
    # y_pred = model.predict(test.values)
    # df_sub = pd.DataFrame({'userid': id_test, 'orderType': y_pred})
    # submission_path = '../result/{}_submission_{}.csv'.format('xgboost',
    #                                                              time.strftime('%Y_%m_%d_%H_%M_%S',
    #                                                                            time.localtime(time.time())))
    # df_sub.to_csv(submission_path, index=False, columns=['userid', 'orderType'])
    # print('-------- predict and valid check  ------')
    # print('test  count mean: {:.6f} , std: {:.6f}'.format(np.mean(df_sub['orderType']), np.std(df_sub['orderType'])))
    # print('done.')


if __name__ == "__main__":
    print("========== catboost model ==========")
    main()
