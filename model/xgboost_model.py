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

import cPickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import auc, roc_curve
from get_datasets import load_train_test, load_571_all_feature_datasets, load_0_97210_datasets
from utils import xgb_utils
from conf.configure import Configure
import model_feature_selector as feature_selector


def evaluate_score(predict, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, predict, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score


def main():
    print("load train test datasets")
    train, test = load_0_97210_datasets()

    # print('贪心算法特征选择')
    # selected_size = 0.9
    # print('original feature counts: {}, after selected count: {}'.format(train.shape[1] - 1, int((train.shape[1] - 1) * selected_size)))
    #
    # best_subset_features_path = Configure.xgboost_best_subfeatures + 'best_subset_{}_features.pkl'.format(selected_size)
    # if not os.path.exists(best_subset_features_path):
    #
    #     # baseline 特征基础上选取前 80 个特征作为 base_features
    #     baseline_imp_features = pd.read_csv('baseline_features_importances_0.97210.csv')
    #     base_features = baseline_imp_features['feature'].values.tolist()[:80]
    #     print('base_features:', len(base_features))
    #     print(np.array(base_features))
    #     decrease_auc_threshold = 0.001
    #     best_subset_features = xgboost_select_features(train, selected_size, Configure.xgboost_best_subfeatures, base_features, decrease_auc_threshold)
    #     with open(best_subset_features_path, "wb") as f:
    #         cPickle.dump(best_subset_features, f, -1)
    # else:
    #     with open(best_subset_features_path, "rb") as f:
    #         best_subset_features = cPickle.load(f)
    #
    # train = train[list(set(best_subset_features + ['orderType', 'userid']))]
    # test = test[list(set(best_subset_features + ['userid']))]

    print('贪心算法删除特征')
    # train, test = load_571_all_feature_datasets()
    remove_ratio = 0.1
    print('original feature counts: {}, after removed feature counts: {}'.format(train.shape[1] - 1, int((train.shape[1] - 1) * (1 - remove_ratio))))

    best_removed_features_path = Configure.xgboost_removed_subfeatures + 'best_removed_{}_features.pkl'.format(remove_ratio)
    if not os.path.exists(best_removed_features_path):

        best_removed_features = feature_selector.xgboost_remove_features(train, remove_ratio,
                                                                         Configure.xgboost_removed_subfeatures,
                                                                         decrease_auc_threshold=0.0008)
        with open(best_removed_features_path, "wb") as f:
            cPickle.dump(best_removed_features, f, -1)
    else:
        with open(best_removed_features_path, "rb") as f:
            best_removed_features = cPickle.load(f)

    train.drop(best_removed_features, axis=1, inplace=True)
    test.drop(best_removed_features, axis=1, inplace=True)

    y_train_all = train['orderType']
    id_test = test['userid']
    del train['orderType']

    df_columns = train.columns.values
    print('train: {}, test: {}, feature count: {}, orderType 1:0 = {}'.format(train.shape[0], test.shape[0], len(df_columns), 1.0*sum(y_train_all) / len(y_train_all)))
    # print('feature check before modeling...')
    # feature_util.feature_check_before_modeling(train, test, df_columns)

    # scale_pos_weight = (np.sum(y_train_all == 0) / np.sum(y_train_all == 1))
    scale_pos_weight = 1
    print('scale_pos_weight = ', scale_pos_weight)

    xgb_params = {
        'eta': 0.01,
        'min_child_weight': 20,
        'colsample_bytree': 0.5,
        'max_depth': 8,
        'subsample': 0.9,
        'lambda': 2.0,
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'updater': 'grow_gpu',
        'gpu_id': 2,
        'nthread': -1,
        'silent': 1,
        'booster': 'gbtree',
    }

    print('---> cv train to choose best_num_boost_round')
    dtrain_all = xgb.DMatrix(train.values, y_train_all, feature_names=df_columns)
    dtest = xgb.DMatrix(test, feature_names=df_columns)

    # 4-折 valid 为 10077 和 测试集大小一致
    nfold = 3
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
    print('---> training on total dataset to predict test and submit')

    model = xgb.train(dict(xgb_params),
                      dtrain_all,
                      num_boost_round=best_num_boost_rounds)

    print('save feature importances')
    importances = xgb_utils.get_xgb_importance(model, df_columns)
    importances.to_csv('../features/features_importances.csv', index=False, columns=['feature', 'importance'])

    print('---> predict test')
    y_pred = model.predict(dtest)
    df_sub = pd.DataFrame({'userid': id_test, 'orderType': y_pred})
    submission_path = '../result/{}_submission_{}.csv'.format('xgboost',
                                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                               time.localtime(time.time())))
    # 规则设置（cool！）
    set_one_index = test[test['2016_2017_first_last_ordertype'] == 1].index
    print('set to one count:', len(set_one_index))
    df_sub.loc[set_one_index, 'orderType'] = 1

    df_sub.to_csv(submission_path, index=False, columns=['userid', 'orderType'])
    print('-------- predict and valid check  ------')
    print('test  count mean: {:.6f} , std: {:.6f}'.format(np.mean(df_sub['orderType']), np.std(df_sub['orderType'])))
    print('done.')


if __name__ == '__main__':
    print('========== xgboost 模型训练 ==========')
    main()
