#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-12-10 下午12:12
"""
from __future__ import absolute_import, division, print_function

import cPickle
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import xgboost as xgb
from sklearn.metrics import auc, roc_curve
from get_datasets import load_datasets


def evaluate_score(predict, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, predict, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score


def main():
    if os.path.exists('removed_features.pkl'):
        with open('removed_features.pkl', "rb") as f:
            removed_features = cPickle.load(f)
        print('removed features:', removed_features)
        return

    print("load train test datasets")
    train, test = load_datasets()

    y_train_all = train['orderType']
    train.drop(['orderType'], axis=1, inplace=True)
    all_features = train.columns.values.tolist()
    print('train: {}, test: {}, feature count: {}, orderType 1:0 = {:.5f}'.format(
        train.shape[0], test.shape[0], len(all_features), 1.0 * sum(y_train_all) / len(y_train_all)))

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
        'gpu_id': 0,
        'nthread': -1,
        'silent': 1,
        'booster': 'gbtree',
    }
    removed_features = []
    max_test_auc = 0

    process_count = 0
    all_features = ['total_feature'] + all_features
    for feature in all_features:
        if feature != 'total_feature':
            all_features.remove(feature)
        else:
            all_features.remove('total_feature')

        print('process count:', process_count, ', total:', len(all_features))
        process_count += 1
        train_df = train[all_features]

        dtrain_all = xgb.DMatrix(train_df.values, y_train_all, feature_names=all_features)

        # 4-折 valid 为 10077 和 测试集大小一致
        nfold = 5
        cv_result = xgb.cv(dict(xgb_params),
                           dtrain_all,
                           nfold=nfold,
                           stratified=True,
                           num_boost_round=4000,
                           early_stopping_rounds=100,
                           )
        best_num_boost_rounds = len(cv_result)
        mean_test_auc = cv_result.loc[best_num_boost_rounds - 11: best_num_boost_rounds - 1, 'test-auc-mean'].mean()

        if mean_test_auc > max_test_auc:
            if feature != 'total_feature':
                removed_features.append(feature)
                print('==> remove {}, mean_test_auc: {:.7f} --> {:.7f}'.format(feature, max_test_auc, mean_test_auc))
                print('removed features:', removed_features)
            max_test_auc = mean_test_auc
        else:
            all_features.append(feature)

    with open('removed_features.pkl', "wb") as f:
        cPickle.dump(removed_features, f, -1)


if __name__ == '__main__':
    print('========== xgboost 模型训练 ==========')
    main()
