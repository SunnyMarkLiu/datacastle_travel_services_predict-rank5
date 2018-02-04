#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-28 下午4:18
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from utils import data_utils


def main():
    feature_name = 'other_features1'
    train_features = pd.read_csv('other_feature1/feature_train.csv')
    test_features = pd.read_csv('other_feature1/feature_test.csv')

    feature_score = pd.read_csv('other_feature1/action_process_features_importances.csv')
    feature_score = feature_score.sort_values(by='importance', ascending=False).reset_index(drop=True)
    used_features = feature_score['feature'].values.tolist()[:80]
    if 'userid' not in used_features:
        used_features.append('userid')
    train = train_features[used_features]
    test = test_features[used_features]

    train_features = pd.read_csv('other_feature2/type_typevalue_train.csv')
    test_features = pd.read_csv('other_feature2/type_typevalue_test.csv')
    train = pd.merge(train, train_features, on='userid', how='left')
    test = pd.merge(test, test_features, on='userid', how='left')

    # train_features = pd.read_csv('other_feature2/type_type_train.csv')
    # test_features = pd.read_csv('other_feature2/type_type_test.csv')
    # train = pd.merge(train, train_features, on='userid', how='left')
    # test = pd.merge(test, test_features, on='userid', how='left')

    # train_features = pd.read_csv('other_feature2/actiontype_cloest_train.csv')
    # test_features = pd.read_csv('other_feature2/actiontype_cloest_test.csv')
    # train = pd.merge(train, train_features, on='userid', how='left')
    # test = pd.merge(test, test_features, on='userid', how='left')

    data_utils.save_features(train, test, feature_name)


if __name__ == "__main__":
    print("========== 合并其他队友特征 ==========")
    main()
