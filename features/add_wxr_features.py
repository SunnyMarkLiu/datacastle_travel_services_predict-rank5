#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-14 下午8:23
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings
warnings.filterwarnings('ignore')

import cPickle
import numpy as np
import pandas as pd
from conf.configure import Configure
from utils import data_utils


def main():
    feature_name = 'wxr_features'
    # if data_utils.is_feature_created(feature_name):
    #     return

    print('add comment score features')
    with open('wxr_train_comment_features.pkl', "rb") as f:
        user_comment_train = cPickle.load(f)
    with open('wxr_test_comment_features.pkl', "rb") as f:
        user_comment_test = cPickle.load(f)

    user_comment_train.fillna(-1, inplace=True)
    user_comment_test.fillna(-1, inplace=True)

    train_features = user_comment_train
    test_features = user_comment_test

    # print('add user_info features')
    # with open('wxr_train_user_info_features.pkl', "rb") as f:
    #     train_user_info = cPickle.load(f)
    # with open('wxr_test_user_info_features.pkl', "rb") as f:
    #     test_user_info = cPickle.load(f)
    # train_user_info.drop(['gender', 'province', 'age'], axis=1, inplace=True)
    # test_user_info.drop(['gender', 'province', 'age'], axis=1, inplace=True)
    #
    # train_features = train_features.merge(train_user_info, on='userid', how='left')
    # test_features = test_features.merge(test_user_info, on='userid', how='left')

    print('save ', feature_name)
    data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== merge wxr features ==========")
    main()
