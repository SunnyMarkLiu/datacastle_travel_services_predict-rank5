#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-18 下午3:51
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from conf.configure import Configure
from utils import data_utils


def main():
    feature_name = 'sqg_features'
    # if data_utils.is_feature_created(feature_name):
    #     return

    # print('add gen_feature_action7.csv')
    # train_features = pd.read_csv('sqg_train_gen_feature_action7.csv')
    # test_features = pd.read_csv('sqg_test_gen_feature_action7.csv')
    #
    # print('save ', feature_name)
    # data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== merge guo-ge features ==========")
    main()
