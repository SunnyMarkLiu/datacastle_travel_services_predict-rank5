#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-26 下午3:14
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)


class Configure(object):
    base_path = '/d_2t/lq/competitions/data_castle/Datacastle_Travel_Services_Predict/'

    features_path = base_path + '/features/'
    processed_train_path = base_path + '/datasets/operate_{}_train.pkl'
    processed_test_path = base_path + '/datasets/operate_{}_test.pkl'

    submit_result_path = '../result/'
