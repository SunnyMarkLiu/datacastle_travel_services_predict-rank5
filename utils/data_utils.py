#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-6 下午3:12
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from os import listdir
from os.path import isfile, join

import cPickle
from conf.configure import Configure


def load_dataset(op_scope):
    with open(Configure.processed_train_path.format(op_scope), "rb") as f:
        train = cPickle.load(f)

    with open(Configure.processed_test_path.format(op_scope), "rb") as f:
        test = cPickle.load(f)

    return train, test


def save_dataset(train, test, op_scope):
    if train is not None:
        with open(Configure.processed_train_path.format(op_scope), "wb") as f:
            cPickle.dump(train, f, -1)

    if test is not None:
        with open(Configure.processed_test_path.format(op_scope), "wb") as f:
            cPickle.dump(test, f, -1)


def load_features(features_name):
    train_path = Configure.features_path + 'train_' + features_name + '.pkl'
    test_path = Configure.features_path + 'test_' + features_name + '.pkl'

    with open(train_path, "rb") as f:
        train = cPickle.load(f)
    with open(test_path, "rb") as f:
        test = cPickle.load(f)

    return train, test


def save_features(train, test, features_name):
    if train is not None:
        train_path = Configure.features_path + 'train_' + features_name + '.pkl'
        with open(train_path, "wb") as f:
            cPickle.dump(train, f, -1)

    if test is not None:
        test_path = Configure.features_path + 'test_' + features_name + '.pkl'
        with open(test_path, "wb") as f:
            cPickle.dump(test, f, -1)


def is_feature_created(feature_name):
    feature_files = [f for f in listdir(Configure.features_path) if isfile(join(Configure.features_path, f))]
    exit_feature = sum([feature_name in feature for feature in feature_files]) > 0
    return exit_feature
