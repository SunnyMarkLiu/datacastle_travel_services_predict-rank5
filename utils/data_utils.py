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
import numpy as np
import pandas as pd
from conf.configure import Configure


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


def load_action_sequence_label_for_nn():
    """ 加载训练神经网络的数据集 """
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')
    action_train = pd.read_csv(Configure.base_path + 'train/action_train.csv')
    action_test = pd.read_csv(Configure.base_path + 'test/action_test.csv')

    # 训练集文本
    train_seqs = []
    train_y = []

    action_grouped = dict(list(action_train.groupby('userid')))
    for uid in train['userid'].values:
        train_seqs.append(' '.join(action_grouped[uid]['actionType'].astype(str).values.tolist()))
        train_y.append(train[train['userid'] == uid]['orderType'].values[0])

    test_seqs = []
    action_grouped = dict(list(action_test.groupby('userid')))
    for uid in test['userid'].values:
        test_seqs.append(' '.join(action_grouped[uid]['actionType'].astype(str).values.tolist()))

    return train_seqs, train_y, test_seqs

class DataWrapper(object):
    def __init__(self, x, y=None, istrain=False, is_shuffle=True):
        self.x = x
        self.y = y
        self.pointer = 0
        self.total_count = self.x.shape[0]
        self.istrain = istrain
        self.is_shuffle = is_shuffle

    def shuffle(self):
        shuffled_index = np.arange(0, self.total_count)
        np.random.seed(10)
        np.random.shuffle(shuffled_index)
        self.x = self.x[shuffled_index]
        if self.istrain:
            self.y = self.y[shuffled_index]

    def load_all_data(self):
        return self.next_batch(self.x.shape[0])

    def next_batch(self, batch_size):
        end = self.pointer + batch_size

        batch_x = self.x[self.pointer: end]
        batch_y = None
        if self.istrain:
            batch_y = self.y[self.pointer: end]

        self.pointer = end

        if self.pointer >= self.total_count:
            self.pointer = 0
            if self.is_shuffle:
                self.shuffle()

        return batch_x, batch_y
