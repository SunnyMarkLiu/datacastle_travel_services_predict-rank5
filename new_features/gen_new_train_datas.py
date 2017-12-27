#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-27 下午4:53
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
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')
    train['orderTime'] = np.array(['2017-09-12 06:06:06'] * train.shape[0])
    train['orderTime'] = pd.to_datetime(train['orderTime'])
    test['orderTime'] = np.array(['2017-09-12 06:06:06'] * test.shape[0])
    test['orderTime'] = pd.to_datetime(test['orderTime'])

    print('train: ', train.shape[0], ', test: ', test.shape[0])
    print('train, ordertype1: ', sum(train['orderType']), ', ordertype0: ', sum(train['orderType'] == 0), ', 1:0 = ',
          1.0 * sum(train['orderType']) / sum(train['orderType'] == 0))

    orderHistory_train = pd.read_csv(Configure.base_path + 'train/orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.base_path + 'test/orderHistory_test.csv', encoding='utf8')
    orderHistory_train['orderTime'] = pd.to_datetime(orderHistory_train['orderTime'], unit='s')
    orderHistory_test['orderTime'] = pd.to_datetime(orderHistory_test['orderTime'], unit='s')

    new_train = pd.concat([train[['userid', 'orderType', 'orderTime']],
                           orderHistory_train[['userid', 'orderType', 'orderTime']],
                           orderHistory_test[['userid', 'orderType', 'orderTime']]], ignore_index=True)

    print('new_train: ', new_train.shape[0], ', test: ', test.shape[0])
    print('new_train, ordertype1: ', sum(new_train['orderType']), ', ordertype0: ', sum(new_train['orderType'] == 0),
          ', 1:0 = ', 1.0 * sum(new_train['orderType']) / sum(new_train['orderType'] == 0))

    print('save new train and test')
    data_utils.save_new_train_test(new_train, test)


if __name__ == "__main__":
    print("========== 合并 order history 和 train 生成新的训练集 ==========")
    main()
