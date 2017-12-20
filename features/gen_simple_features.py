#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-12-10 上午11:27
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
from optparse import OptionParser
from utils import data_utils


def gen_time_features(df):
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    # month
    df['visit_month'] = df['visit_date'].dt.month
    # day
    df['visit_day'] = df['visit_date'].dt.day
    # weekofyear
    df['visit_weekofyear'] = df['visit_date'].dt.weekofyear
    # weekday
    df['visit_weekday'] = df['visit_date'].dt.weekday

    date_info = pd.read_csv(Configure.base_path + '/date_info.csv')
    df = pd.merge(df, date_info.rename(columns={'calendar_date': 'visit_date'}), how='left', on='visit_date')
    del df['day_of_week']
    return df


def gen_air_reserve_features(train, test):
    """
    air_reserve.csv
    """
    train['visit_date'] = train['visit_date'].dt.date
    test['visit_date'] = test['visit_date'].dt.date

    air_reserve = pd.read_csv(Configure.base_path + '/air_reserve.csv')
    air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])
    air_reserve['visit_date'] = air_reserve['visit_datetime'].dt.date

    air_reserve_fea = air_reserve.groupby(['air_store_id', 'visit_date']).sum()['reserve_visitors'].reset_index()
    air_reserve_fea.rename(columns={'reserve_visitors': 'reserve_visitors_sum'}, inplace=True)
    air_reserve_fea['reserve_visitors_sum'] = np.log1p(air_reserve_fea['reserve_visitors_sum'])

    train = pd.merge(train, air_reserve_fea, on=['air_store_id', 'visit_date'], how='left')
    test = pd.merge(test, air_reserve_fea, on=['air_store_id', 'visit_date'], how='left')

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    return train, test


def main(op_scope):
    op_scope = int(op_scope)
    # if os.path.exists(Configure.processed_train_path.format(op_scope)):
    #     return

    print("---> load datasets")
    train = pd.read_csv(Configure.base_path + '/air_visit_data.csv')
    test = pd.read_csv(Configure.base_path + '/sample_submission.csv')
    test['air_store_id'] = test['id'].map(lambda x: x[:20])
    test['visit_date'] = test['id'].map(lambda x: x[21:])
    test.drop(['visitors'], axis=1, inplace=True)
    print("train: {}, test: {}".format(train.shape, test.shape))

    print('---> gen_time_features')
    train = gen_time_features(train)
    test = gen_time_features(test)

    print('---> gen_air_reserve_features')
    train, test = gen_air_reserve_features(train, test)

    print("train: {}, test: {}".format(train.shape, test.shape))
    print("---> save datasets")
    data_utils.save_dataset(train, test, op_scope)


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "-o", "--op_scope",
        dest="op_scope",
        default="0",
        help="""operate scope: 0, 1, 2, 3..."""
    )

    options, _ = parser.parse_args()
    print("========== generate some simple features ==========")
    main(options.op_scope)
