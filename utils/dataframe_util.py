#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-8-2 下午2:26
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
from collections import Counter


def contains_null(dataframe):
    missing_df = dataframe.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['missing_rate'] = 1.0 * missing_df['missing_count'] / dataframe.shape[0]
    missing_df = missing_df[missing_df.missing_count > 0]
    missing_df = missing_df.sort_values(by='missing_count', ascending=False)
    return missing_df


def impute_categories_missing_data(dataframe, column):
    """填充类别属性的缺失值"""
    most_common = Counter(dataframe[column].tolist()).most_common(1)[0][0]
    dataframe.loc[dataframe[column].isnull(), column] = most_common


def simple_impute_missing_data(dataframe, column, value):
    """ 填充缺失数据 """
    if dataframe[column].isnull().sum() > 0:
        dataframe.loc[dataframe[column].isnull(), column] = value


def impute_df_numerical_but_cate_missing_data(dataframe, set_len_threshold=10, impute_value=-1):
    """
    将属性的值数目小于 set_len_threshold (默认10) 的数值型属性的缺失值填充为 impute_value (默认为 -1)
    :param dataframe: 
    :param set_len_threshold: 
    :param impute_value: 
    :return: 
    """
    columns = dataframe.select_dtypes(exclude=['object']).columns
    for c in columns:
        set_len = len(set(dataframe[c].dropna()))
        if set_len <= set_len_threshold:
            simple_impute_missing_data(dataframe, c, impute_value)
            dataframe[c] = dataframe[c].map(lambda x: np.int8(x))
