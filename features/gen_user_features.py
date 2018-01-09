#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-22 下午5:08
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
from pypinyin import lazy_pinyin
from sklearn.preprocessing import LabelEncoder
from conf.configure import Configure
from utils import data_utils


def gender_convert(gender):
    if gender == gender:
        return 'man' if gender == u'男' else 'woman'
    return 'None'


def province_convert(province):
    if province == province:
        return '_'.join(lazy_pinyin(province))
    return 'None'


def main():
    feature_name = 'basic_user_info'
    if data_utils.is_feature_created(feature_name):
        return

    # 用户个人基本信息
    train_user = pd.read_csv(Configure.base_path + 'train/userProfile_train.csv', encoding='utf8')
    test_user = pd.read_csv(Configure.base_path + 'test/userProfile_test.csv', encoding='utf8')

    # 1. 性别 dummy code
    train_user['gender'] = train_user['gender'].map(gender_convert)
    test_user['gender'] = test_user['gender'].map(gender_convert)
    dummies = pd.get_dummies(train_user['gender'], prefix='gender')
    train_user[dummies.columns] = dummies
    dummies = pd.get_dummies(test_user['gender'], prefix='gender')
    test_user[dummies.columns] = dummies

    # province = pd.read_csv('province_economic.csv', encoding='utf8')
    # train_user = train_user.merge(province, on='province', how='left')
    # test_user = test_user.merge(province, on='province', how='left')

    # 2. 省份进行 LabelEncoder
    train_user['province'] = train_user['province'].map(province_convert)
    test_user['province'] = test_user['province'].map(province_convert)
    le = LabelEncoder()
    le.fit(train_user['province'].values)
    train_user['province_code'] = le.transform(train_user['province'])
    test_user['province_code'] = le.transform(test_user['province'])

    # 3. 年龄段进行 dummy code
    train_user['age'] = train_user['age'].map(lambda age: 'lg' + age[:2] if age == age else 'None')
    test_user['age'] = test_user['age'].map(lambda age: 'lg' + age[:2] if age == age else 'None')

    print('save cleaned datasets')
    train_user.to_csv(Configure.cleaned_path + 'cleaned_userProfile_train.csv', index=False, columns=train_user.columns)
    test_user.to_csv(Configure.cleaned_path + 'cleaned_userProfile_test.csv', index=False, columns=test_user.columns)

    dummies = pd.get_dummies(train_user['age'], prefix='age')
    train_user[dummies.columns] = dummies
    dummies = pd.get_dummies(test_user['age'], prefix='age')
    test_user[dummies.columns] = dummies

    print('save ', feature_name)
    data_utils.save_features(train_user, test_user, features_name=feature_name)


if __name__ == "__main__":
    print("========== 生成用户基本特征 ==========")
    main()
