#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-28 下午12:24
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


def pre_order_has_comment(uid, cur_orderTime, comments_grouped, history_grouped, in_total_flag):
    """ 在本次交易 cur_orderTime 之前是否存在评论信息 """
    if in_total_flag == 0:
        return 0

    if uid not in history_grouped:
        return 0

    his_df = history_grouped[uid]
    com_df = comments_grouped[uid]
    if his_df.shape[0] == 0:
        return 0
    elif com_df.shape[0] == 0:
        return 0
    # 如果之前存在交易历史信息
    sub_his_df = his_df[his_df['orderTime'] < cur_orderTime]
    # 遍历交易历史
    orderids = sub_his_df['orderid'].values
    for orderid in orderids:
        if orderid in com_df['orderid']:
            return 1
    return 0


def built_comment_features(df, comments, history):
    features = pd.DataFrame({'userid': df['userid'], 'orderTime': df['orderTime']})

    tags = []
    for c_t in comments['tags'].values:
        tags.extend(c_t)
    unique_tags = set(tags)
    unique_tags.remove('None')

    df_ids = comments['userid'].unique()
    comments_grouped = dict(list(comments.groupby('userid')))
    history_grouped = dict(list(history.groupby('userid')))

    features['has_comment_flag'] = features['userid'].map(lambda uid: uid in df_ids).astype(int)
    features['pre_order_has_comment'] = features.apply(lambda row: pre_order_has_comment(row['userid'], row['orderTime'], comments_grouped, history_grouped, row['has_comment_flag']), axis=1)

    del features['has_comment_flag']  # 存在数据泄露，需要删除
    return features


def comment_preprocess(comments):
    comments.loc[comments['rating'] == 4.33, 'rating'] = 4
    comments.loc[comments['rating'] == 3.67, 'rating'] = 4
    comments.loc[comments['rating'] == 2.33, 'rating'] = 2
    comments['rating'] = comments['rating'].astype(int)
    comments['tags'] = comments['tags'].map(lambda t: t.split('|') if t == t else ['None'])
    comments['commentsKeyWords'] = comments['commentsKeyWords'].map(lambda t: t if t == t else ['None'])
    comments['tags_count'] = comments['tags'].map(lambda x: len(x))
    comments['keywords_count'] = comments['commentsKeyWords'].map(lambda x: len(x))
    return comments

def main():
    feature_name = 'basic_comment_features'
    # if data_utils.is_new_feature_created(feature_name):
    #     return

    train, test = data_utils.load_new_train_test()
    userComment_train = pd.read_csv(Configure.base_path + 'train/userComment_train.csv', encoding='utf8')
    userComment_test = pd.read_csv(Configure.base_path + 'test/userComment_test.csv', encoding='utf8')

    # comment data cleaning
    userComment_train = comment_preprocess(userComment_train)
    userComment_test = comment_preprocess(userComment_test)

    print('save cleaned datasets')
    userComment_train.to_csv(Configure.new_cleaned_path + 'cleaned_userComment_train.csv', index=False, columns=userComment_train.columns, encoding='utf8')
    userComment_test.to_csv(Configure.new_cleaned_path + 'cleaned_userComment_test.csv', index=False, columns=userComment_test.columns, encoding='utf8')

    orderHistory_train = pd.read_csv(Configure.new_cleaned_path + 'cleaned_orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.new_cleaned_path + 'cleaned_orderHistory_test.csv', encoding='utf8')
    orderHistory_train['orderTime'] = pd.to_datetime(orderHistory_train['orderTime'])
    orderHistory_test['orderTime'] = pd.to_datetime(orderHistory_test['orderTime'])

    print('build train features')
    train_features = built_comment_features(train, userComment_train, orderHistory_train)
    print('build test features')
    test_features = built_comment_features(test, userComment_test, orderHistory_test)

    print('save ', feature_name)
    print(train_features.shape)
    data_utils.save_new_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 历史订单评论的基本特征 ==========")
    main()
