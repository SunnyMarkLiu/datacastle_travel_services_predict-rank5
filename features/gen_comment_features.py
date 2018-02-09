#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-25 下午3:27
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from conf.configure import Configure
from utils import data_utils
import json
from qcloudapi3 import QcloudApi
from tqdm import tqdm


def user_rating_sattistic(uid, userid_grouped, flag):
    """ 用户订单的评论信息 """
    if flag == 0:
        return -1, -1, -1, -1, -1, -1

    df = userid_grouped[uid]
    if df.shape[0] == 0:
        return -1, -1, -1, -1, -1, -1
    else:
        return df['rating'].mean(), df['rating'].max(), df['rating'].min(), df['rating'].median(), df['rating'].std(), df.shape[0]


def user_rating_count(uid, userid_grouped, flag):
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    if df.shape[0] == 0:
        return -1
    else:
        return len(set(df['rating']))


def user_rating_ratio(uid, userid_grouped, flag):
    """ 用户打分的比例 """
    if flag == 0:
        return -1, -1, -1, -1, -1

    df = userid_grouped[uid]
    if df.shape[0] == 0:
        return -1, -1, -1, -1, -1
    else:
        ratings = df['rating'].tolist()
        count_1 = float(ratings.count(1))
        count_2 = float(ratings.count(2))
        count_3 = float(ratings.count(3))
        count_4 = float(ratings.count(4))
        count_5 = float(ratings.count(5))
        return count_1 / df.shape[0], count_2 / df.shape[0], count_3 / df.shape[0], count_4 / df.shape[0], count_5 / df.shape[0]

def last_time_rating(uid, userid_grouped, flag):
    """ 最后一次打分 """
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    if df.shape[0] == 0:
        return -1
    else:
        return df.iloc[-1]['rating']


def total_rating(uid, userid_grouped, flag):
    """ 打分总数 """
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    if df.shape[0] == 0:
        return -1
    else:
        return sum(df['rating'])

def last_3_order_has_lower5_rating(uid, userid_grouped, flag, count):
    """ 最后几次是否存在低于 5 分的评论 """
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    if df.shape[0] == 0:
        return -1
    else:
        return int(sum(df['rating'].values[-count:]) < (5.0 * count))


def add_tages_ratio(uid, userid_grouped, flag):
    """ 添加 tags 的比率 """
    if flag == 0:
        return -1

    df = userid_grouped[uid]
    if df.shape[0] == 0:
        return -1
    else:
        return 1.0 * df[df['tags'] == ['None']].shape[0] / df.shape[0]


def comment_tags_keywords_statistic(uid, userid_grouped, flag, name):
    """ 评论标签的统计特征 """
    if flag == 0:
        return -1, -1, -1, -1, -1, -1

    df = userid_grouped[uid]
    if df.shape[0] == 0:
        return -1, -1, -1, -1, -1, -1
    else:
        return df[name].std(), df[name].var(), df[name].min(), df[name].median(), df[name].max(), df[name].mean()


def bad_good_score_count(uid, userid_grouped, flag):
    if flag == 0:
        return -1, -1
    df = userid_grouped[uid]
    return sum(df['rating'] >= 4), sum(df['rating'] < 4)


def comment_has_jingxi(uid, userid_grouped, flag):
    if flag == 0:
        return -1, -1
    df = userid_grouped[uid]
    df = df.fillna(' ')
    tags = ' '.join(df['tags'].values.tolist())
    result = u'行程安排有惊喜' in tags
    return result


def commentKey_score(df):
    """使用百度文智情感分析API对用户评论打分，QcloudApi为百度文智SDK"""
    module = 'wenzhi'
    action = 'TextSentiment'
    config = {
        'Region': 'bj',
        'secretId': '',
        'secretKey': '',
        'method': 'get'
    }
    df_c = df.copy()
    df_c['commentsKeyWords_score'] = np.nan
    df_select_index = df_c[pd.notnull(df_c['commentsKeyWords'])].index
    service = QcloudApi(module, config)
    for i in xrange(len(df_select_index)):
        comment_str = df_c.loc[df_select_index[i], 'commentsKeyWords']
        params = {
            'content': comment_str
        }
        try:
            # 调用接口，发起请求
            res = json.loads(service.call(action, params))
            df_c.loc[df_select_index[i], 'commentsKeyWords_score'] = res['positive']
        except Exception, e:
            print
            'exception:', e
    return df_c

def tag_score(df):
    """分正面、负面统计tag"""
    positive_tags = []
    negative_tags = []
    f = open('./tags.txt', 'r')
    for line in f:
        arr = line.decode('utf8').split()
        if arr[1] == '1':
            positive_tags.append(arr[0])
        else:
            negative_tags.append(arr[0])
    f.close()
    print (len(positive_tags), len(negative_tags))
    df_c = df.copy()
    df_c['positive_tags'] = 0
    df_c['negative_tags'] = 0
    for i in tqdm(xrange(df_c.shape[0])):
        tag_str = df_c.loc[i, 'tags']
        if tag_str == tag_str:
            tag_arr = tag_str.split(u'|')
            p_num = 0
            n_num = 0
            for tag in tag_arr:
                if tag in positive_tags:
                    p_num += 1
                elif tag in negative_tags:
                    n_num += 1
            df_c.loc[i, 'positive_tags'] = p_num
            df_c.loc[i, 'negative_tags'] = n_num
    df_c['tags_score'] = df_c['positive_tags'] - df_c['negative_tags']
    return df_c


def built_comment_features(df, comments):
    features = pd.DataFrame({'userid': df['userid']})

    # comments['tags'] = comments['tags'].map(lambda t: t.split('|') if t == t else ['None'])
    # comments['commentsKeyWords'] = comments['commentsKeyWords'].map(lambda t: t if t == t else ['None'])
    # comments['tags_count'] = comments['tags'].map(lambda x: len(x))
    # comments['keywords_count'] = comments['commentsKeyWords'].map(lambda x: len(x))
    # tags = []
    # for c_t in comments['tags'].values:
    #     tags.extend(c_t)
    # unique_tags = set(tags)
    # unique_tags.remove('None')

    df_ids = comments['userid'].unique()
    userid_grouped = dict(list(comments.groupby('userid')))

    # 基本特征
    features['has_comment_flag'] = features['userid'].map(lambda uid: uid in df_ids).astype(int)
    features['user_rating_sattistic'] = features.apply(lambda row: user_rating_sattistic(row['userid'], userid_grouped, row['has_comment_flag']), axis=1)
    # features['user_rating_mean'] = features['user_rating_sattistic'].map(lambda x: x[0])
    # features['user_rating_max'] = features['user_rating_sattistic'].map(lambda x: x[1])
    # features['user_rating_min'] = features['user_rating_sattistic'].map(lambda x: x[2])
    # features['user_rating_median'] = features['user_rating_sattistic'].map(lambda x: x[3])
    features['user_rating_std'] = features['user_rating_sattistic'].map(lambda x: x[4])
    # features['user_comment_count'] = features['user_rating_sattistic'].map(lambda x: x[5])
    del features['user_rating_sattistic']

    # 用户打分比例
    features['user_rating_ratio'] = features.apply(lambda row: user_rating_ratio(row['userid'], userid_grouped, row['has_comment_flag']), axis=1)
    features['raing1_ratio'] = features['user_rating_ratio'].map(lambda x: x[0])
    features['raing2_ratio'] = features['user_rating_ratio'].map(lambda x: x[1])
    features['raing3_ratio'] = features['user_rating_ratio'].map(lambda x: x[2])
    features['raing4_ratio'] = features['user_rating_ratio'].map(lambda x: x[3])
    # features['raing5_ratio'] = features['user_rating_ratio'].map(lambda x: x[4])
    del features['user_rating_ratio']

    # # 是否好评好评次数
    # features['bad_good_score_count'] = features.apply(lambda row: bad_good_score_count(row['userid'], userid_grouped, row['has_comment_flag']), axis=1)
    # features['good_score_count'] = features['bad_good_score_count'].map(lambda x: x[0])
    # features['bad_score_count'] = features['bad_good_score_count'].map(lambda x: x[1])
    # features['good_score_ratio'] = features['good_score_count'].astype(float) / (features['good_score_count'] + features['bad_score_count'])
    # del features['bad_good_score_count']

    # features['comment_has_jingxi'] = features.apply(lambda row: comment_has_jingxi(row['userid'], userid_grouped, row['has_comment_flag']), axis=1)

    # 最后一次打分
    # features['last_time_rating'] = features.apply(lambda row: last_time_rating(row['userid'], userid_grouped, row['has_comment_flag']), axis=1)
    # # 打分总数
    # features['total_rating'] = features.apply(lambda row: total_rating(row['userid'], userid_grouped, row['has_comment_flag']), axis=1)
    # features['last_time_rating_ratio'] = 1.0 * features['last_time_rating'] / features['total_rating']
    # del features['last_time_rating']; del features['total_rating']

    # 最后几次是否存在低于 5 分的评论
    # features['last_3_order_has_lower5_rating'] = features.apply(lambda row: last_3_order_has_lower5_rating(row['userid'], userid_grouped, row['has_comment_flag'], 3), axis=1)

    # 标签特征
    # features['add_tags_ratio'] = features.apply(lambda row: add_tages_ratio(row['userid'], userid_grouped, row['has_comment_flag']), axis=1)
    # features['comment_keywords_count_sattistic'] = features.apply(lambda row: comment_tags_keywords_statistic(row['userid'], userid_grouped, row['has_comment_flag'], 'keywords_count'), axis=1)
    # features['comment_keywords_count_count_std'] = features['comment_keywords_count_sattistic'].map(lambda x: x[0])
    # features['comment_keywords_count_count_var'] = features['comment_keywords_count_sattistic'].map(lambda x: x[1])
    # features['comment_keywords_count_count_min'] = features['comment_keywords_count_sattistic'].map(lambda x: x[2])
    # features['comment_keywords_count_count_median'] = features['comment_keywords_count_sattistic'].map(lambda x: x[3])
    # features['comment_keywords_count_count_max'] = features['comment_keywords_count_sattistic'].map(lambda x: x[4])
    # features['comment_keywords_count_count_mean'] = features['comment_keywords_count_sattistic'].map(lambda x: x[5])
    # del features['comment_keywords_count_sattistic']

    return features


def avg_rating(feature, comment):
    """用户全部订单平均分，普通订单平均分，精品订单平均分"""
    feature_c = feature.copy()
    df = comment.groupby(['userid'])['rating'].mean().reset_index()
    df.columns = ['userid', 'avg_rating']
    feature_c = pd.merge(feature_c, df, on='userid', how='left')
    feature_c['avg_rating'] = feature_c['avg_rating'].fillna(0)

    comment_c = comment[comment['orderType'] == 0]
    df = comment_c.groupby(['userid'])['rating'].mean().reset_index()
    df.columns = ['userid', 'avg_rating_type0']
    feature_c = pd.merge(feature_c, df, on='userid', how='left')
    feature_c['avg_rating_type0'] = feature_c['avg_rating_type0'].fillna(0)

    comment_c = comment[comment['orderType'] == 1]
    df = comment_c.groupby(['userid'])['rating'].mean().reset_index()
    df.columns = ['userid', 'avg_rating_type1']
    feature_c = pd.merge(feature_c, df, on='userid', how='left')
    feature_c['avg_rating_type1'] = feature_c['avg_rating_type1'].fillna(0)
    del df, comment_c
    return feature_c

def rating_last_order(feature, comment, order):
    """用户最近一次订单打分"""
    feature_c = feature.copy()
    feature_c['rating_last_order'] = 0
    for i in xrange(feature_c.shape[0]):
        userid = feature.loc[i,'userid']
        df = order[(order['userid'] == userid) & (order['order_number'] == 1)]
        if df.shape[0]>0:
            comment_df = comment[comment['orderid'] == df['orderid'].values[0]]
            if comment_df.shape[0]>0:
                feature_c.loc[i,'rating_last_order'] = comment_df['rating'].values[0]
    return feature_c

def user_tag_score(feature, df):
    """用户全部订单tag平均分"""
    feature_c = feature.copy()
    df_select = df.groupby(['userid'])['tags_score'].mean().reset_index()
    df_select.columns = ['userid','tags_score']
    feature_c = pd.merge(feature_c, df_select, on = 'userid', how = 'left')
    feature_c['tags_score'] = feature_c['tags_score'].fillna(0)
    del df_select
    return feature_c

def tag_last_order(feature, comment, order):
    """用户最近一次订单tag打分"""
    feature_c = feature.copy()
    feature_c['tags_score_last_order'] = 0
    for i in tqdm(xrange(feature_c.shape[0])):
        userid = feature.loc[i,'userid']
        df = order[(order['userid'] == userid) & (order['order_number'] == 1)]
        if df.shape[0]>0:
            comment_df = comment[comment['orderid'] == df['orderid'].values[0]]
            if comment_df.shape[0]>0:
                feature_c.loc[i,'tags_score_last_order'] = comment_df['tags_score'].values[0]
    return feature_c

def user_comment_score(feature, df):
    """用户全部commentkey平均分"""
    feature_c = feature.copy()
    df_select = df.groupby(['userid'])['commentsKeyWords_score'].mean().reset_index()
    df_select.columns = ['userid','commentsKeyWords_score']
    feature_c = pd.merge(feature_c, df_select, on = 'userid', how = 'left')
    feature_c['commentsKeyWords_score'] = feature_c['commentsKeyWords_score'].fillna(-1)
    del df_select
    return feature_c


def comment_last_order(feature, comment, order):
    """用户最近一次订单comment打分"""
    feature_c = feature.copy()
    feature_c['commentsKeyWords_score_last_order'] = -1
    for i in tqdm(xrange(feature_c.shape[0])):
        userid = feature.loc[i,'userid']
        df = order[(order['userid'] == userid) & (order['order_number'] == 1)]
        if df.shape[0]>0:
            comment_df = comment[comment['orderid'] == df['orderid'].values[0]]
            if comment_df.shape[0]>0:
                feature_c.loc[i,'commentsKeyWords_score_last_order'] = comment_df['commentsKeyWords_score'].values[0]
    return feature_c

def built_comment_features_wxr(df, comments, order):
    features = pd.DataFrame({'userid': df['userid']})
    features = avg_rating(features,comments)
    features = rating_last_order(features, comments, order)
    features = user_tag_score(features, comments)
    features = tag_last_order(features, comments, order)
    features = user_comment_score(features, comments)
    features = comment_last_order(features, comments, order)
    return features

def main():
    feature_name = 'user_order_comment_features'
    if data_utils.is_feature_created(feature_name):
        return

    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')

    userComment_train = pd.read_csv(Configure.base_path + 'train/userComment_train.csv', encoding='utf8')
    userComment_test = pd.read_csv(Configure.base_path + 'test/userComment_test.csv', encoding='utf8')

    userComment_train.loc[userComment_train['rating'] == 4.33, 'rating'] = 4
    userComment_train.loc[userComment_train['rating'] == 3.67, 'rating'] = 4
    userComment_test.loc[userComment_train['rating'] == 2.33, 'rating'] = 2
    userComment_train['rating'] = userComment_train['rating'].astype(int)
    userComment_test['rating'] = userComment_test['rating'].astype(int)

    orderHistory_train = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_test.csv', encoding='utf8')
    userComment_train = pd.merge(userComment_train, orderHistory_train[['orderid', 'orderType']], on='orderid', how='left')
    userComment_test = pd.merge(userComment_test, orderHistory_test[['orderid', 'orderType']], on='orderid', how='left')
    userComment_train = commentKey_score(userComment_train)
    userComment_test = commentKey_score(userComment_test)
    userComment_train = tag_score(userComment_train)
    userComment_test = tag_score(userComment_test)
    print('save cleaned datasets')
    userComment_train.to_csv(Configure.cleaned_path + 'cleaned_userComment_train.csv', index=False, columns=userComment_train.columns, encoding='utf8')
    userComment_test.to_csv(Configure.cleaned_path + 'cleaned_userComment_test.csv', index=False, columns=userComment_test.columns, encoding='utf8')

    print('build train features')
    train_features = built_comment_features(train, userComment_train)
    print('build test features')
    test_features = built_comment_features(test, userComment_test)
    print('save ', feature_name)
    data_utils.save_features(train_features, test_features, feature_name)

    print('build wxr features')
    feature_name = 'user_order_comment_features_wxr'
    if not data_utils.is_feature_created(feature_name):
        print('build train action history features11')
        train_features = built_comment_features_wxr(train, userComment_train,orderHistory_train)
        print('build test action history features11')
        test_features = built_comment_features_wxr(test, userComment_test,orderHistory_test)
        print('save ', feature_name)
        data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 构造用户订单评论特征 ==========")
    main()
