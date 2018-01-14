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
from conf.configure import Configure
from utils import data_utils


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

    print('save cleaned datasets')
    userComment_train.to_csv(Configure.cleaned_path + 'cleaned_userComment_train.csv', index=False, columns=userComment_train.columns, encoding='utf8')
    userComment_test.to_csv(Configure.cleaned_path + 'cleaned_userComment_test.csv', index=False, columns=userComment_test.columns, encoding='utf8')

    print('build train features')
    train_features = built_comment_features(train, userComment_train)
    print('build test features')
    test_features = built_comment_features(test, userComment_test)

    print('save ', feature_name)
    data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 构造用户订单评论特征 ==========")
    main()
