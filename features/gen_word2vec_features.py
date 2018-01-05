#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 18-1-5 上午9:18
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
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from conf.configure import Configure
from utils import data_utils


def create_action_sentence(action_train, action_test):
    """ actiontype 构造成可训练词向量的格式并写入文件 """
    train_users = action_train['userid'].unique()
    test_users = action_test['userid'].unique()
    train_action_grouped = dict(list(action_train.groupby('userid')))
    test_action_grouped = dict(list(action_test.groupby('userid')))

    with open('action_corpus.txt', 'w') as action_corpus:
        for uid in train_users:
            df = train_action_grouped[uid]
            days = df['days_from_now'].unique()
            for day in days:
                actions = df[df['days_from_now'] == day]['actionType'].astype(str).values.tolist()
                actions = ' '.join(actions)
                action_corpus.writelines(actions + '\n')

        for uid in test_users:
            df = test_action_grouped[uid]
            days = df['days_from_now'].unique()
            for day in days:
                actions = df[df['days_from_now'] == day]['actionType'].astype(str).values.tolist()
                actions = ' '.join(actions)
                action_corpus.writelines(actions + '\n')


def train_word2vec(action_train, action_test):
    """ 训练词向量 """
    print('create action sentence')
    if not os.path.exists('action_corpus.txt'):
        create_action_sentence(action_train, action_test)

    print('train word2vec')
    word_embedding = 100
    corpus = './action_corpus.txt'
    word2vec_model = 'actiontype_word2vec.model'
    word2vec_vector = 'actiontype_word2vec.vector'

    model = Word2Vec(LineSentence(corpus), size=word_embedding, window=2, min_count=2,
                     workers=multiprocessing.cpu_count())

    model.save(word2vec_model)
    model.wv.save_word2vec_format(word2vec_vector, binary=False)


def create_actions_doc2vec(uid, action_grouped, word2vec_model):
    df = action_grouped[uid]
    actions = df['actionType'].astype(str).values.tolist()[-10:]
    doc2mat = []
    for action in actions:
        doc2mat.append(word2vec_model[action])
    doc2mat = np.array(doc2mat)
    doc2vec = np.mean(doc2mat, axis=0)
    return doc2vec.tolist()


def build_doc2vec_features(df, action, word2vec_model):
    features = pd.DataFrame({'userid': df['userid']})

    users = action['userid'].unique()
    action_grouped = dict(list(action.groupby('userid')))

    features['actions_doc2vec'] = features.apply(lambda row: create_actions_doc2vec(row['userid'], action_grouped, word2vec_model), axis=1)
    for i in range(100):
        features['doc2vec_{}'.format(i)] = features['actions_doc2vec'].map(lambda x: x[i])
    del features['actions_doc2vec']
    return features


def main():
    feature_name = 'word2vec_features'
    if data_utils.is_feature_created(feature_name):
        return

    print('load dataset')
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')
    action_train = pd.read_csv(Configure.cleaned_path + 'cleaned_action_train.csv')
    action_test = pd.read_csv(Configure.cleaned_path + 'cleaned_action_test.csv')
    action_train.sort_values(by='actionTime', inplace=True)
    action_test.sort_values(by='actionTime', inplace=True)

    # 训练词向量
    train_word2vec(action_train, action_test)
    # 加载生成的词向量模型
    word2vec_model = Word2Vec.load('actiontype_word2vec.model')

    print('build_doc2vec_features')
    train_features = build_doc2vec_features(train, action_train, word2vec_model)
    test_features = build_doc2vec_features(test, action_test, word2vec_model)
    print('save ', feature_name)
    data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 构造 action 词向量特征 ==========")
    main()
