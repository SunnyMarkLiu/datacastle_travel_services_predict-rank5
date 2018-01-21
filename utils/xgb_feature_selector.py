#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
贪心算法实现特征筛选

@author: SunnyMarkLiu
@time  : 18-1-18 下午8:56
"""
from __future__ import absolute_import, division, print_function

import os
import sys

import cPickle

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings
warnings.filterwarnings('ignore')

import time
from Queue import Queue
import threading
import xgboost as xgb


class XgboostGreedyFeatureSelector(object):
    def __init__(self, X, y):
        """
        :param X: features
        :param y: labels

        """
        self.X = X
        self.y = y

    def _cross_validate_score(self, xgb_params, sub_features, nfold, num_boost_round, early_stopping_rounds,
                              stratified=True, shuffle=True):
        """
        run cross validate to calc features using these features
        :return:
        """
        dtrain = xgb.DMatrix(self.X[sub_features].values, self.y, feature_names=sub_features)
        cv_result = xgb.cv(dict(xgb_params),
                           dtrain,
                           nfold=nfold,
                           stratified=stratified,
                           num_boost_round=num_boost_round,
                           early_stopping_rounds=early_stopping_rounds,
                           shuffle=shuffle,
                           # verbose_eval=50,
                           # show_stdv=False,
                           )
        best_num_boost_rounds = len(cv_result)
        mean_test_auc = cv_result.loc[best_num_boost_rounds - 11: best_num_boost_rounds - 1, 'test-auc-mean'].mean()
        return mean_test_auc

    def remove_worst_features(self, xgb_params, cv_nfold, remove_ratio, num_boost_round, thread_size, save_removed_features_path,
                              early_stopping_rounds, stratified=True, shuffle=True, decrease_auc_threshold=0.0005):
        """
        remove worst features to improve score
        """
        original_features = self.X.columns.values.tolist()
        should_removed_feature_count = int(len(original_features) * remove_ratio)

        print('calc metric using original features')
        base_metric_score = self._cross_validate_score(xgb_params=xgb_params, sub_features=original_features,
                                                       nfold=cv_nfold, num_boost_round=num_boost_round,
                                                       early_stopping_rounds=early_stopping_rounds,
                                                       stratified=stratified, shuffle=shuffle)
        print('base metric: ', base_metric_score)

        removed_features = []

        best_metric_score = base_metric_score
        best_removed_features = removed_features

        while len(removed_features) < should_removed_feature_count:

            print('left removed features:', should_removed_feature_count - len(removed_features))

            start = time.clock()

            left_selected_features = list(set(original_features) - set(removed_features))

            metric_queue = Queue()

            def _thread_clac_metric(thread_features):
                for feature in thread_features:
                    sub_features = list(set(left_selected_features) - {feature})
                    # calc metric using current features
                    metric_score = self._cross_validate_score(xgb_params=xgb_params, sub_features=sub_features,
                                                              nfold=cv_nfold, num_boost_round=num_boost_round,
                                                              early_stopping_rounds=early_stopping_rounds,
                                                              stratified=stratified, shuffle=shuffle)
                    metric_queue.put((metric_score, feature))
                    if metric_queue.qsize() % 50 == 0:
                        print("left: {}, processed: {}, mean cv test score: {:.5f}".format(
                            len(left_selected_features), metric_queue.qsize(), metric_score))

            # starting multi-thread to clac metric on GPU
            threads = []
            for i in range(thread_size):
                delta = int(len(left_selected_features) / thread_size)
                start_i = i * delta
                if i == thread_size - 1:
                    end_i = len(left_selected_features)
                else:
                    end_i = (i + 1) * delta

                # print('thread_{} process features: {} ~ {}'.format(i, start_i, end_i))
                t = threading.Thread(target=_thread_clac_metric, name='thread_{}'.format(i),
                                     args=(left_selected_features[start_i: end_i],))
                threads.append(t)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()  # wait for the thread to finish.

            metric_scores = []
            while not metric_queue.empty():
                metric_scores.append(metric_queue.get())

            metric_scores = sorted(metric_scores)

            elapsed = (time.clock() - start)

            if best_metric_score < metric_scores[-1][0]:
                print('===> remove {}, auc imporved from {} -> {}, cost {}s'.format(
                    metric_scores[-1][1], best_metric_score, metric_scores[-1][0], elapsed))

                removed_features.append(metric_scores[-1][1])  # only add the feature which get "the biggest gain score"
                best_metric_score = metric_scores[-1][0]
                best_removed_features = removed_features

                best_subset_features_path = save_removed_features_path + '/best_cv_{}_removed_feature_count_{}.pkl'.format(
                    best_metric_score,
                    len(best_removed_features)
                )
                with open(best_subset_features_path, "wb") as f:
                    cPickle.dump(best_removed_features, f, -1)

            elif best_metric_score - metric_scores[-1][0] < decrease_auc_threshold:     # 下降在一定范围内继续迭代
                print('===> remove {}, decrease auc a little, from {} -> {}, cost {}s'.format(
                    metric_scores[-1][1], best_metric_score, metric_scores[-1][0], elapsed))
                removed_features.append(metric_scores[-1][1])
            else:   # 下降超出范围，结束迭代
                break

            print('===> removed feature size:', len(best_removed_features))

        print('===> final removed feature size:', len(best_removed_features))
        return best_removed_features



    def select_best_subset_features(self, xgb_params, cv_nfold, selected_feature_size, num_boost_round, base_features,
                                    thread_size, save_tmp_features_path, early_stopping_rounds, stratified=True,
                                    shuffle=True, decrease_auc_threshold=0.001):
        """
        :param decrease_auc_threshold: decrease threshold, default 0.001
        :param thread_size: multi thread
        :param save_tmp_features_path: save tmp sub-features path
        :param base_features: features which start greedy search
        :param shuffle: shuffle the datas
        :param stratified: label balance
        :param early_stopping_rounds: early stop round
        :param num_boost_round: xgboost runing max num_boost_round
        :param cv_nfold: cv fold
        :param xgb_params: xgboost params
        :param selected_feature_size: int or float, select subset features (浮点数后期完善)
        :return:
        """
        total_x_features = set(self.X.columns.values.tolist())

        if base_features is not None and len(base_features) > 0:
            print('calc metric using base features')
            base_metric_score = self._cross_validate_score(xgb_params=xgb_params, sub_features=base_features,
                                                      nfold=cv_nfold, num_boost_round=num_boost_round,
                                                      early_stopping_rounds=early_stopping_rounds,
                                                      stratified=stratified, shuffle=shuffle)
            print('base metric: ', base_metric_score)
            best_subset_features = base_features
            metric_scores_historys = [0] * (len(base_features) - 1) + [base_metric_score]
        else:
            best_subset_features = []
            metric_scores_historys = []

        while len(metric_scores_historys) <= 2 or \
                (metric_scores_historys[-1] > metric_scores_historys[-2]) or \
                    (metric_scores_historys[-2] - metric_scores_historys[-1] < decrease_auc_threshold):   # <=2 为初始条件；进行贪心的下一步的条件

            if len(best_subset_features) == selected_feature_size:
                break

            left_selected_features = list(total_x_features - set(best_subset_features))

            if len(left_selected_features) <= 0:
                break

            metric_queue = Queue()
            def _thread_clac_metric(thread_features):
                for feature in thread_features:
                    tmp_subset_features = best_subset_features + [feature]
                    # calc metric using current features
                    metric_score = self._cross_validate_score(xgb_params=xgb_params, sub_features=tmp_subset_features,
                                                              nfold=cv_nfold, num_boost_round=num_boost_round,
                                                              early_stopping_rounds=early_stopping_rounds,
                                                              stratified=stratified, shuffle=shuffle)
                    metric_queue.put((metric_score, feature))
                    if metric_queue.qsize() % 50 == 0:
                        print("left: {}, processed: {}, mean cv test score: {:.5f}".format(
                                    len(left_selected_features), metric_queue.qsize(), metric_score))

            # starting multi-thread to clac metric on GPU
            threads = []
            for i in range(thread_size):
                delta = int(len(left_selected_features) / thread_size)
                start_i = i * delta
                if i == thread_size - 1:
                    end_i = len(left_selected_features)
                else:
                    end_i = (i + 1) * delta

                # print('thread_{} process features: {} ~ {}'.format(i, start_i, end_i))
                t = threading.Thread(target=_thread_clac_metric, name='thread_{}'.format(i), args=(left_selected_features[start_i: end_i],))
                threads.append(t)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()   # wait for the thread to finish.

            metric_scores = []
            while not metric_queue.empty():
                metric_scores.append(metric_queue.get())

            best_subset_features.append(sorted(metric_scores)[-1][1])  # only add the feature which get "the biggest gain score"
            metric_scores_historys.append(sorted(metric_scores)[-1][0])  # only add the biggest gain score
            print('===>> add {}, current feature size: {}, mean cv metric score: {}'.format(best_subset_features[-1], len(best_subset_features), metric_scores_historys[-1]))

            best_subset_features_path = save_tmp_features_path + '/best_cv_{}_feature_count_{}.pkl'.format(
                                            metric_scores_historys[-1],
                                            len(best_subset_features)
                                        )
            with open(best_subset_features_path, "wb") as f:
                cPickle.dump(best_subset_features, f, -1)

        print('===> final selected feature size:', len(best_subset_features))
        return best_subset_features
