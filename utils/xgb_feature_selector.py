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
                           # verbose_eval=100,
                           # show_stdv=False,
                           )
        best_num_boost_rounds = len(cv_result)
        mean_test_auc = cv_result.loc[best_num_boost_rounds - 11: best_num_boost_rounds - 1, 'test-auc-mean'].mean()
        return mean_test_auc


    def select_best_subset_features(self, xgb_params, cv_nfold, selected_feature_size, num_boost_round, base_features,
                                    save_tmp_features_path, early_stopping_rounds, maximize=True, stratified=True, shuffle=True):
        """
        :param save_tmp_features_path: save tmp sub-features path
        :param base_features: features which start greedy search
        :param shuffle: shuffle the datas
        :param stratified: label balance
        :param early_stopping_rounds: early stop round
        :param num_boost_round: xgboost runing max num_boost_round
        :param cv_nfold: cv fold
        :param xgb_params: xgboost params
        :param selected_feature_size: int or float, select subset features (浮点数后期完善)
        :param maximize: bool, Whether to maximize metric.
        :return:
        """
        if base_features is not None and len(base_features) > 0:
            print('calc metric using base features')
            metric_score = self._cross_validate_score(xgb_params=xgb_params, sub_features=base_features,
                                                      nfold=cv_nfold, num_boost_round=num_boost_round,
                                                      early_stopping_rounds=early_stopping_rounds,
                                                      stratified=stratified, shuffle=shuffle)
            print('base metric: ', metric_score)
            best_subset_features = base_features
            metric_scores_historys = [0] * (len(base_features) - 1) + [metric_score]
        else:
            best_subset_features = []
            metric_scores_historys = []

        while len(metric_scores_historys) <= 2 or (metric_scores_historys[-1] > metric_scores_historys[-2] if maximize
                        else metric_scores_historys[-1] < metric_scores_historys[-2]):   # <=2 为初始条件；进行贪心的下一步的条件
            if len(best_subset_features) == selected_feature_size:
                break

            metric_scores = []
            for feature in self.X.columns.values:
                if feature not in best_subset_features:
                    tmp_subset_features = best_subset_features + [feature]
                    # calc metric using current features
                    metric_score = self._cross_validate_score(xgb_params=xgb_params, sub_features=tmp_subset_features,
                                                              nfold=cv_nfold, num_boost_round=num_boost_round,
                                                              early_stopping_rounds=early_stopping_rounds,
                                                              stratified=stratified, shuffle=shuffle)
                    metric_scores.append((metric_score, feature))   # add features and this metric
                    print('---> add {}, mean cv test score: {:.5f}'.format(feature, metric_score))
            best_subset_features.append(sorted(metric_scores)[-1][1])  # only add the feature which get "the biggest gain score"
            metric_scores_historys.append(sorted(metric_scores)[-1][0])  # only add the biggest gain score
            print('current feature size: {}, mean cv metric score: {}'.format(len(best_subset_features), metric_scores_historys[-1]))

            best_subset_features_path = save_tmp_features_path + '/best_subset_{}_features_cv_{}.pkl'.format(
                                            len(best_subset_features),
                                            metric_scores_historys[-1]
                                        )
            with open(best_subset_features_path, "wb") as f:
                cPickle.dump(best_subset_features, f, -1)

        print('===> final selected feature size:', len(best_subset_features))
        return best_subset_features
