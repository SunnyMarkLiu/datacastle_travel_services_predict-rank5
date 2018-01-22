#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-26 下午3:14
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)


class Configure(object):
    base_path = '/d_2t/lq/competitions/data_castle/Datacastle_Travel_Services_Predict/'

    # 数据清晰后的路径
    cleaned_path = base_path + 'cleaned/'
    # 生成的特征的路径
    features_path = base_path + 'features/'
    # 生成的模型可训练和预测的数据集
    datasets_path = base_path + 'datasets/'

    # 待 merge 的特征（特征名：merge_on 的特征）
    features = {
        'basic_user_info'             : {'on': 'userid', 'how': 'left'},
        'basic_user_action_features'  : {'on': 'userid', 'how': 'left'},
        'user_order_history_features' : {'on': 'userid', 'how': 'left'},
        'user_order_history_features2': {'on': 'userid', 'how': 'left'},
        'user_order_history_features3': {'on': 'userid', 'how': 'left'},
        'user_order_history_features4': {'on': 'userid', 'how': 'left'},
        'user_order_comment_features' : {'on': 'userid', 'how': 'left'},
        'action_history_features'     : {'on': 'userid', 'how': 'left'},
        'action_history_features2'    : {'on': 'userid', 'how': 'left'},
        'action_history_features3'    : {'on': 'userid', 'how': 'left'},
        'action_history_features4'    : {'on': 'userid', 'how': 'left'},
        'action_history_features5'    : {'on': 'userid', 'how': 'left'},
        'action_history_features6'    : {'on': 'userid', 'how': 'left'},
        # 'action_history_features7'   : {'on': 'userid', 'how': 'left'},
        'action_history_features8'    : {'on': 'userid', 'how': 'left'},
        'action_history_features9'    : {'on': 'userid', 'how': 'left'},
        'action_history_features10'   : {'on': 'userid', 'how': 'left'},
        'action_history_features11'   : {'on': 'userid', 'how': 'left'},
        # 'action_history_features12'   : {'on': 'userid', 'how': 'left'},
        # 'baseline_features'          : {'on': 'userid', 'how': 'left'},
        # 'word2vec_features'          : {'on': 'userid', 'how': 'left'},
        'wxr_features'                : {'on': 'userid', 'how': 'left'},
        # 'sqg_features'                : {'on': 'userid', 'how': 'left'}
    }

    # 新提取的特征
    new_features = {
        'advance_order_history_features' : {'on': 'userid', 'how': 'left'},
        # 'advance_action_features'        : {'on': 'userid', 'how': 'left'},
        # 'sqg_features'                : {'on': 'userid', 'how': 'left'}
    }

    # 特征选择后各模型最佳特征保存路径
    xgboost_best_subfeatures = '../model/xgboost_best_subfeatures/'
    lightgbm_best_subfeatures = '../model/lightgbm_best_subfeatures/'
    catboost_best_subfeatures = '../model/catboost_best_subfeatures/'

    # 特征选择后各模型最佳特征保存路径
    xgboost_removed_subfeatures = '../model/xgboost_removed_subfeatures/'

    submit_result_path = '../result/'
