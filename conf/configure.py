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

    # 生成的特征的路径
    features_path = base_path + '/features/'
    # 生成的模型可训练和预测的数据集
    datasets_path = base_path + '/datasets/'

    # 待 merge 的特征（特征名：merge_on 的特征）
    features = {
        'basic_user_info'            : {'on': 'userid', 'how': 'left'},
        'basic_user_action_features' : {'on': 'userid', 'how': 'left'},
        'user_order_history_features': {'on': 'userid', 'how': 'left'},
    }

    submit_result_path = '../result/'
