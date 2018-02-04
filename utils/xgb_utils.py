#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-1 下午5:21
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd


def get_xgb_importance(clf, features):

    weughts_imp = clf.get_score(importance_type='weight')
    gains_imp = clf.get_score(importance_type='gain')
    covers_imp = clf.get_score(importance_type='cover')

    weights = []
    gains = []
    covers = []
    for f in features:
        weights.append(weughts_imp.get(f, 0))
        gains.append(gains_imp.get(f, 0))
        covers.append(covers_imp.get(f, 0))

    features_imp = pd.DataFrame({'feature': features, 'weights': weights, 'gains': gains, 'covers': covers})
    sum_weight = sum(features_imp['weights']) * 1.0
    features_imp['weights'] = features_imp['weights'] / sum_weight
    features_imp['importance'] = features_imp['weights'] + features_imp['gains'] + features_imp['covers']
    del features_imp['weights']
    del features_imp['gains']
    del features_imp['covers']

    impdf = features_imp.sort_values(by='importance', ascending=False).reset_index(drop=True)
    impdf['importance'] /= impdf['importance'].sum()
    return impdf


def get_xgb_importance_by_weights(clf, features):

    weughts_imp = clf.get_score(importance_type='weight')

    weights = []
    for f in features:
        weights.append(weughts_imp.get(f, 0))

    features_imp = pd.DataFrame({'feature': features, 'weights': weights})
    impdf = features_imp.sort_values(by='weights', ascending=False).reset_index(drop=True)

    return impdf


def get_xgb_importance_by_gains(clf, features):

    gains_imp = clf.get_score(importance_type='gain')

    gains = []
    for f in features:
        gains.append(gains_imp.get(f, 0))

    features_imp = pd.DataFrame({'feature': features, 'gains': gains})
    impdf = features_imp.sort_values(by='gains', ascending=False).reset_index(drop=True)

    return impdf


def get_xgb_importance_by_covers(clf, features):

    covers_imp = clf.get_score(importance_type='cover')

    covers = []
    for f in features:
        covers.append(covers_imp.get(f, 0))

    features_imp = pd.DataFrame({'feature': features, 'covers': covers})

    impdf = features_imp.sort_values(by='covers', ascending=False).reset_index(drop=True)
    return impdf
