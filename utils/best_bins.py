# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:40:46 2017

@author: Hank Kuang
@title: 最优分箱&最优降基
"""

import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target


def _check_target_binary(y):
    """
    check if the target variable is binary
    ------------------------------
    Param
    y:exog variable,pandas Series contains binary variable
    ------------------------------
    Return
    if y is not binary, raise a error
    """
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('目标变量必须是二元的！')


def _isNullZero(x):
    """
    check x is null or equal zero
    -----------------------------
    Params
    x: data
    -----------------------------
    Return
    bool obj
    """
    cond1 = np.isnan(x)
    cond2 = x == 0
    return cond1 or cond2


def _Gvalue(binDS, method):
    """
    Calculation of the metric of current split
    ----------------------------------------
    Params
    binDS: pandas dataframe
    method: int obj, metric to split x(1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    -----------------------------------------
    Return
    M_value: float or np.nan
    """
    R = binDS['bin'].max()
    N = binDS['total'].sum()

    N_mat = np.empty((R, 3))
    # calculate sum of 0,1
    N_s = [binDS[0].sum(), binDS[1].sum()]
    # calculate each bin's sum of 0,1,total
    # store values in R*3 ndarray
    for i in range(int(R)):
        subDS = binDS[binDS['bin'] == (i + 1)]
        N_mat[i][0] = subDS[0].sum()
        N_mat[i][1] = subDS[1].sum()
        N_mat[i][2] = subDS['total'].sum()

    # Gini
    if method == 1:
        G_list = [0] * R
        for i in range(int(R)):

            for j in range(2):
                G_list[i] = G_list[i] + N_mat[i][j] * N_mat[i][j]
            G_list[i] = 1 - G_list[i] / (N_mat[i][2] * N_mat[i][2])
        G = 0
        for j in range(2):
            G = G + N_s[j] * N_s[j]

        G = 1 - G / (N * N)
        Gr = 0
        for i in range(int(R)):
            Gr = Gr + N_mat[i][2] * (G_list[i] / N)
        M_value = 1 - Gr / G
    # Entropy
    elif method == 2:
        for i in range(int(R)):
            for j in range(2):
                if np.isnan(N_mat[i][j]) or N_mat[i][j] == 0:
                    M_value = 0

        E_list = [0] * R
        for i in range(int(R)):
            for j in range(2):
                E_list[i] = E_list[i] - ((N_mat[i][j] / float(N_mat[i][2])) * np.log(N_mat[i][j] / N_mat[i][2]))

            E_list[i] = E_list[i] / np.log(2)  # plus
        E = 0
        for j in range(2):
            a = (N_s[j] / N)
            E = E - a * (np.log(a))

        E = E / np.log(2)
        Er = 0
        for i in range(2):
            Er = Er + N_mat[i][2] * E_list[i] / N
        M_value = 1 - (Er / E)
        return M_value
    # Pearson X2
    elif method == 3:
        N = N_s[0] + N_s[1]
        X2 = 0
        M = np.empty((R, 2))
        for i in range(int(R)):
            for j in range(2):
                M[i][j] = N_mat[i][2] * N_s[j] / N
                X2 = X2 + (N_mat[i][j] - M[i][j]) * (N_mat[i][j] - M[i][j]) / (M[i][j])

        M_value = X2
    # Info value
    else:
        if any([_isNullZero(N_mat[i][0]),
                _isNullZero(N_mat[i][1]),
                _isNullZero(N_s[0]),
                _isNullZero(N_s[1])]):
            M_value = np.NaN
        else:
            IV = 0
            for i in range(int(R)):
                IV = IV + (N_mat[i][0] / N_s[0] - N_mat[i][1] / N_s[1]) \
                     * np.log((N_mat[i][0] * N_s[1]) / (N_mat[i][1] * N_s[0]))
            M_value = IV

    return M_value


def _calCMerit(temp, ix, method):
    """
    Calculation of the merit function for the current table temp
    ---------------------------------------------
    Params
    temp: pandas dataframe, temp table in _bestSplit
    ix: single int obj,index of temp, from length of temp
    method: int obj, metric to split x(1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    ---------------------------------------------
    Return
    M_value: float or np.nan
    """
    # split data by ix
    temp_L = temp[temp['i'] <= ix]
    temp_U = temp[temp['i'] > ix]
    # calculate sum of 0, 1, total for each splited data
    n_11 = float(sum(temp_L[0]))
    n_12 = float(sum(temp_L[1]))
    n_21 = float(sum(temp_U[0]))
    n_22 = float(sum(temp_U[1]))
    n_1s = float(sum(temp_L['total']))
    n_2s = float(sum(temp_U['total']))
    # calculate sum of 0, 1 for whole data
    n_s1 = float(sum(temp[0]))
    n_s2 = float(sum(temp[1]))
    N_mat = np.array([[n_11, n_12, n_1s],
                      [n_21, n_22, n_2s]])
    N_s = [n_s1, n_s2]
    # Gini
    if method == 1:
        N = n_1s + n_2s
        G1 = 1 - ((n_11 * n_11 + n_12 * n_12) / float(n_1s * n_1s))
        G2 = 1 - ((n_21 * n_21 + n_22 * n_22) / float(n_2s * n_2s))
        G = 1 - ((n_s1 * n_s1 + n_s2 * n_s2) / float(N * N))
        M_value = 1 - ((n_1s * G1 + n_2s * G2) / float(N * G))
    # Entropy
    elif method == 2:
        N = n_1s + n_2s
        E1 = -((n_11 / n_1s) * (np.log((n_11 / n_1s))) + (n_12 / n_1s) * (np.log((n_12 / n_1s)))) / (np.log(2))
        E2 = -((n_21 / n_2s) * (np.log((n_21 / n_2s))) + (n_22 / n_2s) * (np.log((n_22 / n_2s)))) / (np.log(2))
        E = -(((n_s1 / N) * (np.log((n_s1 / N))) + ((n_s2 / N) * np.log((n_s2 / N)))) / (np.log(2)))
        M_value = 1 - (n_1s * E1 + n_2s * E2) / (N * E)
    # Pearson chisq
    elif method == 3:
        N = n_1s + n_2s
        X2 = 0
        M = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                M[i][j] = N_mat[i][2] * N_s[j] / N
                X2 = X2 + ((N_mat[i][j] - M[i][j]) * (N_mat[i][j] - M[i][j])) / M[i][j]

        M_value = X2
    # Info Value
    else:
        try:
            IV = ((n_11 / n_s1) - (n_12 / n_s2)) * np.log((n_11 * n_s2) / (n_12 * n_s1)) \
                 + ((n_21 / n_s1) - (n_22 / n_s2)) * np.log((n_21 * n_s2) / (n_22 * n_s1))
            M_value = IV
        except ZeroDivisionError:
            M_value = np.nan
    return M_value


def _bestSplit(binDS, method, BinNo):
    """
    find the best split for one bin dataset
    middle procession functions for _candSplit
    --------------------------------------
    Params
    binDS: pandas dataframe, middle bining table
    method: int obj, metric to split x
        (1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    BinNo: int obj, bin number of binDS
    --------------------------------------
    Return
    newbinDS: pandas dataframe
    """
    binDS = binDS.sort_values(by=['bin', 'pdv1'])
    mb = len(binDS[binDS['bin'] == BinNo])

    bestValue = 0
    bestI = 1
    for i in range(1, mb):
        # split data by i
        # metric: Gini,Entropy,pearson chisq,Info value
        value = _calCMerit(binDS, i, method)
        # if value>bestValue，then make value=bestValue，and bestI = i
        if bestValue < value:
            bestValue = value
            bestI = i
    # create new var split
    binDS['split'] = np.where(binDS['i'] <= bestI, 1, 0)
    binDS = binDS.drop('i', axis=1)
    newbinDS = binDS.sort_values(by=['split', 'pdv1'])
    # rebuild var i
    newbinDS_0 = newbinDS[newbinDS['split'] == 0]
    newbinDS_1 = newbinDS[newbinDS['split'] == 1]
    newbinDS_0['i'] = range(1, len(newbinDS_0) + 1)
    newbinDS_1['i'] = range(1, len(newbinDS_1) + 1)
    newbinDS = pd.concat([newbinDS_0, newbinDS_1], axis=0)
    return newbinDS  # .sort_values(by=['split','pdv1'])


def _candSplit(binDS, method):
    """
    Generate all candidate splits from current Bins
    and select the best new bins
    middle procession functions for binContVar & reduceCats
    ---------------------------------------------
    Params
    binDS: pandas dataframe, middle bining table
    method: int obj, metric to split x
        (1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    --------------------------------------------
    Return
    newBins: pandas dataframe, split results
    """
    # sorted data by bin&pdv1
    binDS = binDS.sort_values(by=['bin', 'pdv1'])
    # get the maximum of bin
    Bmax = max(binDS['bin'])
    # screen data and cal nrows by diffrence bin
    # and save the results in dict
    temp_binC = dict()
    m = dict()
    for i in range(1, Bmax + 1):
        temp_binC[i] = binDS[binDS['bin'] == i]
        m[i] = len(temp_binC[i])
    """
    CC
    """
    # create null dataframe to save info
    temp_trysplit = dict()
    temp_main = dict()
    bin_i_value = []
    for i in range(1, Bmax + 1):
        if m[i] > 1:  # if nrows of bin > 1
            # split data by best i
            temp_trysplit[i] = _bestSplit(temp_binC[i], method, i)
            temp_trysplit[i]['bin'] = np.where(temp_trysplit[i]['split'] == 1,
                                               Bmax + 1,
                                               temp_trysplit[i]['bin'])
            # delete bin == i
            temp_main[i] = binDS[binDS['bin'] != i]
            # vertical combine temp_main[i] & temp_trysplit[i]
            temp_main[i] = pd.concat([temp_main[i], temp_trysplit[i]], axis=0)
            # calculate metric of temp_main[i]
            value = _Gvalue(temp_main[i], method)
            newdata = [i, value]
            bin_i_value.append(newdata)
    # find maxinum of value bintoSplit
    bin_i_value.sort(key=lambda x: x[1], reverse=True)
    # binNum = temp_all_Vals['BinToSplit']
    binNum = bin_i_value[0][0]
    newBins = temp_main[binNum].drop('split', axis=1)
    return newBins.sort_values(by=['bin', 'pdv1'])


def _EqualWidthBinMap(x, Acc, adjust):
    """
    Data bining function,
    middle procession functions for binContVar
    method: equal width
    Mind: Generate bining width and interval by Acc
    --------------------------------------------
    Params
    x: pandas Series, data need to bining
    Acc: float less than 1, partition ratio for equal width bining
    adjust: float or np.inf, bining adjust for limitation
    --------------------------------------------
    Return
    bin_map: pandas dataframe, Equal width bin map
    """
    varMax = x.max()
    varMin = x.min()
    # generate range by Acc
    Mbins = int(1. / Acc)
    minMaxSize = (varMax - varMin) / Mbins
    # get upper_limit and loewe_limit
    ind = range(1, Mbins + 1)
    Upper = pd.Series(index=ind, name='upper')
    Lower = pd.Series(index=ind, name='lower')
    for i in ind:
        Upper[i] = varMin + i * minMaxSize
        Lower[i] = varMin + (i - 1) * minMaxSize

    # adjust the min_bin's lower and max_bin's upper
    Upper[Mbins] = Upper[Mbins] + adjust
    Lower[1] = Lower[1] - adjust
    bin_map = pd.concat([Lower, Upper], axis=1)
    bin_map.index.name = 'bin'
    return bin_map


def _applyBinMap(x, bin_map):
    """
    Generate result of bining by bin_map
    ------------------------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, map table
    ------------------------------------------------
    Return
    bin_res: pandas Series, result of bining
    """
    bin_res = np.array([0] * x.shape[-1], dtype=int)

    for i in bin_map.index:
        upper = bin_map['upper'][i]
        lower = bin_map['lower'][i]
        x1 = x[np.where((x >= lower) & (x <= upper))[0]]
        mask = np.in1d(x, x1)
        bin_res[mask] = i

    bin_res = pd.Series(bin_res, index=x.index)
    bin_res.name = x.name + "_BIN"

    return bin_res


def _combineBins(temp_cont, target):
    """
    merge all bins that either 0 or 1 or total =0
    middle procession functions for binContVar
    ---------------------------------
    Params
    temp_cont: pandas dataframe, middle results of binContVar
    target: target label
    --------------------------------
    Return
    temp_cont: pandas dataframe
    """
    for i in temp_cont.index:
        rowdata = temp_cont.ix[i, :]

        if i == temp_cont.index.max():
            ix = temp_cont[temp_cont.indexi].index.min()
        if any(rowdata[:3] == 0):  # 如果0,1,total有一项为0，则运行
            #
            temp_cont.ix[ix, target] = temp_cont.ix[ix, target] + rowdata[target]
            temp_cont.ix[ix, 0] = temp_cont.ix[ix, 0] + rowdata[0]
            temp_cont.ix[ix, 'total'] = temp_cont.ix[ix, 'total'] + rowdata['total']
            #
            if i < temp_cont.index.max():
                temp_cont.ix[ix, 'lower'] = rowdata['lower']
            else:
                temp_cont.ix[ix, 'upper'] = rowdata['upper']
            temp_cont = temp_cont.drop(i, axis=0)

    return temp_cont.sort_values(by='pdv1')


def _getNewBins(sub, i):
    """
    get new lower, upper, bin, total for sub
    middle procession functions for binContVar
    -----------------------------------------
    Params
    sub: pandas dataframe, subdataframe of temp_map
    i: int, bin number of sub
    ----------------------------------------
    Return
    df: pandas dataframe, one row
    """
    l = len(sub)
    total = sub['total'].sum()
    first = sub.iloc[0, :]
    last = sub.iloc[l - 1, :]

    lower = first['lower']
    upper = last['upper']
    df = pd.DataFrame()
    df = df.append([i, lower, upper, total], ignore_index=True).T
    df.columns = ['bin', 'lower', 'upper', 'total']
    return df


def _groupCal(x, y, badlabel=1):
    """
    group calulate for x by y
    middle proporcessing function for reduceCats
    -------------------------------------
    Params
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    badlabel: target label
    ------------------------------------
    Return
    temp_cont: group calulate table
    m: nrows of temp_cont
    """

    temp_cont = pd.crosstab(index=x, columns=y, margins=False)
    temp_cont['total'] = temp_cont.sum(axis=1)
    temp_cont['pdv1'] = temp_cont[badlabel] / temp_cont['total']

    temp_cont['i'] = range(1, temp_cont.shape[0] + 1)
    temp_cont['bin'] = 1
    m = temp_cont.shape[0]
    return temp_cont, m


def binContVar(x, y, method, mmax=5, Acc=0.01, target=1, adjust=0.0001):
    """
    Optimal binings for contiouns var x by (y & method)
    method is represent by number,
        1:Gini, 2:Entropy, 3:person chisq, 4:Info value
    ---------------------------------------------
    Params
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    method: int obj, metric to split x
    mmax: int, bining number
    Acc: float less than 1, partition ratio for equal width bining
    badlabel: target label
    adjust: float or np.inf, bining adjust for limitation
    ---------------------------------------------
    Return
    temp_Map: pandas dataframe, Optimal bining map
    """
    # if y is not 0-1 binary variable, then raise a error
    _check_target_binary(y)
    # data bining by Acc, method: width equal
    bin_map = _EqualWidthBinMap(x, Acc, adjust=adjust)
    # mapping x to bin number and combine with x&y
    bin_res = _applyBinMap(x, bin_map)
    temp_df = pd.concat([x, y, bin_res], axis=1)
    # calculate freq of 0, 1 in y group by bin_res
    t1 = pd.crosstab(index=temp_df[bin_res.name], columns=y)
    # calculate freq of bin, and combine with t1
    t2 = temp_df.groupby(bin_res.name).count().ix[:, 0]
    t2 = pd.DataFrame(t2)
    t2.columns = ['total']
    t = pd.concat([t1, t2], axis=1)
    # merge t & bin_map by t,
    # if all(0,1,total) == 1, so corresponding row will not appear in temp_cont
    temp_cont = pd.merge(t, bin_map,
                         left_index=True, right_index=True,
                         how='left')
    temp_cont['pdv1'] = temp_cont.index
    # if any(0,1,total)==0, then combine it with per bin or next bin
    temp_cont = _combineBins(temp_cont, target)
    # calculate other temp vars
    temp_cont['bin'] = 1
    temp_cont['i'] = range(1, len(temp_cont) + 1)
    temp_cont['var'] = temp_cont.index
    nbins = 1
    # exe candSplit mmax times
    while (nbins < mmax):
        temp_cont = _candSplit(temp_cont, method=method)
        nbins += 1

    temp_cont = temp_cont.rename(columns={'var': 'oldbin'})
    temp_Map1 = temp_cont.drop([0, target, 'pdv1', 'i'], axis=1)
    temp_Map1 = temp_Map1.sort_values(by=['bin', 'oldbin'])
    # get new lower, upper, bin, total for sub
    data = pd.DataFrame()
    s = set()
    for i in temp_Map1['bin']:
        if i in s:
            pass
        else:
            sub_Map = temp_Map1[temp_Map1['bin'] == i]
            rowdata = _getNewBins(sub_Map, i)
            data = data.append(rowdata, ignore_index=True)
            s.add(i)

    # resort data
    data = data.sort_values(by='lower')
    data['newbin'] = range(1, mmax + 1)
    data = data.drop('bin', axis=1)
    data.index = data['newbin']
    return data


def reduceCats(x, y, method=1, mmax=5, badlabel=1):
    """
    Reduce category for x by y & method
    method is represent by number,
        1:Gini, 2:Entropy, 3:person chisq, 4:Info value
    ----------------------------------------------
    Params:
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    method: int obj, metric to split x
    mmax: number to reduce
    badlabel: target label
    ---------------------------------------------
    Return
    temp_cont: pandas dataframe, reduct category map
    """
    _check_target_binary(y)
    temp_cont, m = _groupCal(x, y, badlabel=badlabel)
    nbins = 1
    while (nbins < mmax):
        temp_cont = _candSplit(temp_cont, method=method)
        nbins += 1

    temp_cont = temp_cont.rename(columns={'var': x.name})
    temp_cont = temp_cont.drop([0, 1, 'i', 'pdv1'], axis=1)
    return temp_cont.sort_values(by='bin')


def applyMapCats(x, bin_map):
    """
    convert x to newbin by bin_map
    ------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, mapTable contain new bins
    ------------------------------
    Return
    new_x: pandas Series, convert results
    """
    d = dict()
    for i in bin_map.index:
        subData = bin_map[bin_map.index == i]
        value = subData.ix[i, 'bin']
        d[i] = value

    new_x = x.map(d)
    new_x.name = x.name + '_BIN'
    return new_x


def tableTranslate(red_map):
    """
    table tranlate for red_map
    ---------------------------
    Params
    red_map: pandas dataframe,reduceCats results
    ---------------------------
    Return
    res: pandas series
    """
    l = red_map['bin'].unique()
    res = pd.Series(index=l)
    for i in l:
        value = red_map[red_map['bin'] == i].index
        value = list(value.map(lambda x: str(x) + ';'))
        value = "".join(value)
        res[i] = value
    return res
