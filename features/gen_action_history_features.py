#!/home/sunnymarkliu/softwares/anaconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 17-12-25 下午9:18
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


def last_time_order_now_action_count(uid, history_grouped, action_grouped, flag):
    """ 最后一次 order 距离现在的 action 操作的次数 """
    a_df = action_grouped[uid]

    if flag == 0:
        sub_action_df = a_df
        if sub_action_df.shape[0] == 0:
            return 0, 0, 0, 0, 0, 0, 0, 0

        actionTypes = sub_action_df['actionType'].tolist()
        return len(actionTypes), actionTypes.count('browse_product'), actionTypes.count('fillin_form5'), \
               actionTypes.count('fillin_form6'), actionTypes.count('fillin_form7'), actionTypes.count('open_app'), \
               actionTypes.count('pay_money'), actionTypes.count('submit_order')

    h_df = history_grouped[uid]
    if a_df.shape[0] == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0
    else:
        last_order_time = h_df.iloc[-1]['orderTime']
        sub_action_df = a_df[a_df['actionTime'] > last_order_time]
        if sub_action_df.shape[0] == 0:
            return 0, 0, 0, 0, 0, 0, 0, 0

        actionTypes = sub_action_df['actionType'].tolist()
        return len(actionTypes), actionTypes.count('browse_product'), actionTypes.count('fillin_form5'), \
               actionTypes.count('fillin_form6'), actionTypes.count('fillin_form7'), actionTypes.count('open_app'), \
               actionTypes.count('pay_money'), actionTypes.count('submit_order')


def build_action_history_features(df, action, history):
    features = pd.DataFrame({'userid': df['userid']})

    df_ids = history['userid'].unique()
    action_grouped = dict(list(action.groupby('userid')))
    history_grouped = dict(list(history.groupby('userid')))

    #给trade表打标签，若id在login表中，则打标签为1，否则为0
    features['has_history_flag'] = features['userid'].map(lambda uid: uid in df_ids).astype(int)

    # action 表
    print('距离现在每个用户的 action 特征')
    # 最后一次 order 距离现在的 action 操作的次数
    features['last_time_order_now_action_info_count'] = features.apply(lambda row: last_time_order_now_action_count(row['userid'], history_grouped, action_grouped, row['has_history_flag']), axis=1)
    features['last_time_order_now_action_total_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[0])
    features['last_time_order_now_action_browse_product_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[1])
    features['last_time_order_now_action_fillin_form5_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[2])
    features['last_time_order_now_action_fillin_form6_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[3])
    features['last_time_order_now_action_fillin_form7_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[4])
    features['last_time_order_now_action_open_app_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[5])
    features['last_time_order_now_action_pay_money_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[6])
    features['last_time_order_now_action_submit_order_count'] = features['last_time_order_now_action_info_count'].map(lambda x: x[7])
    del features['last_time_order_now_action_info_count']
    # # 是否有支付操作和提交订单操作
    # features['last_time_order_now_has_paied_money'] = features['last_time_order_now_action_pay_money_count'].map(lambda x: int(x > 0))
    # features['last_time_order_now_has_submited_order'] = features['last_time_order_now_action_submit_order_count'].map(lambda x: int(x > 0))


    del features['has_history_flag']

    return features


def main():
    feature_name = 'action_history_features'
    # if data_utils.is_feature_created(feature_name):
    #     return

    print('load cleaned datasets')
    train = pd.read_csv(Configure.base_path + 'train/orderFuture_train.csv', encoding='utf8')
    test = pd.read_csv(Configure.base_path + 'test/orderFuture_test.csv', encoding='utf8')

    user_train = pd.read_csv(Configure.cleaned_path + 'cleaned_userProfile_train.csv', encoding='utf8')
    user_test = pd.read_csv(Configure.cleaned_path + 'cleaned_userProfile_test.csv', encoding='utf8')
    action_train = pd.read_csv(Configure.cleaned_path + 'cleaned_action_train.csv')
    action_test = pd.read_csv(Configure.cleaned_path + 'cleaned_action_test.csv')
    orderHistory_train = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_train.csv', encoding='utf8')
    orderHistory_test = pd.read_csv(Configure.cleaned_path + 'cleaned_orderHistory_test.csv', encoding='utf8')
    userComment_train = pd.read_csv(Configure.cleaned_path + 'cleaned_userComment_train.csv', encoding='utf8')
    userComment_test = pd.read_csv(Configure.cleaned_path + 'cleaned_userComment_test.csv', encoding='utf8')

    action_train['actionTime'] = pd.to_datetime(action_train['actionTime'])
    action_test['actionTime'] = pd.to_datetime(action_test['actionTime'])
    orderHistory_train['orderTime'] = pd.to_datetime(orderHistory_train['orderTime'])
    orderHistory_test['orderTime'] = pd.to_datetime(orderHistory_test['orderTime'])

    action_train.sort_values(by='actionTime', inplace=True)
    action_test.sort_values(by='actionTime', inplace=True)
    orderHistory_train.sort_values(by='orderTime', inplace=True)
    orderHistory_test.sort_values(by='orderTime', inplace=True)

    print('build train action history features')
    train_features = build_action_history_features(train, action_train, orderHistory_train)
    print('build test action history features')
    test_features = build_action_history_features(test, action_test, orderHistory_test)

    print('save ', feature_name)
    data_utils.save_features(train_features, test_features, feature_name)


if __name__ == "__main__":
    print("========== 结合 action、 history 和 comment 提取历史特征 ==========")
    main()
