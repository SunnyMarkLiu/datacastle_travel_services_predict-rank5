#encoding=utf8
import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import datetime

def encode_onehot(df,column_name):
    feature_df=pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1),feature_df], axis=1)
    return all

def encode_count(df,column_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df

def merge_count(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_nunique(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_median(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_mean(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_sum(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_max(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_min(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_std(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def feat_count(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_nunique(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_nunique" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_mean(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_std(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_median(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_max(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_min(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_sum(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_sum" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_var(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].var()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_var" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_quantile(df, df_feature, fe,value,n,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].quantile(n)).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_quantile" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_skew(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].skew()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_skew" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def action_feats(df, df_features,fe="userid"):
    a = pd.get_dummies(df_features, columns=['actionType']).groupby(fe).sum()
    a = a[[i for i in a.columns if 'actionType' in i]].reset_index()
    df = df.merge(a, on=fe, how='left')
    return df


# 分组排序
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

############################################################################################

#encoding=utf8
import pandas as pd
import lightgbm as lgb
from com_util import *
import re
import time
import numpy as np
import math
from sklearn.metrics import roc_auc_score

path="../input/"
userProfile_train=pd.read_csv(path+"userProfile_train.csv")
userProfile_test=pd.read_csv(path+"userProfile_test.csv")
userComment_train=pd.read_csv(path+"userComment_train.csv")
userComment_test=pd.read_csv(path+"userComment_test.csv")
orderHistory_train=pd.read_csv(path+"orderHistory_train.csv")
orderHistory_test=pd.read_csv(path+"orderHistory_test.csv")
orderFuture_train=pd.read_csv(path+"orderFuture_train.csv")
orderFuture_test=pd.read_csv(path+"orderFuture_test.csv")
action_train=pd.read_csv(path+"action_train.csv")
action_test=pd.read_csv(path+"action_test.csv")
#对日期做一些处理
def get_date(timestamp) :
    time_local = time.localtime(timestamp)
    #dt = time.strftime("%Y-%m-%d %H",time_local)
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    return dt

orderFuture_test["orderType"]=-1
data=pd.concat([orderFuture_train,orderFuture_test])
user_profile=pd.concat([userProfile_train,userProfile_test]).fillna(-1)
user_comment=pd.concat([userComment_train,userComment_test])

order_history=pd.concat([orderHistory_train,orderHistory_test])
order_history["date"]=order_history["orderTime"].apply(get_date)
order_history["date"]=pd.to_datetime(order_history["date"])
order_history["weekday"]=order_history["date"].dt.weekday
order_history["hour"]=order_history["date"].dt.hour
order_history["month"]=order_history["date"].dt.month
order_history["day"]=order_history["date"].dt.day
order_history["minute"]=order_history["date"].dt.minute
order_history["second"]=order_history["date"].dt.second
order_history['tm_hour']=order_history['hour']+order_history['minute']/60.0
order_history['tm_hour_sin'] = order_history['tm_hour'].map(lambda x: math.sin((x-12)/24*2*math.pi))
order_history['tm_hour_cos'] = order_history['tm_hour'].map(lambda x: math.cos((x-12)/24*2*math.pi))

action=pd.concat([action_train,action_test])

#增加历史订单
order_history_action=order_history[["userid","orderTime","orderType"]].copy()
order_history_action.columns=["userid","actionTime","actionType"]
order_history_action["actionType"]=order_history_action["actionType"].apply(lambda x:x+10)
action=pd.concat([action,order_history_action])

action["date"]=action["actionTime"].apply(get_date)
action["date"]=pd.to_datetime(action["date"])
action["weekday"]=action["date"].dt.weekday
action["hour"]=action["date"].dt.hour
action["month"]=action["date"].dt.month
action["day"]=action["date"].dt.day
action["minute"]=action["date"].dt.minute
action["second"]=action["date"].dt.second
action['tm_hour']=action['hour']+action['minute']/60.0
action['tm_hour_sin'] = action['tm_hour'].map(lambda x: math.sin((x-12)/24*2*math.pi))
action['tm_hour_cos'] = action['tm_hour'].map(lambda x: math.cos((x-12)/24*2*math.pi))

action=action.sort_values(["userid","actionTime"])
action["date"]=action["actionTime"].apply(get_date)
action["actionTime_gap"]=action["actionTime"]-action["actionTime"].shift(1)
action["actionType_gap"]=action["actionType"].shift(1)-action["actionType"]
action["actionTime_long"]=action["actionTime"].shift(-1)-action["actionTime"]
action["actionTime_gap_2"]=action["actionTime_gap"]-action["actionTime_gap"].shift(1)
action["actionTime_long_2"]=action["actionTime_long"]-action["actionTime_long"].shift(1)
#action["user_id"]=action["userid"].shift(1) #上移，获取下次userid
#action["action_type"]=action["actionType"].shift(1) #上移，获取下次type

order_history_time=order_history[["userid","orderTime"]].copy()
this_time=action.drop_duplicates("userid",keep="last")[["userid","actionTime"]].copy()
this_time.columns=["userid","orderTime"]
order_history_time=pd.concat([order_history_time,this_time])
order_history_time=order_history_time.drop_duplicates()
order_history_time=order_history_time.sort_values(["userid","orderTime"])
order_history_time["orderTime_gap"]=order_history_time["orderTime"]-order_history_time["orderTime"].shift(1)

user_profile=encode_count(user_profile,"gender")
user_profile=encode_count(user_profile,"province")
user_profile=encode_count(user_profile,"age")

data=data.merge(user_profile,on="userid",how="left")
#######################################################################################################
order_history_diff=order_history.sort_values(["userid","orderTime"])
order_history_diff["order_diff"]=order_history_diff["orderid"]-order_history_diff["orderid"].shift(1)
order_history_diff["order_diff"]=order_history_diff["order_diff"].apply(lambda x:1 if x<0 else 0)
data=feat_sum(data,order_history_diff,["userid"],"order_diff")

#历史表
data=feat_count(data,order_history,["userid"],"orderid","history_count")
data=feat_max(data,order_history,["userid"],"orderTime")
data=feat_min(data,order_history,["userid"],"orderTime")
data=feat_sum(data,order_history,["userid"],"orderType")

for i in ["日本","美国","澳大利亚","新加坡","泰国"]:
    order_history_select=order_history[order_history.country==i]
    print order_history_select.shape
    data=feat_count(data,order_history_select,["userid"],"orderid","%s_count"%i)

for i in ["亚洲","欧洲","大洋洲","北美洲"]:
    order_history_select=order_history[order_history.continent==i]
    print order_history_select.shape
    data=feat_count(data,order_history_select,["userid"],"orderid")

#评论表
data=feat_min(data,user_comment,["userid"],"rating")
data=feat_count(data,user_comment,["userid"],"rating")

#此用户的评论是否在历史表中
kk=order_history[["orderid"]].copy()
kk["not_in_history"]=0
user_comment_orderid=user_comment.merge(kk,on="orderid",how="left").fillna(1)
user_comment_orderid=user_comment_orderid[user_comment_orderid.not_in_history==1]
data=feat_sum(data,user_comment_orderid,["userid"],"not_in_history")
data=feat_mean(data,user_comment_orderid,["userid"],"orderid","orderid_no")
data=feat_max(data,order_history,["userid"],"orderid","orderid_max")
data["ooxx"]=data["orderid_no"]-data["orderid_max"]

order_history["uo_rt"]=(order_history["userid"])/(order_history["orderid"])
data=feat_std(data,order_history,["userid"],"uo_rt")
#######################################################################################################
#行为表
action_last=pd.DataFrame(action.groupby(["userid"]).actionTime.max()).reset_index()
action_last.columns=["userid","actionTime_last"]
action=action.merge(action_last,on="userid",how="left")
action["actionTime_last_dif"]=action["actionTime_last"]-action["actionTime"]

action_567=action[(action.actionType>=5)&(action.actionType<=7)]
for i in [600,1800,3600,36000,100000,100000000]:
    action_select=action_567[action_567.actionTime_last_dif<i].copy()
    data=action_feats(data,action_select)

#用户5,6,7操作的平均时长
for i in range(5,8):
    action_select = action_567[action_567.actionType == i].copy()
    data = feat_mean(data, action_select, ["userid"], "actionTime_long", "action_user_onlytype_mean_%s" % i)
    #data = feat_quantile(data, action_select, ["userid"], "actionTime_long",0.75, "action_user_onlytype_quantile_0.75_%s" % i)
    #data = feat_quantile(data, action_select, ["userid"], "actionTime_long",0.25, "action_user_onlytype_quantile_0.25_%s" % i)
    data = feat_median(data, action_select, ["userid"], "actionTime_long", "action_user_onlytype_median_%s" % i)
    data = feat_max(data, action_select, ["userid"], "actionTime_long", "action_user_onlytype_max_%s" % i)
    data = feat_min(data, action_select, ["userid"], "actionTime_long", "action_user_onlytype_min_%s" % i)
    data = feat_std(data, action_select, ["userid"], "actionTime_long", "action_user_onlytype_std_%s" % i)

    data = feat_mean(data, action_select, ["userid"], "actionTime_gap", "gap_action_user_onlytype_mean_%s" % i)
    #data = feat_quantile(data, action_select, ["userid"], "actionTime_gap",0.75, "gap_action_user_onlytype_quantile_0.75_%s" % i)
    #data = feat_quantile(data, action_select, ["userid"], "actionTime_gap",0.25, "gap_action_user_onlytype_quantile_0.25_%s" % i)
    data = feat_median(data, action_select, ["userid"], "actionTime_gap", "gap_action_user_onlytype_median_%s" % i)
    data = feat_max(data, action_select, ["userid"], "actionTime_gap", "gap_ction_user_onlytype_max_%s" % i)
    data = feat_min(data, action_select, ["userid"], "actionTime_gap", "gap_action_user_onlytype_min_%s" % i)
    data = feat_std(data, action_select, ["userid"], "actionTime_gap", "gap_action_user_onlytype_std_%s" % i)

    data = feat_max(data, action_select, ["userid"], "actionTime", "actionType_max_%s" % i)
    data = feat_min(data, action_select, ["userid"], "actionTime", "actionType_min_%s" % i)

for i in [10]:
    action_select = action[action.actionType == i].copy()
    data = feat_mean(data, action_select, ["userid"], "actionTime_long", "action_newtype_mean_%s" % i)
    data = feat_max(data, action_select, ["userid"], "actionTime_long", "action_newtype_max_%s" % i)
    data = feat_min(data, action_select, ["userid"], "actionTime_long", "action_newtype_min_%s" % i)
    data = feat_std(data, action_select, ["userid"], "actionTime_long", "action_newtype_std_%s" % i)

    data = feat_mean(data, action_select, ["userid"], "actionTime_gap", "gap_action_newtype_mean_%s" % i)
    data = feat_max(data, action_select, ["userid"], "actionTime_gap", "gap_action_newtype_max_%s" % i)
    data = feat_min(data, action_select, ["userid"], "actionTime_gap", "gap_action_newtype_min_%s" % i)
    data = feat_std(data, action_select, ["userid"], "actionTime_gap", "gap_action_newtype_std_%s" % i)

for a,b in [(6,5),(7,5)]:
    data["max_%s-max_%s"%(a,b)] = data["actionType_max_%s"%a] - data["actionType_max_%s"%b]
    data["min_%s-min_%s"%(a,b)] = data["actionType_min_%s"%a] - data["actionType_min_%s"%b]
    data["%s_%s_rt"%(a,b)] = data["max_%s-max_%s"%(a,b)] / data["min_%s-min_%s"%(a,b)]
    data["%s_%s_dif"%(a,b)] = data["max_%s-max_%s"%(a,b)] - data["min_%s-min_%s"%(a,b)]

#所有actionType占比
type_prob=pd.DataFrame(action.groupby(["userid","actionType"]).actionTime.count()).reset_index()
type_prob.columns=["userid","actionType","type_count"]
type_prob=feat_count(type_prob,action,["userid"],"actionType","all_count")
type_prob["type_rt"]=type_prob["type_count"]/type_prob["all_count"]
action_user_type = pd.pivot_table(type_prob, index=["userid"], columns=["actionType"],values="type_rt", fill_value=0).reset_index()
data = data.merge(action_user_type, on="userid", how="left")

#到最近的每种type的时间距离
action_last=pd.DataFrame(action.groupby(["userid","actionType"]).actionTime.max()).reset_index()
action_last.columns=["userid","actionType","type_actionTime_last"]
action_last["actionType"]=action_last["actionType"].apply(lambda x:"action_last_"+str(x))
action_last=feat_max(action_last,action,["userid"],"actionTime","user_last_time")
action_last["before_type_time_gap"]=action_last["user_last_time"]-action_last["type_actionTime_last"]
action_user_type = pd.pivot_table(action_last, index=["userid"], columns=["actionType"], values="before_type_time_gap",fill_value=100000).reset_index()
data = data.merge(action_user_type, on="userid", how="left")

data["action_last_5_6"]=data["action_last_5"]-data["action_last_6"]
data["action_last_1_5"]=data["action_last_1"]-data["action_last_5"]
data["action_last_1_7"]=data["action_last_1"]-data["action_last_7"]

#最后一次type5，6的持续时间
action_56=action[(action.actionType>=1)&(action.actionType<=6)]
action_56=action_56.sort_values("actionTime")
action_56=action_56.drop_duplicates(["userid","actionType"],keep="last")
action_56["actionType"]=action_56["actionType"].apply(lambda x:"action_long_"+str(x))
action_user_type = pd.pivot_table(action_56, index=["userid"], columns=["actionType"], values="actionTime_long",fill_value=100000).reset_index()
data = data.merge(action_user_type, on="userid", how="left")
action_56["actionType"]=action_56["actionType"].apply(lambda x:x.replace("action_long_","action_gap_"))
action_user_type = pd.pivot_table(action_56, index=["userid"], columns=["actionType"], values="actionTime_gap",fill_value=100000).reset_index()
data = data.merge(action_user_type, on="userid", how="left")
#data["action_long_5_rt"]=data["action_long_5"]/data["action_user_onlytype_mean_5"]
data["action_long_6_rt"]=data["action_long_6"]/data["action_user_onlytype_mean_6"]

#第一次，最后一次操作实间
data=feat_max(data,action,["userid"],"actionTime","last_time")
data=feat_min(data,action,["userid"],"actionTime","early_time")
data=feat_count(data,action,["userid"],"actionTime","action_count")

last_time_dt=pd.to_datetime(data["last_time"].apply(get_date))
data["month"]=last_time_dt.dt.month
data["day"]=last_time_dt.dt.day
data["weekday"]=last_time_dt.dt.weekday
data["hour"]=last_time_dt.dt.hour
data["minute"]=last_time_dt.dt.minute
data["second"]=last_time_dt.dt.second
data['tm_hour']=data['hour']+data['minute']/60.0
data['tm_hour_sin'] = data['tm_hour'].map(lambda x: math.sin((x-12)/24*2*math.pi))
data['tm_hour_cos'] = data['tm_hour'].map(lambda x: math.cos((x-12)/24*2*math.pi))

data['tm_day']=data['day']+data['hour']/24.0
data['tm_day_sin'] = data['tm_day'].map(lambda x: math.sin((x-30)/30*2*math.pi))
data['tm_day_cos'] = data['tm_day'].map(lambda x: math.cos((x-30)/30*2*math.pi))

data=feat_count(data,order_history,["userid","month"],"orderid")
data=feat_count(data,order_history,["userid","day"],"orderid")
data=feat_count(data,order_history,["userid","weekday"],"orderid")
data=feat_count(data,order_history,["userid","hour"],"orderid")
data=feat_count(data,order_history,["userid","minute"],"orderid")
data=feat_count(data,order_history,["userid","second"],"orderid")


"""
for i in [2]:
    action_tail = pd.DataFrame(action.groupby("userid").tail(i)).reset_index()
    data=feat_min(data,action_tail,["userid"],"actionTime","action_time_%s"%i)
    last_time_dt=pd.to_datetime(data["action_time_%s"%i].apply(get_date))
    data["hour_%s"%i]=last_time_dt.dt.hour
    data["minute_%s"%i]=last_time_dt.dt.minute
    data["second_%s"%i]=last_time_dt.dt.second
    del data["action_time_%s"%i]
"""

"""
for i in range(5,8):
    action_select = action_567[action_567.actionType == i].copy()
    data=feat_mean(data,action_select,["userid"],"tm_hour_sin","tm_hour_sin_mean_%s"%i)
    data=feat_mean(data,action_select,["userid"],"tm_hour_cos","tm_hour_cos_mean_%s"%i)
    data['tm_hour_std'] = list(map(lambda x, y: x * x + y * y, data['tm_hour_sin_mean_%s'%i],data['tm_hour_cos_mean_%s'%i]))
    data['tm_hour_mean'] = list(map(lambda x, y: math.atan(x / y) if y > 0 else math.atan(x / y) + math.pi,data['tm_hour_sin_mean_%s'%i], data['tm_hour_cos_mean_%s'%i]))
"""

#
#for a,b in [(6,7)]:
    #action_select=action[(action.userid==action.user_id)&(action.actionType==a)&(action.action_type==b)&(action.actionTime_long<1000)]
    #data=feat_count(data,action_select,["userid"],"actionTime_long")
#######################################################################################################
#字符串级别匹配
import os
import re
#是否购买精品
if not os.path.exists("../input/string_match.csv"):
    action_str = pd.DataFrame(action.groupby("userid").actionType.apply(lambda x: "".join([str(i) for i in list(x)]))).reset_index()
    action_str.columns = ["userid", "action_str"]
    all_str="".join(list(action_str["action_str"]))
    for i in [1,2,3,4,5,6,7,8]:
        action_str["last_%s_str"%i] = action_str["action_str"].apply(lambda x:x[-i:])
        action_str_last=action_str[["last_%s_str"%i]].drop_duplicates()
        action_str_last["last_%s_search_rt"%i]=action_str_last["last_%s_str"%i].apply(lambda x:len(re.findall(x+"8",all_str))/float(len(re.findall(x,all_str))))
        #action_str_last["last_%s_search_rt"%i]=action_str_last["last_%s_str"%i].apply(lambda x:len(re.findall(x+("8" if x[-1]=="7" else ("78" if x[-1]=="6" else ("678" if x[-1]=="5" else "5678"))),all_str))/float(len(re.findall(x,all_str))))
        action_str=action_str.merge(action_str_last,on="last_%s_str"%i,how="left")

        del action_str["last_%s_str"%i]
    del action_str["action_str"]
    print action_str
    action_str.to_csv("../input/string_match.csv",index=None)
else:
    action_str=pd.read_csv("../input/string_match.csv")
data=data.merge(action_str,on="userid",how="left")

#
###
for i in [1]:
    action_tail=pd.DataFrame(action.groupby("userid").tail(i)).reset_index()
    data = feat_mean(data, action, ["userid"], "actionTime_long", "last_type_long")


#######################################################################################################
#github
X_train=pd.read_csv("../input/data_train.csv")
del X_train["futureOrderType"]
X_test=pd.read_csv("../input/data_test.csv")
X=pd.concat([X_train,X_test])
data=data.merge(X,on="userid",how="left")

print data.head()
#################################
valid=data[data.orderType==-1].copy()
del valid["orderType"]
data=data[data.orderType!=-1].copy()
data.to_csv("plantsgo_model_2_train.csv",index=None)
valid.to_csv("plantsgo_model_2_test.csv",index=None)

def stacking(clf,train_x,train_y,test_x,clf_name,class_num=1):
    train=np.zeros((train_x.shape[0],class_num))
    test=np.zeros((test_x.shape[0],class_num))
    test_pre=np.empty((folds,test_x.shape[0],class_num))
    cv_scores=[]
    for i,(train_index,test_index) in enumerate(kf):
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        te_x=train_x[test_index]
        te_y = train_y[test_index]

        train_matrix = clf.Dataset(tr_x, label=tr_y)
        test_matrix = clf.Dataset(te_x, label=te_y)

        params = {
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'auc',
                  'min_child_weight': 1.5,
                  'num_leaves': 2**5,
                  'lambda_l2': 10,
                  'subsample': 0.7,
                  'colsample_bytree': 0.5,
                  'colsample_bylevel': 0.5,
                  'learning_rate': 0.01,
                  'seed': 2017,
                  'nthread': 12,
                  'silent': True,
                  }


        num_round = 15000
        early_stopping_rounds = 100
        if test_matrix:
            model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                              early_stopping_rounds=early_stopping_rounds
                              )
            pre= model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0],1))
            train[test_index]=pre
            test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration).reshape((test_x.shape[0],1))
            cv_scores.append(roc_auc_score(te_y, pre))

        print "%s now score is:"%clf_name,cv_scores
    test[:]=test_pre.mean(axis=0)
    print "%s_score_list:"%clf_name,cv_scores
    print "%s_score_mean:"%clf_name,np.mean(cv_scores)
    with open("score_cv.txt", "a") as f:
        f.write("%s now score is:" % clf_name + str(cv_scores) + "\n")
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1,class_num),test.reshape(-1,class_num),np.mean(cv_scores)


def lgb(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"lgb")
    return xgb_train, xgb_test,cv_scores


import lightgbm
from sklearn.cross_validation import KFold
folds = 5
seed = 2017

train = data.copy()
test = valid.copy()


y_train = train['orderType'].astype(int).values
x_train = np.array(train.drop(['orderType'], axis=1))
x_test = np.array(test)
kf = KFold(x_train.shape[0], n_folds=folds, shuffle=True, random_state=seed)
lgb_train, lgb_test,m=lgb(x_train, y_train, x_test)

data["orderType"]=lgb_train
data[["userid","orderType"]].to_csv("../stacking/train_model_2.csv",index=None)

test["orderType"]=lgb_test
test[["userid","orderType"]].to_csv("../stacking/test_model_2.csv",index=None)
test["orderType"]=1-test["orderType"]
test[["userid","orderType"]].to_csv("../sub/sub_%s.csv"%("1-"+str(m)),index=None)