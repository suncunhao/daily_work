#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/5 9:53
# @Author  : sch
# @File    : 2017_12_05.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('variable_analysis')
from variable_analysis.univariate import analysis
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cross_validation import train_test_split

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

path = 'data_output/ZTmodule/index'
result = {}
for files in os.listdir(path):
    try:
        result[files.split('.')[0]] = pd.read_csv(path + r'/' + files, encoding='utf8').drop('Unnamed: 0', 1)
    except:
        result[files.split('.')[0]] = pd.read_csv(path + r'/' + files, encoding='gbk').drop('Unnamed: 0', 1)


df = []
for i in result.keys():
    obj = analysis.UnivariateAnalysis(data=result[i])
    temp = pd.merge(obj.basic_info().reset_index(), obj.describe()['Number'].reset_index(), on='index', how='left')
    df.append(temp)
result_df = pd.concat(df, ignore_index=True)
result_df.to_csv('data_output/20171205/index_describe.csv')

dff = []
for i in result.keys():
    obj = analysis.UnivariateAnalysis(data=result[i])
    dff.append(obj.basic_info(pct=True, row_total=False))
result_dff = pd.concat(dff, ignore_index=True)
result_dff.to_csv('data_output/20171205/index_basic_info_percent.csv')


###################################
# json处理
import os
jsonpath = 'data_output/ZTmodule/BBD/result/'
import json
jsondata = {}
for i in os.listdir(jsonpath):
    path = jsonpath + i
    f = open(path, 'r', encoding='utf8')
    result = []
    for line in f:
        result.append(line)
    result_json_str = ','.join(result)
    result_json_str = '[' + result_json_str + ']'
    data = json.loads(result_json_str)
    jsondata[i.split('.')[0]] = data

data = pd.DataFrame()
for i in jsondata.keys():
     data = pd.concat([data, pd.DataFrame(jsondata[i])], axis=0)
data.to_csv('data_output/20171206/result.csv')

################################

OS = pd.read_csv('data_output/ZTmodule/OSmodule_index20171205NEW.csv', encoding='gbk')
CS = pd.read_csv('data_output/ZTmodule/CSmoduleNEW.csv', encoding='gbk')
OS = OS.drop('Unnamed: 0', 1)
CS = CS.drop('Unnamed: 0', 1)
# CS = CS.drop('CS8', 1)
# OS = OS.drop('OS10', 1)

sys.path.append('variable_analysis')
from variable_analysis.univariate import analysis
obj1 = analysis.UnivariateAnalysis(data=CS)
CS_describe = pd.merge(obj1.basic_info().reset_index(), obj1.describe()['Number'].reset_index(), on='index', how='left')
CS_describe.to_csv('data_output/20171205/CS_describe.csv')
obj2 = analysis.UnivariateAnalysis(data=OS)
OS_describe = pd.merge(obj2.basic_info().reset_index(), obj2.describe()['Number'].reset_index(), on='index', how='left')
OS_describe.to_csv('data_output/20171205/OS_describe.csv')

# Random Forest
def change_rating(x):
    if x == 'A':
        return 'A'
    elif x == 'B':
        return 'B'
    elif x == np.NaN:
        return np.NaN
    else:
        return 'C'

def fill_inf(x):
    if x == np.inf:
        return np.nan
    elif x == -np.inf:
        return np.nan
    else:
        return x

for data in [CS, OS]:
    for i in data.columns:
        data[i] = data[i].apply(fill_inf)

CS_rating = CS.drop('credit_ratio', 1).dropna()
CS_ratio = CS.drop('credit_rating', 1).dropna()
CS_rating['credit_rating'] = CS_rating['credit_rating'].apply(change_rating)
OS_rating = OS.drop('credit_ratio', 1).dropna()
OS_ratio = OS.drop('credit_rating', 1).dropna()
OS_rating['credit_rating'] = OS_rating['credit_rating'].apply(change_rating)

# rating随机森林
X_CS_rating_all = CS_rating[CS_rating.columns[3:]]
y_CS_rating_all = CS_rating['credit_rating'].apply(change_rating)
clf_CS_rating_all = RandomForestClassifier(n_estimators=20, random_state=0)
clf_CS_rating_all.fit(X_CS_rating_all, y_CS_rating_all)
# clf_CS_rating_all.predict(X_CS_rating_all)
metrics.confusion_matrix(y_CS_rating_all, clf_CS_rating_all.predict(X_CS_rating_all))
clf_CS_rating_all.feature_importances_

X_CS_rating_train_name, X_CS_rating_test_name, y_CS_rating_train, y_CS_rating_test = train_test_split(
    CS_rating, CS_rating['credit_rating'], test_size=0.3, random_state=0
)

X_CS_rating_train = X_CS_rating_train_name[X_CS_rating_train_name.columns[3:]]
X_CS_rating_test = X_CS_rating_test_name[X_CS_rating_train_name.columns[3:]]
y_CS_rating_train = y_CS_rating_train.apply(change_rating)
y_CS_rating_test = y_CS_rating_test.apply(change_rating)

clf_CS_rating_train = RandomForestClassifier(n_estimators=20, random_state=0)
clf_CS_rating_train.fit(X_CS_rating_train, y_CS_rating_train)
metrics.confusion_matrix(y_CS_rating_train, clf_CS_rating_train.predict(X_CS_rating_train))
metric = metrics.confusion_matrix(y_CS_rating_test, clf_CS_rating_train.predict(X_CS_rating_test))
clf_CS_rating_train.feature_importances_
# 全局错误率
1 - metrics.accuracy_score(y_CS_rating_test, clf_CS_rating_train.predict(X_CS_rating_test))
# 分指标错误率
(np.sum(metric[0, :]) - metric[0, 0])/np.sum(metric[0, :])
(np.sum(metric[1, :]) - metric[1, 1])/np.sum(metric[1, :])
(np.sum(metric[2, :]) - metric[2, 2])/np.sum(metric[2, :])


# CSratio随机森林回归
X_CS_ratio_all = CS_ratio[CS_rating.columns[3:]]
y_CS_ratio_all = CS_ratio['credit_ratio']
clf_CS_ratio_all = RandomForestRegressor(n_estimators=20, random_state=0)
clf_CS_ratio_all.fit(X_CS_ratio_all, y_CS_ratio_all)
# clf_ratio_all.predict(X_CS_ratio_all)
metrics.mean_squared_error(y_CS_ratio_all, clf_CS_ratio_all.predict(X_CS_ratio_all))
clf_CS_ratio_all.feature_importances_

X_CS_ratio_train_name, X_CS_ratio_test_name, y_CS_ratio_train, y_CS_ratio_test = train_test_split(
    CS_ratio, CS_ratio['credit_ratio'], test_size=0.3, random_state=0
)

X_CS_ratio_train = X_CS_ratio_train_name[X_CS_ratio_train_name.columns[3:]]
X_CS_ratio_test = X_CS_ratio_test_name[X_CS_ratio_train_name.columns[3:]]

clf_CS_ratio_train = RandomForestRegressor(n_estimators=20, random_state=0)
clf_CS_ratio_train.fit(X_CS_ratio_train, y_CS_ratio_train)
metrics.mean_squared_error(y_CS_ratio_train, clf_CS_ratio_train.predict(X_CS_ratio_train))
metrics.mean_squared_error(y_CS_ratio_test, clf_CS_ratio_train.predict(X_CS_ratio_test))
clf_CS_ratio_train.feature_importances_

# OSrating随机森林分类
X_OS_rating_all = OS_rating[OS_rating.columns[4:]]
y_OS_rating_all = OS_rating['credit_rating'].apply(change_rating)
clf_OS_rating_all = RandomForestClassifier(n_estimators=20, random_state=0)
clf_OS_rating_all.fit(X_OS_rating_all, y_OS_rating_all)
# clf_rating_all.predict(X_OS_rating_all)
metrics.confusion_matrix(y_OS_rating_all, clf_OS_rating_all.predict(X_OS_rating_all))
clf_OS_rating_all.feature_importances_

X_OS_rating_train_name, X_OS_rating_test_name, y_OS_rating_train, y_OS_rating_test = train_test_split(
    OS_rating, OS_rating['credit_rating'], test_size=0.3, random_state=0
)

X_OS_rating_train = X_OS_rating_train_name[X_OS_rating_train_name.columns[4:]]
X_OS_rating_test = X_OS_rating_test_name[X_OS_rating_train_name.columns[4:]]
y_OS_rating_train = y_OS_rating_train.apply(change_rating)
y_OS_rating_test = y_OS_rating_test.apply(change_rating)

clf_OS_rating_train = RandomForestClassifier(n_estimators=20, random_state=0)
clf_OS_rating_train.fit(X_OS_rating_train, y_OS_rating_train)
metrics.confusion_matrix(y_OS_rating_train, clf_OS_rating_train.predict(X_OS_rating_train))
metric = metrics.confusion_matrix(y_OS_rating_test, clf_OS_rating_train.predict(X_OS_rating_test))
clf_OS_rating_train.feature_importances_
# 全局错误率
1 - metrics.accuracy_score(y_OS_rating_test, clf_OS_rating_train.predict(X_OS_rating_test))
# 分指标错误率
(np.sum(metric[0, :]) - metric[0, 0])/np.sum(metric[0, :])
(np.sum(metric[1, :]) - metric[1, 1])/np.sum(metric[1, :])
(np.sum(metric[2, :]) - metric[2, 2])/np.sum(metric[2, :])


# OSratio随机森林回归
X_OS_ratio_all = OS_ratio[OS_rating.columns[4:]]
y_OS_ratio_all = OS_ratio['credit_ratio']
clf_OS_ratio_all = RandomForestRegressor(n_estimators=20, random_state=0)
clf_OS_ratio_all.fit(X_OS_ratio_all, y_OS_ratio_all)
# clf_ratio_all.predict(X_OS_ratio_all)
metrics.mean_squared_error(y_OS_ratio_all, clf_OS_ratio_all.predict(X_OS_ratio_all))
clf_OS_ratio_all.feature_importances_

X_OS_ratio_train_name, X_OS_ratio_test_name, y_OS_ratio_train, y_OS_ratio_test = train_test_split(
    OS_ratio, OS_ratio['credit_ratio'], test_size=0.3, random_state=0
)

X_OS_ratio_train = X_OS_ratio_train_name[X_OS_ratio_train_name.columns[4:]]
X_OS_ratio_test = X_OS_ratio_test_name[X_OS_ratio_train_name.columns[4:]]

clf_OS_ratio_train = RandomForestRegressor(n_estimators=20, random_state=0)
clf_OS_ratio_train.fit(X_OS_ratio_train, y_OS_ratio_train)
metrics.mean_squared_error(y_OS_ratio_train, clf_OS_ratio_train.predict(X_OS_ratio_train))
metrics.mean_squared_error(y_OS_ratio_test, clf_OS_ratio_train.predict(X_OS_ratio_test))
clf_OS_ratio_train.feature_importances_

# 数据输出
for i, j, k in zip([CS_rating, CS_ratio, X_CS_rating_train_name, X_CS_rating_test_name, X_CS_ratio_train_name,
                X_CS_ratio_test_name], [clf_CS_rating_all.predict(X_CS_rating_all), clf_CS_ratio_all.predict(X_CS_ratio_all),
                clf_CS_rating_train.predict(X_CS_rating_train), clf_CS_rating_train.predict(X_CS_rating_test),
                clf_CS_ratio_train.predict(X_CS_ratio_train), clf_CS_ratio_train.predict(X_CS_ratio_test)], [
                'CS_rating', 'CS_ratio', 'X_CS_rating_train_name', 'X_CS_rating_test_name', 'X_CS_ratio_train_name', 'X_CS_ratio_test_name']):
    y_hat = pd.DataFrame(j)
    y_hat.columns = ['y_hat']
    df = pd.concat([i.reset_index().drop('index', 1), y_hat], 1)
    df.to_csv('data_output/20171205/%s.csv' % k)

for i, j, k in zip([OS_rating, OS_ratio, X_OS_rating_train_name, X_OS_rating_test_name, X_OS_ratio_train_name,
                X_OS_ratio_test_name], [clf_OS_rating_all.predict(X_OS_rating_all), clf_OS_ratio_all.predict(X_OS_ratio_all),
                clf_OS_rating_train.predict(X_OS_rating_train), clf_OS_rating_train.predict(X_OS_rating_test),
                clf_OS_ratio_train.predict(X_OS_ratio_train), clf_OS_ratio_train.predict(X_OS_ratio_test)], [
                'OS_rating', 'OS_ratio', 'X_OS_rating_train_name', 'X_OS_rating_test_name', 'X_OS_ratio_train_name', 'X_OS_ratio_test_name']):
    y_hat = pd.DataFrame(j)
    y_hat.columns = ['y_hat']
    df = pd.concat([i.reset_index().drop('index', 1), y_hat], 1)
    df.to_csv('data_output/20171205/%s.csv' % k)



# 画图
# 折线图
for i, j, k in zip([CS_ratio['credit_ratio'], X_CS_ratio_train_name['credit_ratio'], X_CS_ratio_test_name['credit_ratio']],
                [clf_CS_ratio_all.predict(X_CS_ratio_all), clf_CS_ratio_train.predict(X_CS_ratio_train), clf_CS_ratio_train.predict(X_CS_ratio_test)],
                   ['CS_ratio', 'CS_ratio_train', 'CS_ratio_test']):
    # fig = plt.figure()
    plt.plot(i.reset_index().drop('index', 1), label='y')
    plt.plot(j, label='y_hat')
    plt.title(k)
    plt.savefig('data_output/20171205/%s' % k)
    plt.close()

for i, j, k in zip([OS_ratio['credit_ratio'], X_OS_ratio_train_name['credit_ratio'], X_OS_ratio_test_name['credit_ratio']],
                [clf_OS_ratio_all.predict(X_OS_ratio_all), clf_OS_ratio_train.predict(X_OS_ratio_train), clf_OS_ratio_train.predict(X_OS_ratio_test)],
                   ['OS_ratio', 'OS_ratio_train', 'OS_ratio_test']):
    # fig = plt.figure()
    plt.plot(i.reset_index().drop('index', 1), label='y')
    plt.plot(j, label='y_hat')
    plt.title(k)
    plt.savefig('data_output/20171205/%s' % k)
    plt.close()



# 混淆矩阵
for i, j, k in zip([y_CS_rating_all, y_CS_rating_train, y_CS_rating_test],
                [clf_CS_rating_all.predict(X_CS_rating_all), clf_CS_rating_train.predict(X_CS_rating_train), clf_CS_rating_train.predict(X_CS_rating_test)],
                   ['CS_rating', 'CS_rating_train', 'CS_rating_test']):
    plt.matshow(metrics.confusion_matrix(i, j), cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title(k)
    plt.savefig('data_output/20171205/%s' % k)
    plt.close()

for i, j, k in zip([y_OS_rating_all, y_OS_rating_train, y_OS_rating_test],
                [clf_OS_rating_all.predict(X_OS_rating_all), clf_OS_rating_train.predict(X_OS_rating_train), clf_OS_rating_train.predict(X_OS_rating_test)],
                   ['OS_rating', 'OS_rating_train', 'OS_rating_test']):
    plt.matshow(metrics.confusion_matrix(i, j), cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title(k)
    plt.savefig('data_output/20171205/%s' % k)
    plt.close()


########################
# EG梳理
import re
data = pd.read_csv('data_output/20171205/result.csv', encoding='gbk')
data = data.drop('Unnamed: 0', 1)
columns = data.columns
base_columns = ['company_name', 'bbd_qyxx_id', 'RegcapTrans', 'CompanyType']

result = []
for year in ['2014', '2015', '2016']:
    select_data = data[base_columns + [col for col in columns if year in col]]
    select_data.columns = [re.sub(r'\d+', '', col) for col in select_data.columns]
    select_data.insert(0, 'year', year)
    result.append(select_data)

result_df = pd.concat(result, ignore_index=True)
result_df.to_csv('data_output/20171205/result_final.csv')



data = pd.read_csv('data_output/ZTmodule/index/EG.csv', encoding='gbk')
columns = data.columns
base_columns = ['姓名', 'eg1', 'eg3']

result = []
for year in ['2014', '2015', '2016']:
    select_data = data[base_columns + [col for col in columns if year in col]]
    select_data.columns = [col.split('_')[0] for col in select_data.columns]
    select_data.insert(0, 'year', year)
    result.append(select_data)

result_df = pd.concat(result, ignore_index=True)
result_df.to_csv('data_output/20171205/EG_final.csv')

def clean_na(x):
    if x == '缺失值':
        return np.nan
    else:
        return x

for i in result_df.columns:
    result_df[i] = result_df[i].apply(clean_na)
