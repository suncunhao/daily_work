#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/4 9:28
# @Author  : sch
# @File    : 2017_12_04.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

CS = pd.read_csv('data_output/ZTmodule/CSmudule_index20171130V0.8.csv', encoding='gbk')
OS = pd.read_csv('data_output/ZTmodule/OSmudule_index20171130V0.5.csv', encoding='gbk')
HI = pd.read_csv('data_output/ZTmodule/HImudule_index20171204V0.8.csv', encoding='utf-8')

CS = CS.drop('CS8', 1)
HI = HI.drop('HI6', 1)

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

for data in [CS, OS, HI]:
    for i in data.columns:
        data[i] = data[i].apply(fill_inf)

CS_rating = CS.drop('credit_ratio', 1).dropna()
CS_ratio = CS.drop('credit_rating', 1).dropna()
OS_rating = OS.drop('credit_ratio', 1).dropna()
OS_ratio = OS.drop('credit_rating', 1).dropna()
HI_rating = HI.drop('credit_ratio', 1).dropna()
HI_ratio = HI.drop('credit_rating', 1).dropna()

# CSrating随机森林分类
X_CS_rating_all = CS_rating[CS_rating.columns[4:]]
y_CS_rating_all = CS_rating['credit_rating'].apply(change_rating)
clf_CS_rating_all = RandomForestClassifier(n_estimators=20, random_state=0)
clf_CS_rating_all.fit(X_CS_rating_all, y_CS_rating_all)
# clf_CS_rating_all.predict(X_CS_rating_all)
metrics.confusion_matrix(y_CS_rating_all, clf_CS_rating_all.predict(X_CS_rating_all))

X_CS_rating_train_name, X_CS_rating_test_name, y_CS_rating_train, y_CS_rating_test = train_test_split(
    CS_rating, CS_rating['credit_rating'], test_size=0.3, random_state=0
)

X_CS_rating_train = X_CS_rating_train_name[X_CS_rating_train_name.columns[4:]]
X_CS_rating_test = X_CS_rating_test_name[X_CS_rating_train_name.columns[4:]]
y_CS_rating_train = y_CS_rating_train.apply(change_rating)
y_CS_rating_test = y_CS_rating_test.apply(change_rating)

clf_CS_rating_train = RandomForestClassifier(n_estimators=20, random_state=0)
clf_CS_rating_train.fit(X_CS_rating_train, y_CS_rating_train)
metrics.confusion_matrix(y_CS_rating_train, clf_CS_rating_train.predict(X_CS_rating_train))
metric = metrics.confusion_matrix(y_CS_rating_test, clf_CS_rating_train.predict(X_CS_rating_test))
# 全局错误率
1 - metrics.accuracy_score(y_CS_rating_test, clf_CS_rating_train.predict(X_CS_rating_test))
# 分指标错误率
(np.sum(metric[0, :]) - metric[0, 0])/np.sum(metric[0, :])
(np.sum(metric[1, :]) - metric[1, 1])/np.sum(metric[1, :])
(np.sum(metric[2, :]) - metric[2, 2])/np.sum(metric[2, :])


# CSratio随机森林回归
X_CS_ratio_all = CS_ratio[CS_rating.columns[4:]]
y_CS_ratio_all = CS_ratio['credit_ratio']
clf_CS_ratio_all = RandomForestRegressor(n_estimators=20, random_state=0)
clf_CS_ratio_all.fit(X_CS_ratio_all, y_CS_ratio_all)
# clf_ratio_all.predict(X_CS_ratio_all)
metrics.mean_squared_error(y_CS_ratio_all, clf_CS_ratio_all.predict(X_CS_ratio_all))

X_CS_ratio_train_name, X_CS_ratio_test_name, y_CS_ratio_train, y_CS_ratio_test = train_test_split(
    CS_ratio, CS_ratio['credit_ratio'], test_size=0.3, random_state=0
)

X_CS_ratio_train = X_CS_ratio_train_name[X_CS_ratio_train_name.columns[4:]]
X_CS_ratio_test = X_CS_ratio_test_name[X_CS_ratio_train_name.columns[4:]]

clf_CS_ratio_train = RandomForestRegressor(n_estimators=20, random_state=0)
clf_CS_ratio_train.fit(X_CS_ratio_train, y_CS_ratio_train)
metrics.mean_squared_error(y_CS_ratio_train, clf_CS_ratio_train.predict(X_CS_ratio_train))
metrics.mean_squared_error(y_CS_ratio_test, clf_CS_ratio_train.predict(X_CS_ratio_test))


# OSrating随机森林分类
X_OS_rating_all = OS_rating[OS_rating.columns[4:]]
y_OS_rating_all = OS_rating['credit_rating'].apply(change_rating)
clf_OS_rating_all = RandomForestClassifier(n_estimators=20, random_state=0)
clf_OS_rating_all.fit(X_OS_rating_all, y_OS_rating_all)
# clf_rating_all.predict(X_OS_rating_all)
metrics.confusion_matrix(y_OS_rating_all, clf_OS_rating_all.predict(X_OS_rating_all))

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

X_OS_ratio_train_name, X_OS_ratio_test_name, y_OS_ratio_train, y_OS_ratio_test = train_test_split(
    OS_ratio, OS_ratio['credit_ratio'], test_size=0.3, random_state=0
)

X_OS_ratio_train = X_OS_ratio_train_name[X_OS_ratio_train_name.columns[4:]]
X_OS_ratio_test = X_OS_ratio_test_name[X_OS_ratio_train_name.columns[4:]]

clf_OS_ratio_train = RandomForestRegressor(n_estimators=20, random_state=0)
clf_OS_ratio_train.fit(X_OS_ratio_train, y_OS_ratio_train)
metrics.mean_squared_error(y_OS_ratio_train, clf_OS_ratio_train.predict(X_OS_ratio_train))
metrics.mean_squared_error(y_OS_ratio_test, clf_OS_ratio_train.predict(X_OS_ratio_test))


# HIrating随机森林分类
X_HI_rating_all = HI_rating[HI_rating.columns[4:]]
y_HI_rating_all = HI_rating['credit_rating'].apply(change_rating)
clf_HI_rating_all = RandomForestClassifier(n_estimators=20, random_state=0)
clf_HI_rating_all.fit(X_HI_rating_all, y_HI_rating_all)
# clf_rating_all.predict(X_HI_rating_all)
metrics.confusion_matrix(y_HI_rating_all, clf_HI_rating_all.predict(X_HI_rating_all))

X_HI_rating_train_name, X_HI_rating_test_name, y_HI_rating_train, y_HI_rating_test = train_test_split(
    HI_rating, HI_rating['credit_rating'], test_size=0.3, random_state=0
)

X_HI_rating_train = X_HI_rating_train_name[X_HI_rating_train_name.columns[4:]]
X_HI_rating_test = X_HI_rating_test_name[X_HI_rating_train_name.columns[4:]]
y_HI_rating_train = y_HI_rating_train.apply(change_rating)
y_HI_rating_test = y_HI_rating_test.apply(change_rating)

clf_HI_rating_train = RandomForestClassifier(n_estimators=20, random_state=0)
clf_HI_rating_train.fit(X_HI_rating_train, y_HI_rating_train)
metrics.confusion_matrix(y_HI_rating_train, clf_HI_rating_train.predict(X_HI_rating_train))
metric = metrics.confusion_matrix(y_HI_rating_test, clf_HI_rating_train.predict(X_HI_rating_test))
# 全局错误率
1 - metrics.accuracy_score(y_HI_rating_test, clf_HI_rating_train.predict(X_HI_rating_test))
# 分指标错误率
(np.sum(metric[0, :]) - metric[0, 0])/np.sum(metric[0, :])
(np.sum(metric[1, :]) - metric[1, 1])/np.sum(metric[1, :])
(np.sum(metric[2, :]) - metric[2, 2])/np.sum(metric[2, :])

# HIratio随机森林回归
X_HI_ratio_all = HI_ratio[HI_rating.columns[4:]]
y_HI_ratio_all = HI_ratio['credit_ratio']
clf_HI_ratio_all = RandomForestRegressor(n_estimators=20, random_state=0)
clf_HI_ratio_all.fit(X_HI_ratio_all, y_HI_ratio_all)
# clf_ratio_all.predict(X_HI_ratio_all)
metrics.mean_squared_error(y_HI_ratio_all, clf_HI_ratio_all.predict(X_HI_ratio_all))

X_HI_ratio_train_name, X_HI_ratio_test_name, y_HI_ratio_train, y_HI_ratio_test = train_test_split(
    HI_ratio, HI_ratio['credit_ratio'], test_size=0.3, random_state=0
)

X_HI_ratio_train = X_HI_ratio_train_name[X_HI_ratio_train_name.columns[4:]]
X_HI_ratio_test = X_HI_ratio_test_name[X_HI_ratio_train_name.columns[4:]]

clf_HI_ratio_train = RandomForestRegressor(n_estimators=20, random_state=0)
clf_HI_ratio_train.fit(X_HI_ratio_train, y_HI_ratio_train)
metrics.mean_squared_error(y_HI_ratio_train, clf_HI_ratio_train.predict(X_HI_ratio_train))
metrics.mean_squared_error(y_HI_ratio_test, clf_HI_ratio_train.predict(X_HI_ratio_test))


# 数据输出
for i, j, k in zip([CS_rating, CS_ratio, X_CS_rating_train_name, X_CS_rating_test_name, X_CS_ratio_train_name,
                X_CS_ratio_test_name], [clf_CS_rating_all.predict(X_CS_rating_all), clf_CS_ratio_all.predict(X_CS_ratio_all),
                clf_CS_rating_train.predict(X_CS_rating_train), clf_CS_rating_train.predict(X_CS_rating_test),
                clf_CS_ratio_train.predict(X_CS_ratio_train), clf_CS_ratio_train.predict(X_CS_ratio_test)], [
                'CS_rating', 'CS_ratio', 'X_CS_rating_train_name', 'X_CS_rating_test_name', 'X_CS_ratio_train_name', 'X_CS_ratio_test_name']):
    y_hat = pd.DataFrame(j)
    y_hat.columns = ['y_hat']
    df = pd.concat([i.reset_index().drop('index', 1), y_hat], 1)
    df.to_csv('data_output/20171204/%s.csv' % k)

for i, j, k in zip([OS_rating, OS_ratio, X_OS_rating_train_name, X_OS_rating_test_name, X_OS_ratio_train_name,
                X_OS_ratio_test_name], [clf_OS_rating_all.predict(X_OS_rating_all), clf_OS_ratio_all.predict(X_OS_ratio_all),
                clf_OS_rating_train.predict(X_OS_rating_train), clf_OS_rating_train.predict(X_OS_rating_test),
                clf_OS_ratio_train.predict(X_OS_ratio_train), clf_OS_ratio_train.predict(X_OS_ratio_test)], [
                'OS_rating', 'OS_ratio', 'X_OS_rating_train_name', 'X_OS_rating_test_name', 'X_OS_ratio_train_name', 'X_OS_ratio_test_name']):
    y_hat = pd.DataFrame(j)
    y_hat.columns = ['y_hat']
    df = pd.concat([i.reset_index().drop('index', 1), y_hat], 1)
    df.to_csv('data_output/20171204/%s.csv' % k)

for i, j, k in zip([HI_rating, HI_ratio, X_HI_rating_train_name, X_HI_rating_test_name, X_HI_ratio_train_name,
                X_HI_ratio_test_name], [clf_HI_rating_all.predict(X_HI_rating_all), clf_HI_ratio_all.predict(X_HI_ratio_all),
                clf_HI_rating_train.predict(X_HI_rating_train), clf_HI_rating_train.predict(X_HI_rating_test),
                clf_HI_ratio_train.predict(X_HI_ratio_train), clf_HI_ratio_train.predict(X_HI_ratio_test)], [
                'HI_rating', 'HI_ratio', 'X_HI_rating_train_name', 'X_HI_rating_test_name', 'X_HI_ratio_train_name', 'X_HI_ratio_test_name']):
    y_hat = pd.DataFrame(j)
    y_hat.columns = ['y_hat']
    df = pd.concat([i.reset_index().drop('index', 1), y_hat], 1)
    df.to_csv('data_output/20171204/%s.csv' % k)


# 画图
# 折线图
for i, j, k in zip([CS_ratio['credit_ratio'], X_CS_ratio_train_name['credit_ratio'], X_CS_ratio_test_name['credit_ratio']],
                [clf_CS_ratio_all.predict(X_CS_ratio_all), clf_CS_ratio_train.predict(X_CS_ratio_train), clf_CS_ratio_train.predict(X_CS_ratio_test)],
                   ['CS_ratio', 'CS_ratio_train', 'CS_ratio_test']):
    # fig = plt.figure()
    plt.plot(i.reset_index().drop('index', 1), label='y')
    plt.plot(j, label='y_hat')
    plt.title(k)
    plt.savefig('data_output/20171204/%s' % k)
    plt.close()

for i, j, k in zip([OS_ratio['credit_ratio'], X_OS_ratio_train_name['credit_ratio'], X_OS_ratio_test_name['credit_ratio']],
                [clf_OS_ratio_all.predict(X_OS_ratio_all), clf_OS_ratio_train.predict(X_OS_ratio_train), clf_OS_ratio_train.predict(X_OS_ratio_test)],
                   ['OS_ratio', 'OS_ratio_train', 'OS_ratio_test']):
    # fig = plt.figure()
    plt.plot(i.reset_index().drop('index', 1), label='y')
    plt.plot(j, label='y_hat')
    plt.title(k)
    plt.savefig('data_output/20171204/%s' % k)
    plt.close()

for i, j, k in zip([HI_ratio['credit_ratio'], X_HI_ratio_train_name['credit_ratio'], X_HI_ratio_test_name['credit_ratio']],
                [clf_HI_ratio_all.predict(X_HI_ratio_all), clf_HI_ratio_train.predict(X_HI_ratio_train), clf_HI_ratio_train.predict(X_HI_ratio_test)],
                   ['HI_ratio', 'HI_ratio_train', 'HI_ratio_test']):
    # fig = plt.figure()
    plt.plot(i.reset_index().drop('index', 1), label='y')
    plt.plot(j, label='y_hat')
    plt.title(k)
    plt.savefig('data_output/20171204/%s' % k)
    plt.close()

# 混淆矩阵
for i, j, k in zip([y_CS_rating_all, y_CS_rating_train, y_CS_rating_test],
                [clf_CS_rating_all.predict(X_CS_rating_all), clf_CS_rating_train.predict(X_CS_rating_train), clf_CS_rating_train.predict(X_CS_rating_test)],
                   ['CS_rating', 'CS_rating_train', 'CS_rating_test']):
    plt.matshow(metrics.confusion_matrix(i, j), cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title(k)
    plt.savefig('data_output/20171204/%s' % k)
    plt.close()

for i, j, k in zip([y_OS_rating_all, y_OS_rating_train, y_OS_rating_test],
                [clf_OS_rating_all.predict(X_OS_rating_all), clf_OS_rating_train.predict(X_OS_rating_train), clf_OS_rating_train.predict(X_OS_rating_test)],
                   ['OS_rating', 'OS_rating_train', 'OS_rating_test']):
    plt.matshow(metrics.confusion_matrix(i, j), cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title(k)
    plt.savefig('data_output/20171204/%s' % k)
    plt.close()

for i, j, k in zip([y_HI_rating_all, y_HI_rating_train, y_HI_rating_test],
                [clf_HI_rating_all.predict(X_HI_rating_all), clf_HI_rating_train.predict(X_HI_rating_train), clf_HI_rating_train.predict(X_HI_rating_test)],
                   ['HI_rating', 'HI_rating_train', 'HI_rating_test']):
    plt.matshow(metrics.confusion_matrix(i, j), cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title(k)
    plt.savefig('data_output/20171204/%s' % k)
    plt.close()



#######################
# import sys
# sys.path.append('univariate')
# from univariate.analysis import UnivariateAnalysis
#
# OS_data = OS.drop(['year', 'client_name', 'Unnamed: 0'], 1)
# obj1 = UnivariateAnalysis(data=OS_data)
# obj1.get_variable_description()
# obj1.get_numeric_variable_description()
# OS_describe = pd.merge(obj1.get_variable_description().reset_index(), obj1.get_numeric_variable_description().reset_index(), on='index',how='left')
# OS_describe.to_csv('data_output/20171204/OS_describe.csv')
#
# CS_data = CS.drop(['year', 'client_name', 'Unnamed: 0'], 1)
# obj2 = UnivariateAnalysis(data=CS_data)
# obj2.get_variable_description()
# obj2.get_numeric_variable_description()
# CS_describe = pd.merge(obj2.get_variable_description().reset_index(), obj2.get_numeric_variable_description().reset_index(), on='index',how='left')
# CS_describe.to_csv('data_output/20171204/CS_describe.csv')
#
# HI_data = HI.drop(['year', 'client_name', 'Unnamed: 0'], 1)
# obj1 = UnivariateAnalysis(data=HI_data)
# obj1.get_variable_description()
# obj1.get_numeric_variable_description()
# HI_describe = pd.merge(obj1.get_variable_description().reset_index(), obj1.get_numeric_variable_description().reset_index(), on='index',how='left')
# HI_describe.to_csv('data_output/20171204/HI_describe.csv')