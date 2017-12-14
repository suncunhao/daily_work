#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/28 9:46
# @Author  : sch
# @File    : 2017_11_28.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

client_credit = pd.read_csv('data_output/data_for_model/client_credit.csv', encoding='gbk')
client_credit_select = client_credit[client_credit['year'] == 2016].append(client_credit[client_credit['year'] == 2017])
client_credit_selected = client_credit_select[['year', 'client_name', 'credit_ratio']]
client_credit_selected['year'] =client_credit_selected['year'] - 1

CSmodule = pd.read_clipboard()
HImodule = pd.read_csv('data_output/ZTmodule/HImudule_index20171124V0.2.csv')
HImodule = HImodule.drop('HI3', 1)
OSmodule = pd.read_csv('data_output/ZTmodule/OSmudule_index20171127V0.3.csv')
OSmodule = OSmodule.drop('OS5', 1)
new_cs_data = pd.merge(CSmodule, client_credit_selected, on=['year', 'client_name'], how='inner')
new_hi_data = pd.merge(HImodule, client_credit_selected, on=['year', 'client_name'], how='inner')
new_os_data = pd.merge(OSmodule, client_credit_selected, on=['year', 'client_name'], how='inner')
new_cs_data_all = new_cs_data.dropna()
new_hi_data_all = new_hi_data.dropna()
new_os_data_all = new_os_data.dropna()

# pairplot()
# var主要适用于分类变量，hue为想进行分类的指标
sns.pairplot(new_cs_data_all, vars=['CS1', 'CS2', 'CS3', 'CS4', 'credit_ratio'])
plt.show()
sns.pairplot(new_cs_data_all, vars=['CS5', 'CS6', 'CS7', 'CS8', 'CS9', 'credit_ratio'])
plt.show()
sns.pairplot(new_hi_data_all, vars=['HI1', 'HI2', 'HI4', 'HI5', 'HI6', 'credit_ratio'])
plt.show()
sns.pairplot(new_os_data_all, vars=['OS1', 'OS2', 'OS3', 'OS4', 'credit_ratio'])
plt.show()
sns.pairplot(new_os_data_all, vars=['OS6', 'OS7', 'OS8', 'credit_ratio'])
plt.show()
for i in ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'CS8', 'CS9']:
    sns.jointplot(i, 'credit_ratio', new_cs_data_all)
for i in ['HI1', 'HI2', 'HI4', 'HI5', 'HI6']:
    sns.jointplot(i, 'credit_ratio', new_hi_data_all)
for i in ['OS1', 'OS2', 'OS3', 'OS4', 'OS6', 'OS7', 'OS8']:
    sns.jointplot(i, 'credit_ratio', new_os_data_all)
sns.jointplot('CS1', 'credit_ratio', new_cs_data_all)
plt.show()


#
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation, metrics
X1 = new_cs_data_all[['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'CS8', 'CS9']]
y1 = new_cs_data_all['credit_ratio']
y1 = y1.reset_index().drop('index', 1)
y1_train = y1.loc[:30]
y1_test = y1.loc[31:]
clf1 = RandomForestRegressor(n_estimators=10, random_state=10)
clf1.fit(X1, y1)
clf1.predict(X1)
X1_train = X1.loc[:30]
X1_test = X1.loc[30:]
clf1.fit(X1_train, y1_train)
clf1.predict(X1_train)
clf1.predict(X1_test)
metrics.mean_squared_error(y1, clf1.predict(X1))

X2 = new_hi_data_all[['HI1', 'HI2', 'HI4', 'HI5', 'HI6']]
y2 = new_hi_data_all['credit_ratio']
clf2 = RandomForestRegressor(n_estimators=10,random_state=10)
clf2.fit(X2, y2)
clf2.predict(X2)
metrics.mean_squared_error(y2, clf2.predict(X2))

X3 = new_os_data_all[['OS1', 'OS2', 'OS3', 'OS4', 'OS6', 'OS7', 'OS8']]
y3 = new_os_data_all['credit_ratio'] * 100
clf3 = RandomForestRegressor(n_estimators=100)
clf3.fit(X3, y3)
clf3.predict(X3)
metrics.mean_squared_error(y3, clf3.predict(X3))


X1 = X1.reset_index().drop('index', 1)
y1 = y1.reset_index().drop('index', 1)
X1_train = X1.loc[:30]
X1_test = X1.loc[31:]
y1_train = y1.loc[:30]
y1_test = y1.loc[30:]
clf4 = RandomForestRegressor(n_estimators=20)
clf4.fit(X1_train, y1_train)
clf4.predict(X1_test)
metrics.confusion_matrix(y1_test, clf4.predict(X1_test))

from sklearn.ensemble import RandomForestClassifier
y11 = new_cs_data_all['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
y11 = y11.reset_index().drop('index', 1)
y11_train = y11.loc[:30]
y11_test = y11.loc[30:]
clf5 = RandomForestClassifier(n_estimators=10, random_state=10)
clf5.fit(X1_train, y11_train)
clf5.predict(X1_train)
clf5.predict(X1_test)
metrics.confusion_matrix(y11_train, clf5.predict(X1_train))
metrics.confusion_matrix(y11_test, clf5.predict(X1_test))
clf5.fit(X1, y11)
clf5.predict(X1)
metrics.confusion_matrix(y11, clf5.predict(X1))


y22 = new_hi_data_all['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
X2 = new_hi_data_all[['HI1', 'HI2', 'HI4', 'HI5', 'HI6']]
y2 = new_hi_data_all['credit_ratio']
X2 = X2.reset_index().drop('index', 1)
y2 = y2.reset_index().drop('index', 1)
y22 = y22.reset_index().drop('index', 1)
X2_train = X2.loc[:150]
X2_test = X2.loc[151:]
y2_train = y2.loc[:150]
y2_test = y2.loc[151:]
y22_train = y22.loc[:150]
y22_test = y22.loc[151:]
clf2 = RandomForestRegressor(n_estimators=10, random_state=10)
clf22 = RandomForestClassifier(n_estimators=10, random_state=10)
clf2.fit(X2, y2)
clf2.predict(X2)
clf2.fit(X2_train, y2_train)
clf2.predict(X2_train)
clf2.predict(X2_test)
clf22.fit(X2, y22)
clf22.predict(X2)
clf22.fit(X2_train, y22_train)
clf22.predict(X2_train)
clf22.predict(X2_test)


y33 = new_os_data_all['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
X3 = new_os_data_all[['OS1', 'OS2', 'OS3', 'OS4', 'OS6', 'OS7', 'OS8']]
y3 = new_os_data_all['credit_ratio']
X3 = X3.reset_index().drop('index', 1)
y3 = y3.reset_index().drop('index', 1)
y33 = y33.reset_index().drop('index', 1)
X3_train = X3.loc[:80]
X3_test = X3.loc[81:]
y3_train = y3.loc[:80]
y3_test = y3.loc[81:]
y33_train = y33.loc[:80]
y33_test = y33.loc[81:]
clf3 = RandomForestRegressor(n_estimators=10, random_state=10)
clf33 = RandomForestClassifier(n_estimators=10, random_state=10)
clf3.fit(X3, y3)
clf3.predict(X3)
clf3.fit(X3_train, y3_train)
clf3.predict(X3_train)
clf3.predict(X3_test)
clf33.fit(X3, y33)
clf33.predict(X3)
clf33.fit(X3_train, y33_train)
clf33.predict(X3_train)
clf33.predict(X3_test)








