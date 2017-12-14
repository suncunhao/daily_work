#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/24 9:20
# @Author  : sch
# @File    : 2017_11_24.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

path = 'data_output/data_for_model'
result = []
for files in os.listdir(path):
    result.append(pd.read_csv(path+r'/'+files, encoding='gbk'))


client_balance = pd.DataFrame(result[0])
client_balance = client_balance.drop('Unnamed: 0', 1)
balance_analysis = client_balance[client_balance['receivable'] > 0].groupby(['year', 'month', 'client_name'])['receivable'].agg({
    'receivable_total': lambda x: np.sum(x),
    'receivable_frequency': lambda x: np.count_nonzero(x)
}).reset_index()

time = pd.date_range('2014-1-1', '2016-12-31', freq='M')
timedf = pd.DataFrame([time.year, time.month]).T
timedf.columns = ['year', 'month']

client_map = pd.DataFrame(result[-2])
jingxiaoshang = client_map['client_name']
jingxiaoshang = pd.DataFrame(jingxiaoshang.dropna(0).unique())
jingxiaoshang.columns = ['client_name']

analysis_temp = pd.merge(jingxiaoshang, balance_analysis, on=['client_name'], how='left')
# analysis = pd.merge(timedf, analysis_temp, on=['month', 'year'], how='left')
result2 = []
for i in jingxiaoshang['client_name']:
    # print(i)
    analysis = pd.merge(timedf, analysis_temp[analysis_temp['client_name'] == i], on=['year', 'month'], how='left')
    analysis['client_name'] = analysis['client_name'].fillna(i)
    analysis['receivable_total'] = analysis['receivable_total'].fillna(0)
    analysis['receivable_frequency'] = analysis['receivable_frequency'].fillna(0)
    result2.append(analysis)
result2_df = pd.concat(result2)
result2_df.to_csv('data_output/20171124/balance_analysis.csv')






data = pd.read_clipboard()
target = pd.read_clipboard()
y = target.applymap(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 3 if x == 'D' else 4)

import mord
from sklearn import metrics
data = data.fillna(0)
data = data.drop(['client_name'], 1)
data = data.drop(['year'], 1)
X = data
clf2 = mord.LogisticAT(alpha=1.)
clf2.fit(X, y.values.flatten())
a = clf2.predict(X)
b = a - y.values.flatten()
b

clf3 = mord.LogisticIT(alpha=1.)
clf3.fit(X, y.values.flatten())
c = clf3.predict(X)
d = c - y.values.flatten()
d

from sklearn import linear_model
clf1 = linear_model.LogisticRegression()
clf1.fit(X, y)
e = clf1.predict(X)
f = e - y.values.flatten()
f
np.count_nonzero(f)


print('Mean Absolute Error of LogisticRegression: %s' %
      metrics.mean_absolute_error(clf1.predict(X), y))
print('Mean Absolute Error of LogisticAT %s' %
      metrics.mean_absolute_error(clf2.predict(X), y))
print('Mean Absolute Error of LogisticIT %s' %
      metrics.mean_absolute_error(clf3.predict(X), y))

clf4 = mord.LogisticSE(alpha=1.)
clf4.fit(X, y.values.flatten())
print('Mean Absolute Error of LogisticSE %s' %
      metrics.mean_absolute_error(clf4.predict(X), y))
