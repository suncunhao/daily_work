#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/11 10:02
# @Author  : sch
# @File    : 2017_12_11.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# CS_NEWindex
# 日期插值
new_date = pd.date_range('2014-1-1', '2017-01-01')
date_df = pd.concat([pd.DataFrame(new_date), pd.DataFrame(new_date.year), pd.DataFrame(new_date.month), pd.DataFrame(new_date.day)], axis=1)
date_df.columns = ['date', 'year', 'month', 'day']
balance = pd.read_csv('data_output/data_for_model/client_balance.csv', encoding='gbk')

result = {}
# 每个经销商逐日交易记录
for i in np.unique(balance['client_name'].values):
    result[i] = pd.merge(date_df, balance[balance['client_name'] == i], on=['year', 'month', 'day'], how='left')
# 余额补全
for i in result.keys():
    result[i]['balance'] = result[i]['balance'].ffill()
# 仅保留同一日最后一笔交易数据
for i in result.keys():
    result[i] = result[i].drop_duplicates(subset=['year', 'month', 'day'], keep='last')
# 求出季度
for i in result.keys():
    result[i]['season'] = result[i]['month'].apply(lambda x: divmod(x-1, 3)[0] + 1)
# 应收与应付用0补全
for i in result.keys():
    result[i]['receivable'] = result[i]['receivable'].fillna(0)
    result[i]['payment'] = result[i]['payment'].fillna(0)
# 人名补全
for i in result.keys():
    result[i]['client_name'] = i
# 有两个指标的计算要取非负值
def fill_positive(x):
    if x < 0:
        return 0
    else:
        return x
for i in result.keys():
    result[i]['receivable_positive'] = result[i]['receivable'].apply(fill_positive)
    result[i]['balance_positive'] = result[i]['balance'].apply(fill_positive)
# 数据集成
data = pd.DataFrame()
for i in result.keys():
    data = pd.concat([data, result[i]], axis=0)
# 保存data
data.to_clipboard()
# groupby
data_group = data.groupby(by=['year', 'season', 'client_name'])
# 计算指标平均
numerical = data_group.agg({
    'receivable_positive': np.sum,
    'payment': np.sum,
    'balance_positive': np.sum
}).reset_index()
# 求balance_positive平均值

# 计算新指标
numerical['回款率'] = numerical['payment']['pay_sum']/numerical['receivable']['rec_sum']
numerical['周转率'] = numerical['receivable']['rec_sum']/numerical['balance']['bal_avg']
numerical['周转回款天数'] = 90/numerical['回款率']
numerical.to_csv('data_output/20171211/CSNEWindex1211V0.4.csv')




# 线性回归求p-value
import statsmodels.api as sm
X = np.random.random(40).reshape(10, 4)
y = np.random.random(10)
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

################
from sklearn.linear_model import LinearRegression
from scipy import stats
lm = LinearRegression()
lm.fit(X,y)
params = np.append(lm.intercept_,lm.coef_)
predictions = lm.predict(X)

newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
# newX = np.append(np.ones((len(X),1)), X, axis=1)
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
print(myDF3)

