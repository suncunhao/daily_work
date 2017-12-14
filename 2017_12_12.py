#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 9:36
# @Author  : sch
# @File    : 2017_12_12.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

CS = pd.read_csv('data_output/ZTmodule/CS_index_nan20171210V0.4.csv', encoding='gbk')
OS = pd.read_csv('data_output/ZTmodule/OS_index_nan20171210V0.4.csv', encoding='gbk')
EG = pd.read_csv('data_output/ZTmodule/EG_index_nan20171210V0.4.csv', encoding='gbk')

CS_useful = CS[['CS1', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'CS9',
           'CS10', 'CS11', 'CS1_n', 'CS3_n', 'CS4_n', 'CS9_n', 'CS10_n', 'credit_ratio']].dropna()
CS_X = CS_useful[['CS1', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'CS9',
           'CS10', 'CS11', 'CS1_n', 'CS3_n', 'CS4_n', 'CS9_n', 'CS10_n']]
CS_y = CS_useful['credit_ratio']
clf1 = linear_model.LinearRegression()
clf1.fit(CS_X, CS_y)
pd.concat([pd.DataFrame(clf1.coef_) ,pd.DataFrame(CS_X.columns)], axis=1)
# 求p-value
X2 = sm.add_constant(CS_X)
est = sm.OLS(CS_y, X2)
est2 = est.fit()
print(est2.summary())

#############################################

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
# 余额数据需要仅保留同一日最后一笔交易数据
result_balance = {}
for i in result.keys():
    result_balance[i] = result[i].drop_duplicates(subset=['year', 'month', 'day'], keep='last')
# 数据集成
data = pd.DataFrame()
for i in result.keys():
    data = pd.concat([data, result[i]], axis=0)
data_balance = pd.DataFrame()
for i in result_balance.keys():
    data_balance = pd.concat([data_balance, result_balance[i]], axis=0)
# 保存data
# data.to_csv('data_output/dateorder/20171212/receivable_all.csv')
# data_balance.to_csv('data_output/dateorder/20171212/balance_all.csv')
# groupby
data_group = data.groupby(by=['year', 'season', 'client_name'])
data_balance_group = data_balance.groupby(by=['year', 'season', 'client_name'])
# 计算指标平均
numerical = data_group.agg({
    'receivable_positive': np.sum,
    'payment': np.sum
}).reset_index()
numerical_balance = data_balance_group.agg({
    'balance_positive': np.sum
}).reset_index()
# 求balance_positive平均值
numerical_balance['balance_positive_avg'] = numerical_balance['balance_positive'] / 90
# 合并两张表
final = pd.merge(numerical, numerical_balance, on=['year', 'season', 'client_name'], how='outer')
# 计算新指标
final['回款率'] = final['payment']/final['receivable_positive']
final['周转率'] = final['receivable_positive']/final['balance_positive_avg']
final['周转回款天数'] = 90/(final['payment']/final['receivable_positive'])
final.to_csv('data_output/dateorder/20171212/CSNEWindex1212V0.5.csv')

# 清理inf
where_is_inf = np.isinf(final['回款率'])
final.loc[~where_is_inf, '回款率'].max()
final.loc[where_is_inf, '回款率'] = final.loc[~where_is_inf, '回款率'].max()

where_is_inf = np.isinf(final['周转率'])
final.loc[~where_is_inf, '周转率'].max()
final.loc[where_is_inf, '周转率'] = final.loc[~where_is_inf, '周转率'].max()

where_is_inf = np.isinf(final['周转回款天数'])
final.loc[~where_is_inf, '周转回款天数'].max()
final.loc[where_is_inf, '周转回款天数'] = final.loc[~where_is_inf, '周转回款天数'].max()
final.to_csv('data_output/dateorder/20171212/CSNEWindex1212V0.6.csv')
