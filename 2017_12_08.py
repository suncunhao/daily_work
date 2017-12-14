#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/8 9:18
# @Author  : sch
# @File    : 2017_12_08.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# 日期插值
new_date = pd.date_range('2016-1-1', '2016-12-31')
date_df = pd.concat([pd.DataFrame(new_date), pd.DataFrame(new_date.year), pd.DataFrame(new_date.month), pd.DataFrame(new_date.day)], axis=1)
date_df.columns = ['date', 'year', 'month', 'day']
balance = pd.read_csv('data_output/data_for_model/client_balance.csv', encoding='gbk')
balance_2016 = balance[balance['year'] == 2016]

result = {}
for i in np.unique(balance['client_name'].values):
    result[i] = pd.merge(date_df, balance_2016[balance_2016['client_name'] == i], on=['month', 'day'], how='left')

# 删除同日期的前几行
# drop_duplicates(subset=['除了那一列的其它所有列'], keep='last')

for i in result.keys():
    result[i]['balance'] = result[i]['balance'].ffill()

# for k in result.keys():
#     result[k] = result[k].drop('year_y', 1)
#     result[k].rename(columns={'year_x': 'year'}, inplace=True)
#
# for m in result.keys():
#     result[m] = result[m].drop_duplicates(subset=['date', 'year', 'month', 'day', 'client_name', 'company_name','receivable', 'payment', 'balance', 'document_type', 'document_id', 'memo'], keep='last')
#

# for i in result.keys():
#     for j in range(result[i].shape[0]):
#         if j > 0:
#             result[i]['timedelta'] = np.nan
#             result[i]['timedelta'][j] = (result[i]['date'][j] - result[i]['date'][j-1]).days

for i in result.keys():
    result[i] = result[i].drop_duplicates(subset=['year_x', 'month', 'day'], keep='last')

for i in result.keys():
    result[i]['season'] = result[i]['month'].apply(lambda x: divmod(x-1, 3)[0] + 1)



# FIN指标
def fill_positive(x):
    if x < 0:
        return 0
    else:
        return x

# 季度每日余额
# data = pd.read_csv('data_output/20171208/balance_alldate.csv', encoding='gbk')
data = pd.read_clipboard()
data['season'] = data['month'].apply(lambda x: divmod(x-1, 3)[0] + 1)
data['balance'] = data['balance'].apply(fill_positive)

all_balance_group = data.groupby(by=['姓名', 'year', 'season'])
positive_balance_group = data[data['balance'] >= 0].groupby(by=['姓名', 'year', 'season'])

all_balance_sum = all_balance_group.agg({
    'balance': np.sum
}).reset_index()
all_balance_sum['all_avg'] = all_balance_sum['balance']/90

positive_balance_sum = positive_balance_group.agg({
    'balance': np.sum
}).reset_index()
positive_balance_sum['all_positive_avg'] = positive_balance_sum['balance']/90

# 周转回款率
balance = pd.read_csv('data_output/data_for_model/client_balance.csv', encoding='gbk')
balance['season'] = balance['month'].apply(lambda x: divmod(x-1, 3)[0] + 1)
balance['receivable'] = balance['receivable'].apply(fill_positive)
group_rec_posivible_sum = balance.groupby(by=['year', 'season', 'client_name'])
# group_rec_posivible_sum = balance[balance['receivable'] > 0].groupby(by=['year', 'season', 'client_name'])
group_pay_all_sum = balance.groupby(by=['year', 'season', 'client_name'])

rec_posivible_sum = group_rec_posivible_sum.agg({
    'receivable': np.sum
}).reset_index()

pay_all_sum = group_pay_all_sum.agg({
    'payment': np.sum
}).reset_index()



# FNI指标
temp = pd.merge(all_balance_sum, pay_all_sum, left_on=['姓名', 'year', 'season'], right_on=['client_name', 'year', 'season'], how='outer')
final = pd.merge(temp, rec_posivible_sum, left_on=['姓名', 'year', 'season'], right_on=['client_name', 'year', 'season'], how='outer')

final['回款率'] = final['payment']/final['receivable']
final['周转率'] = final['receivable']/final['all_avg']
final['周转回款天数'] = 90/final['周转率']
# 周转回款天数最好不要用周转率作为中间量
# 如果rec<0 补全为0
# 分母为0的时候计算累计量cum进行计算



