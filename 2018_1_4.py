#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/3 13:07
# @Author  : sch
# @File    : 2018_1_4.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# client_map
# 读取2017年经销商信息
data2017 = pd.read_clipboard()
data2017 = data2017[['客户名称', '客户简称']]
# 白名单与授信表合并
jingxiaoshang = pd.read_clipboard()

# 经销商名称提取
data2017['经销商名称'] = data2017['客户名称'].str.split('-')
data2017['经销商名称'] = data2017['经销商名称'].apply(lambda x: x[0] if type(x) == list else x)
data2017 = data2017.drop('客户名称', 1)
# 汇集两表
data_all = pd.merge(jingxiaoshang, data2017, on='经销商名称', how='left')
data_all['year'] = 2017

# 保存
data_all.to_csv('data_output/dateorder/20180103/client_map2017.csv')


# client_balance
# 时间清理
def clean_time(data, day, month):
    data['日'] = data['日'].fillna(day)
    data['月'] = data['月'].fillna(month)

balance1 = pd.read_clipboard()
balance2 = pd.read_clipboard()
balance3 = pd.read_clipboard()

balance1['年'] = 2017
balance1 = balance1[['年', '月', '日', '客户编码', '客户名称', '客户简称', '本期应收', '本期收回', '余额', '单据类型', '销售类型', '单据号', '结算方式']]

clean_time(balance1, 1, 1)
# balance2 = balance2.dropna(0)
# balance3 = balance3.dropna(0)
balance3['客户简称'] = balance3['客户名称'].str.split('-')
balance3['客户简称'] = balance3['客户简称'].apply(lambda x: x[0] if type(x) == list else x)

temp = pd.concat([balance1, balance2], axis=0)
final = pd.concat([temp, balance3], axis=0)

final['经销商名称'] = final['客户名称'].str.split('-')
final['经销商名称'] = final['经销商名称'].apply(lambda x: x[0] if type(x) == list else x)

jingxiaoshang = pd.read_clipboard()
client_balance = pd.merge(jingxiaoshang, final, on='经销商名称', how='left')
client_balance = client_balance[['年', '月', '日', '客户编码', '经销商名称', '客户简称', '本期应收', '本期收回', '余额', '单据类型', '销售类型', '单据号', '结算方式']]
# client_balance['年'] = client_balance['年'].fillna(2017)
client_balance.columns = ['year', 'month', 'day', 'client_order', 'client_name', 'company_name','receivable', 'payment', 'balance', 'document_type', 'sales_type', 'document_id', 'payment_way']
client_balance.to_csv('data_output/dateorder/20180103/client_balance2017.csv')

balance1.to_csv('data_output/dateorder/20180103/balance1.csv')
balance2.to_csv('data_output/dateorder/20180103/balance2.csv')
balance3.to_csv('data_output/dateorder/20180103/balance3.csv')



# CS_NEWindex
# 日期插值
new_date = pd.date_range('2017-1-1', '2017-12-31')
date_df = pd.concat([pd.DataFrame(new_date), pd.DataFrame(new_date.year), pd.DataFrame(new_date.month), pd.DataFrame(new_date.day)], axis=1)
date_df.columns = ['date', 'year', 'month', 'day']
balance = pd.read_csv('data_output/dateorder/20180103/client_balance2017.csv', encoding='gbk')

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
final.to_csv('data_output/dateorder/20180103/CSNEWindex1212V0.5.csv')

