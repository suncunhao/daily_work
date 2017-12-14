#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/23 9:06
# @Author  : sch
# @File    : 2017_11_23.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# 授信总表处理
data = pd.read_clipboard()
columns = data.columns
base_columns = ['是否属于白名单', '经销商名称']

result = []
for year in ['14', '15', '16', '17']:
    select_data = data[base_columns + [col for col in columns if year in col]]
    select_data.columns =[re.sub(r'\d+', '', col) for col in select_data.columns]
    select_data.insert(0, 'year', '20'+year)
    result.append(select_data)

result_df = pd.concat(result, ignore_index=True)
result_df = result_df[[
    'year', '是否属于白名单', '经销商名称', '销售总指标', '财务总指标',
    '授信比例', '年初信用额', '信用额', '备注', '信用等级'
]]
# result_df.to_clipboard()
result_df.to_csv('data_output/20171123/shouxinzhibiao.csv')


# client_balance
client_balance = pd.read_clipboard()
client_balance['receivable'] = client_balance['receivable'].fillna('0')
client_balance['receivable'] = client_balance['receivable'].apply(
    lambda x: float(re.sub(',', '', x))
)


df = client_balance[client_balance.receivable > 0 & (client_balance.document_type == '发货单')][['year', 'month', 'client_name', 'receivable']]
df_final = df.groupby(['year', 'month', 'client_name']).sum().reset_index()
df_final = df.groupby(['year', 'month', 'client_name'])['receivable'].agg({
    'receivable_sum': lambda x: np.sum(x),
    'receivable_count': lambda x: np.count_nonzero(x)
}).reset_index()
df_final.to_csv('data_output/20171123/yingshou_analysis.csv')

jingxiaoshang = pd.read_clipboard()
new_data = pd.merge(jingxiaoshang, df_final, on='client_name', how='left')


# client_feature
client_feature = pd.read_clipboard()
province_gdp = pd.read_clipboard()
new_data2 = pd.merge(client_feature, province_gdp, on='province', how='left')

start_time = pd.read_clipboard()
new_data3 = pd.merge(new_data2, start_time, on='client_name', how='left')

whitelist = pd.read_clipboard()
new_client_feature = pd.merge(client_feature, whitelist, on='client_name', how='left')
new_client_feature.to_clipboard()

# client_map
client_map = pd.read_clipboard()
company_violation = pd.read_clipboard()
new_client_map = pd.merge(client_map, company_violation, on=['year', 'company_name'], how='left')
new_client_map.to_clipboard()

new_client_map[new_client_map['client_name'] == new_client_map['company_name']]


# client_credit
client_credit = pd.read_clipboard()
for i in ['销售总指标', '财务总指标', '年初信用额', '信用额']:
    client_credit[i] = client_credit[i].apply(
        lambda x: x*10000
    )
client_credit.to_clipboard()


road = 'data_output/zhengtai_csv'
for i in ['shouxin_%s' % year for year in  ['2014', '2015', '2016']]:
    path = road+i
    print(path)


# save_csv
client_balance = pd.read_clipboard()
client_balance.to_csv('data_output/data_for_model/client_balance.csv')

client_feature = pd.read_clipboard()
client_feature.to_csv('data_output/data_for_model/client_feature.csv')

province_gdp = pd.read_clipboard()
province_gdp.to_csv('data_output/data_for_model/province_gdp.csv')

path = 'data_output/data_for_model'
result = []
for file in os.listdir('data_output/data_for_model'):
    result.append(pd.read_csv(path+r'/'+file, encoding='gbk'))


