#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/6 9:31
# @Author  : sch
# @File    : 2012_12_06.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

data = pd.read_csv('data_output/20171205/result_final.csv', encoding='gbk')
client_list = pd.read_csv('data_output/data_for_model/client_map.csv', encoding='gbk')
client_list = client_list.drop(['start_date', 'violation'], 1)
temp = pd.merge(client_list, data, on=['year', 'company_name'], how='left')
temp = temp.drop('Unnamed: 0', 1)
temp.rename(columns={'CompanyType': 'EG1'}, inplace=True)
temp.rename(columns={'CompanyStatus_': 'EG2', 'RegcapTrans': 'EG3', 'OpenDuration_': 'EG4'}, inplace=True)
temp.rename(columns={'ZhixingInThree_': 'EG12', 'DishonestyInAll_': 'EG17', 'DishonestyInFive_': 'EG16', 'DishonestyInThree_': 'EG15', 'PatentIndex_': 'EG19', 'PatentNum_': 'EG18', 'ZhixingInAll_': 'EG14', 'ZhixingInFive_': 'EG13'}, inplace=True)
temp.to_csv('data_output/20171206/EG_result.csv')


temp = temp[temp['EG1'].isnull() == False]
temp = temp.drop(['company_name', 'bbd_qyxx_id', 'EG1'], 1)
temp['EG2'] = temp['EG2'].apply(lambda x: 0 if x == '正常' else 1)
group = temp.groupby(['year', 'client_name'])
df = group.agg({
    'EG2': np.max,
    'EG3': np.sum,
    'EG4': np.max,
    'EG12': np.sum,
    'EG13': np.sum,
    'EG14': np.sum,
    'EG15': np.sum,
    'EG16': np.sum,
    'EG17': np.sum,
    'EG18': np.sum,
    'EG19': np.mean
}).reset_index()

df = df[(df['year'] == 2015) | (df['year'] == 2016)]

y_data = pd.read_csv('data_output/ZTmodule/CSmoduleNEW.csv', encoding='gbk')
y_data = y_data[['credit_rating', 'credit_ratio', 'client_name', 'year']]

new_EG = pd.merge(df, y_data, on=['year', 'client_name'], how='right')
new_EG.to_csv('data_output/20171206/EGmodule.csv')






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

# 格式重列
import re
# data = data.drop('Unnamed: 0', 1)
columns = data.columns
base_columns = ['company_name', 'bbd_qyxx_id', 'RegcapTrans', 'CompanyType']

result = []
for year in ['2014', '2015', '2016']:
    select_data = data[base_columns + [col for col in columns if year in col]]
    select_data.columns = [re.sub(r'\d+', '', col) for col in select_data.columns]
    select_data.insert(0, 'year', year)
    result.append(select_data)

result_df = pd.concat(result, ignore_index=True)
result_df.to_csv('data_output/20171206/result_final.csv')

# 列名梳洗
data = pd.read_csv('data_output/20171206/result_final.csv', encoding='gbk')
client_list = pd.read_csv('data_output/data_for_model/client_map.csv', encoding='gbk').dropna()
client_list = client_list.drop(['start_date', 'violation'], 1)
temp = pd.merge(client_list, data, on=['year', 'company_name'], how='left')
temp = temp.drop('Unnamed: 0', 1)
temp.rename(columns={'CompanyType': 'EG1'}, inplace=True)
temp.rename(columns={'CompanyStatus_': 'EG2', 'RegcapTrans': 'EG3', 'OpenDuration_': 'EG4'}, inplace=True)
temp.rename(columns={'ZhixingInThree_': 'EG12', 'DishonestyInAll_': 'EG17', 'DishonestyInFive_': 'EG16', 'DishonestyInThree_': 'EG15', 'PatentIndex_': 'EG19', 'PatentNum_': 'EG18', 'ZhixingInAll_': 'EG14', 'ZhixingInFive_': 'EG13'}, inplace=True)
temp.rename(columns={'KtggInFive_': 'EG6', 'KtggInThree_': 'EG5'}, inplace=True)
temp.to_csv('data_output/20171206/EG_result.csv')

# 列为模块
temp = temp[temp['EG1'].isnull() == False]
temp = temp.drop(['company_name', 'bbd_qyxx_id', 'EG1'], 1)
temp['EG2'] = temp['EG2'].apply(lambda x: 0 if x == '正常' else 1)
group = temp.groupby(['year', 'client_name'])
df = group.agg({
    'EG2': np.max,
    'EG3': np.sum,
    'EG4': np.max,
    'EG5': np.sum,
    'EG6': np.sum,
    'EG12': np.sum,
    'EG13': np.sum,
    'EG14': np.sum,
    'EG15': np.sum,
    'EG16': np.sum,
    'EG17': np.sum,
    'EG18': np.sum,
    'EG19': np.mean
}).reset_index()

df = df[(df['year'] == 2015) | (df['year'] == 2016)]

y_data = pd.read_csv('data_output/ZTmodule/CSmoduleNEW.csv', encoding='gbk')
y_data = y_data[['credit_rating', 'credit_ratio', 'client_name', 'year']]

new_EG = pd.merge(df, y_data, on=['year', 'client_name'], how='right')
new_EG.to_csv('data_output/20171206/EGmodule1206V0.2.csv')
