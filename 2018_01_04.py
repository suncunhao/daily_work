#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/4 10:56
# @Author  : sch
# @File    : 2018_01_04.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# MY2017 = pd.read_clipboard()
# JZ2017 = pd.read_clipboard()
# path = 'data_output/dateorder/20180104/'
# MY2017.to_csv(path + 'MY2017.csv')
# JZ2017.to_csv(path + 'JZ2017.csv')


# client_map
# 读取2017年经销商信息
MY2017 = pd.read_csv('data_output/dateorder/20180104/MY2017.csv', encoding='gbk')
JZ2017 = pd.read_csv('data_output/dateorder/20180104/JZ2017.csv', encoding='gbk')
Information2017 = pd.concat([MY2017, JZ2017], axis=0)
Information2017 = Information2017[['客户名称', '客户简称']]

# 白名单与授信表合并
jingxiaoshang = pd.read_csv('data_output/zhengtai_csv/jingxiaoshang.csv', encoding='gbk')
jingxiaoshang = jingxiaoshang[['经销商名称']]
jingxiaoshang.columns = ['client_name']

# 经销商名称提取
Information2017['client_name'] = Information2017['客户名称'].str.split('-')
Information2017['client_name'] = Information2017['client_name'].apply(lambda x: x[0] if type(x) == list else x)
Information2017 = Information2017.drop('客户名称', 1)
#
# Information2017['company_name'] = Information2017['客户简称'].str.split('（')
# Information2017['company_name'] = Information2017['company_name'].apply(lambda x: x[0] if type(x) == list else x)
# Information2017['company_name'] = Information2017['company_name'].str.split('(')
# Information2017['company_name'] = Information2017['company_name'].apply(lambda x: x[0] if type(x) == list else x)
# Information2017 = Information2017.drop('客户简称', 1)

# 汇集两表
data_all = pd.merge(jingxiaoshang, Information2017, on='client_name', how='left')
data_all['year'] = 2017

# 去重
data_all_drop = data_all.drop_duplicates()
data_all_drop = data_all_drop.reset_index().drop('index', 1)
data_all_drop.to_csv('data_output/dateorder/20180104/client_map_2017_test.csv')

# map与balance表比对公司数目
map_num = data_all_drop.groupby('client_name').agg({
    'company_name': np.count_nonzero
})
balance = pd.read_csv('data_output/dateorder/20180103/client_balance2017.csv', encoding='gbk')
balance = balance[['client_name', 'company_name']]

balance['company_name'] = balance['company_name'].str.split('（')
balance['company_name'] = balance['company_name'].apply(lambda x: x[0] if type(x) == list else x)
balance['company_name'] = balance['company_name'].str.split('(')
balance['company_name'] = balance['company_name'].apply(lambda x: x[0] if type(x) == list else x)
balance = balance.drop_duplicates()

bal_num = balance.groupby('client_name').agg({
    'company_name': np.count_nonzero
})

data = pd.concat([bal_num['company_name'], map_num['company_name']], axis=1)
data.columns = ['bal_company_num', 'map_company_num']       # data即为比对结果



