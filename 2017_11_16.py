#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/16 9:33
# @Author  : sch
# @File    : 2017_11_16.py

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

import pandas as pd
import numpy as np

data = pd.read_clipboard()

# 客户档案整合
data_2014 = pd.read_clipboard()
data_2015 = pd.read_clipboard()
data_2016 = pd.read_clipboard()
# def get_name(data):
#     data['经销商'] = data['客户简称'].str.split('-')
#     data['经销商'] = data['经销商'].apply(lambda x: x[0] if type(x) == list else x)
# get_name(data_2014)

data_temp = pd.merge(data_2014, data_2015, on='客户编码', how='outer')
data_all = pd.merge(data_temp, data_2016, on='客户编码', how='outer')


kind = '客户总公司编码'
group_list = np.unique(data[kind].dropna().values)
for group in group_list:
    new_data = data[data[kind] == group]
    new_data.to_csv('data_output/yingshou_new/2016/%s.csv' % int(group))
