#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/21 9:34
# @Author  : sch
# @File    : 2017_12_21.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

y_label = pd.read_csv('data_output/ZTmodule/EG_index_final_20171221.csv', encoding='gbk')
y_label = y_label[['client_name', 'year', 'credit_ratio', 'credit_rating']]
data = pd.read_excel('data_output/ZTmodule/rp.xlsx', encoding='gbk')

new_data = pd.merge(data, y_label, left_on=['year', 'name'], right_on=['year', 'client_name'], how='right')
new_data = new_data.drop('name', 1)
# new_data = new_data.rename(columns={'name': 'client_name'})

new_data.to_csv('data_output/ZTmodule/RP_index_1221V0.2.csv')
