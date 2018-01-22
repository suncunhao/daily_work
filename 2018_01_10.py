#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/10 11:42
# @Author  : sch
# @File    : 2018_01_10.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

file = 'data_output/data_for_model/client_map.csv'

dele = ['（停）',
'（停用）',
'（停用 ）',
'（停用1）',
'（停用2）',
'（聊城棚改项目）',
'（工程项目）',
'（工程）',
'（照明）',
'（排插）',
'（海尔）',
'（华润置地）',
'（济南华润置地）',
'（合生集团）',
'（销售中心）',
'（自营出口）']

client_map = pd.read_csv(file, encoding='gbk')

def change(a):
    for i in dele:
        if i in str(a):
            return a.replace(i, '')
        else:
            return str(a)

client_map['company_name'] = client_map['company_name'].apply(change)
client_map['company_name']

