#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/18 9:47
# @Author  : sch
# @File    : 2018_01_18.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

data1 = pd.read_csv('data_output/dateorder/20171227/L_2015.csv', encoding='gbk')
cInvdata2015 = data1.cInvName
cInvdata2015.value_counts().to_csv('data_output/dateorder/20180118/cInv2015.csv')

data2 = pd.read_csv('data_output/dateorder/20171227/L_2016.csv', encoding='gbk')
cInvdata2016 = data2.cInvName
cInvdata2016.value_counts().to_csv('data_output/dateorder/20180118/cInv2016.csv')

data3 = pd.read_csv('data_output/dateorder/20171227/L_2017_JZ.csv', encoding='gbk')
cInvdata2017JZ = data3.cInvName
cInvdata2017JZ.value_counts().to_csv('data_output/dateorder/20180118/cInv2017_JZ.csv')

data4 = pd.read_csv('data_output/dateorder/20171227/L_2017_MY.csv', encoding='gbk')
cInvdata2017MY = data4.cInvName
cInvdata2017MY.value_counts().to_csv('data_output/dateorder/20180118/cInv2017_MY.csv')

