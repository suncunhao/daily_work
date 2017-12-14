#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/14 9:34
# @Author  : sch
# @File    : 2017_12_14.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

data = pd.read_csv('data_output/dateorder/20171214/CS_NEWindex1214.csv', encoding='gbk', index_col=0)
data = data.rename(columns={'回款率':'payrate', '周转率':'turnrate', '周转回款天数':'returnday'})
data.head()
data.to_csv('data_output/dateorder/20171214/CS_NEWindex1214.csv')


