#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/11 10:45
# @Author  : sch
# @File    : 2018_01_11.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

data = pd.read_csv('data_output/dateorder/20171227/L_2015.csv', encoding='gbk')
def clean(a):
    if '电工' in str(a):
        return '电工'
    elif '照明' in str(a):
        return '照明'
    elif '排插' in str(a):
        return '排插'

data['cInvName'].apply(clean)
