#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 14:28
# @Author  : sch
# @File    : 2017_12_26.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

x = np.arange(0, 50, 1)
y = 2 * x + 10 * np.power(x, 2) - np.power(x-20, 3)
x_step = np.arange(0, 50, 0.5)
y_step = 2 * x_step + 10 * np.power(x_step, 2) - np.power(x_step-20, 3)
plt.plot(x, y)
plt.step(x_step, y_step, where='post', color='k', alpha=0.3)
for i in range(len(x_step)):
    plt.vlines(x_step[i], 0, y_step[i], alpha=0.3)

#############################
os.getcwd()
os.chdir('D:\\python_object')

list_header = pd.read_clipboard(header=None)
list_header = list(list_header.loc[0])
lists_header = pd.read_clipboard(header=None)
lists_header = list(lists_header.loc[0])

list_data = pd.read_clipboard(header=None)
list_data.columns = list_header

list_data.to_csv('data_output/dateorder/20171226/2017LIST_MY.csv')

lists_data = pd.read_clipboard(header=None)
lists_data.columns = lists_header

lists_data.to_csv('data_output/dateorder/20171226/2017LISTS_MY.csv')
