#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/27 9:47
# @Author  : sch
# @File    : 2017_12_27.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from variable_analysis.univariate.analysis import UnivariateAnalysis
import csv

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

data = pd.read_csv('data_output/dateorder/20171227/list_data.csv', encoding='gbk', index_col=0)
obj = UnivariateAnalysis(data=data)
obj.basic_info()

with open('data_output/dateorder/20171227/list_data.csv') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

result = {}
for files in os.listdir('data_output/dateorder/20171226/'):
    result[files] = pd.read_csv('data_output/dateorder/20171226/' + files, encoding='gbk', index_col=0)

L_2015 = pd.merge(result['2015LIST.csv'], result['2015LISTS.csv'], on='DLID', how='outer')
L_2016 = pd.merge(result['2016LIST.csv'], result['2016LISTS.csv'], on='DLID', how='outer')
L_2017_JZ = pd.merge(result['2017LIST_JZ.csv'], result['2017LISTS_JZ.csv'], on='DLID', how='outer')
L_2017_MY = pd.merge(result['2017LIST_MY.csv'], result['2017LISTS_MY.csv'], on='DLID', how='outer')
for i, j in zip([L_2015, L_2016, L_2017_JZ, L_2017_MY], ['L_2015', 'L_2016', 'L_2017_JZ', 'L_2017_MY']):
    i.to_csv('data_output/dateorder/20171227/%s.csv' % j)

def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    for i in range(0, len(nums)):
        for j in range(i + 1, len(nums)):
            if i + j == target:
                return [i, j]

