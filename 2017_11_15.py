#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/15 10:44
# @Author  : sch
# @File    : 2017_11_15.py

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

import pandas as pd
import numpy as np
import copy

# 读取数据
data = pd.read_clipboard()
dictionary = pd.read_clipboard()

# 保留数据副本
data_copy = data.copy()

# 时间清理
def clean_time(data, day, month):
    data['日'] = data['日'].fillna(day)
    data['月'] = data['月'].fillna(month)

# 提取客户总公司编码(数据格式不规范，抛弃)
# data['客户总公司'] = data['客户编码'].apply(lambda x: str(x)[:6])

# 整理经销商(难以避免重名情况，抛弃)
# data['经销商'] = data['客户简称'].str.split('-')
# data['经销商'] = data['经销商'].apply(lambda x: x[0] if type(x) == list else x)

# 构建字典
no_one = {}
for i in range(len(dictionary)):
    no_one[dictionary['客户编码'][i]] = dictionary['客户总公司编码'][i]

# 构建新列
data['客户总公司编码'] = data['客户编码'].map(no_one)

# 保存
path1 = 'data_output/yingshou_new/yingshou_2017/yingshou201702.csv'
data.to_csv(path1)

# 合并多个分表
data1 = pd.read_clipboard()
data2 = pd.read_clipboard()
data_temp = data1.append(data2)
data3 = pd.read_clipboard()
data_all = data_temp.append(data3)
path2 = 'data_output/yingshou_new/yingshou_2016/yingshou2016_error.csv'
data_all.to_csv(path2)


# 清洗客户总公司编码
data = pd.read_clipboard()
# data['客户总公司编码_new'] = data['客户总公司编码']
#
# for i in range(len(data)):
#     if np.isnan(data['客户总公司编码'][i]):
#         continue
#     else:
#         if len(str(int(data['客户总公司编码'][i]))) > 7:
#             data['a'][i] = str(int(data['客户总公司编码'][i]))[:-2]
#         else:
#             data['a'][i] = str(int(data['客户总公司编码'][i]))

for i in range(len(data)):
    if np.isnan(data['客户总公司编码'][i]) == False:
        data['客户总公司编码'] = data['客户总公司编码'].apply(lambda x: str(int(x))[:-2] if len(str(int(x))) > 7 else x)
    else:
        continue


# 数位规范
a = data['客户总公司编码'].apply(
    lambda x: int(x/100) if x > 1e7 else x
    )