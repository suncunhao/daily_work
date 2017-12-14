#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/14 9:34
# @Author  : sch
# @File    : 2017_11_14.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# 读取数据
data = pd.read_clipboard()
trans = pd.read_clipboard()

# 创建字典
dictionary = pd.read_clipboard()
no_one = {}
for i in range(len(dictionary)):
    no_one[dictionary['客户名称'][i]] = dictionary['客户简称'][i]

# 构建新列
data['客户简称'] = data['客户名称'].map(no_one)

# 时间清理
data['日'] = data['日'].fillna(1)
data['月'] = data['月'].fillna(10)

# 保存
path = 'data_output/yingshou/yingshou_2016_new.csv'
data.to_csv(path)

# 整理经销商
data['经销商'] = data['客户简称'].str.split('-')
data['经销商'] = data['经销商'].apply(lambda x: x[0] if type(x) == list else x)

kehudangan = pd.read_clipboard()
mingcheng = np.unique(kehudangan['客户名称'])
warning = []
for ids in mingcheng:
    if kehudangan['客户名称'].value_counts()[ids] > 1:
        warning.append(ids)
chongfu = pd.DataFrame()
for i in warning:
    chongfu = chongfu.append(kehudangan[kehudangan['客户名称'] == i])
