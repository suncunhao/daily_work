#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/2 11:16
# @Author  : sch
# @File    : 2017_11_2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

data_whitelist = pd.read_clipboard()
data_2015shouxin = pd.read_clipboard()
data_2016shouxin = pd.read_clipboard()

pd.merge(left=data_whitelist, right=data_2015shouxin)
data_all

new_columns = {'地区':'province',
                '客户简称':'dealer',
               '年份':'year',
               '14信用等级':'credit_rating_2014',
               '15电工指标':'electrical_goal_2015',
               '15排插工具指标':'insertion_goal_2015',
               '15年照明指标':'lighting_goal_2015',
               '15授信比例':'credit_ration_2015',
               '15信用额':'credit_amount_2015',
               '备注':'memo',
               '总指标':'goal_2015'}

new_columns = {'地区':'province',
               '客户简称':'dealer',
               '年份':'year',
               '15信用等级':'credit_rating',
               '16电工指标':'electrical_goal',
               '16年排插指标':'insertion_goal',
               '16年照明指标':'lighting_goal',
               '16授信比例':'credit_ration',
               '16年信用额':'credit_amount',
               '备注':'memo',
               '总指标':'goal'}


new_columns = {'地区':'province',
                '客户简称':'dealer',
               '年份':'year',
               '14信用等级':'credit_rating',
               '15电工指标':'electrical_goal',
               '15排插工具指标':'insertion_goal',
               '15年照明指标':'lighting_goal',
               '15授信比例':'credit_ration',
               '15信用额':'credit_amount',
               '备注':'memo',
               '总指标':'goal'}


data_whitelist = pd.read_clipboard()
data_2015shouxin = pd.read_clipboard()
data_2016shouxin = pd.read_clipboard()
data_all = pd.concat([data_2015shouxin, data_2016shouxin], keys=['2015', '2016'], join='outer')
data_all_merge = pd.merge(data_all, data_whitelist, on='dealer_name')
data_all_merge.to_csv('data_all_merge.csv')
# 参数名相同的处理

data_temp = pd.merge(data_2015shouxin, data_2016shouxin, on='dealer_name', how='outer')
data_merge_all = pd.merge(data_temp, data_whitelist, on='dealer_name', how='inner')
data_merge_all.to_csv('data_merge_all.csv')