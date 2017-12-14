#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/3 10:18
# @Author  : sch
# @File    : 2017_11_3.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

data_sheet1 = pd.read_clipboard()
data_sheet1
data_sheet1 = data_sheet1.rename(columns={'经销商名称': 'dealer_name','经销商编码': 'dealer_code', '经销商联系电话': 'dealer_number'})
data_sheet1 = data_sheet1.drop('经销商身份证号',1)
data_sheet1.to_csv('data_sheet1.csv')

data_sheet2 = pd.read_clipboard()
data_sheet2 = data_sheet2.drop(['结算单位法定代表人证件号码', '结算单位联系电话'], 1)
data_sheet2 = data_sheet2.rename(columns={'经销商名称':'dealer_name','经销商编码': 'dealer_code',
                                          '结算单位名称': 'company_name', '结算单位编码':'company_code',
                                          '结算单位统一社会信用代码': 'credit_code',
                                          '结算单位法定代表人姓名': 'frname',
                                          '结算单位注册地址':'company_address',
                                          '是否为经销商默认结算单位': 'default_company'})
data_sheet2.to_csv('data_sheet2.csv')

data_sheet3 = pd.read_clipboard()
data_sheet3 = data_sheet3.rename(columns={'经销商名称': 'dealer_name', '经销商编码': 'dealer_code', '省': 'authorized_province', '市': 'authorized_city'})
data_sheet3.to_csv('data_sheet3.csv')

data_whitelist = pd.read_clipboard()

data_temp = pd.merge(data_sheet1, data_sheet2, on='dealer_name', how='outer')
data_merge_all = pd.merge(data_temp, data_sheet3, on='dealer_name', how='outer')
data_merge_all.to_csv('sheet_merge.csv')
data_final = pd.merge(data_merge_all, data_whitelist, on='dealer_name', how='inner')
data_final.to_csv('data_final.csv')