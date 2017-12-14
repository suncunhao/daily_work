#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/20 15:46
# @Author  : sch
# @File    : 2017_11_20.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# client_map
# 读取2014年经销商信息
data2014 = pd.read_clipboard()
data2014 = data2014.drop(['信用额度', '是否控制信用额度', '是否允限销控制', '国内', '地区编码', '地区名称','发货地址', '地址', '邮政编码', '纳税人登记号', '开户银行', '银行账号'], 1)

# 白名单与授信表合并
whitelist = pd.read_clipboard()

# 经销商名称提取
data2014['经销商名称'] = data2014['客户简称'].str.split('-')
data2014['经销商名称'] = data2014['经销商名称'].apply(lambda x: x[0] if type(x) == list else x)
data2014 = data2014.drop('客户简称', 1)
# 汇集两表
data_all = pd.merge(whitelist, data2014, on='经销商名称', how='left')

# 保存
data_all.to_csv('data_output/20171122/2016company.csv')


# client_feature
# data = pd.read_clipboard(parse_dates=['start_date'])
data = pd.read_clipboard()
data.head()

# data.groupby('client_name').agg({
#     'start_date': {
#         'min': np.min,
#         'max': np.max,
#         'range': lambda x: (np.max(x) - np.min(x)).days
#     }
# })

data_new = data.groupby('client_name').agg({
        'start_date': np.min
    })

data_new = pd.read_clipboard()
information = pd.read_clipboard()
data_alll = pd.merge(data_new, information, on='client_name', by='inner')
data_alll = data_alll.drop('经销商编码', 1)
data_alll['main_business_zhengtai'] = data_alll['是否主营正泰'].apply(lambda x: 1 if x=='是' else 0)
data_alll = data_alll.drop('是否主营正泰', 1)
data_alll.to_csv('data_output/20171120/client_feature.csv')


# client_credit
data = pd.read_clipboard()



# client_balance
data1 = pd.read_clipboard()
data1['经销商'] = data1['客户简称'].str.split('-')
data1['经销商'] = data1['经销商'].apply(lambda x: x[0] if type(x) == list else x)

data2 = pd.read_clipboard()
data2.drop()
data2['经销商'] = data2['客户简称'].str.split('-')
data2['经销商'] = data2['经销商'].apply(lambda x: x[0] if type(x) == list else x)
