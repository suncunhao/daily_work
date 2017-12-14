#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/30 9:35
# @Author  : sch
# @File    : 2017_11_30.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

zhibiao_2014 = pd.read_clipboard()
zhibiao_2014.to_csv('data_output/zhengtai_csv/zhibiao_2014.csv')
zhibiao_2015 = pd.read_clipboard()
zhibiao_2015.to_csv('data_output/zhengtai_csv/zhibiao_2015.csv')

shouxin_2015 = pd.read_clipboard()
shouxin_2015['15信用额'] = shouxin_2015['15信用额'] * 10000
shouxin_2015.to_csv('data_output/zhengtai_csv/shouxin_2015.csv')
shouxin_2016 = pd.read_clipboard()
shouxin_2016['16年信用额'] = shouxin_2016['16年信用额'] * 10000
shouxin_2016.to_csv('data_output/zhengtai_csv/shouxin_2016.csv')
shouxin_2017 = pd.read_clipboard()
shouxin_2017['17年信用额'] = shouxin_2017['17年信用额'] * 10000
shouxin_2017.to_csv('data_output/zhengtai_csv/shouxin_2017.csv')

zhibiao_2016_diangong = pd.read_clipboard()
zhibiao_2016_diangong.to_csv('data_output/zhengtai_csv/zhibiao_2016_diangong.csv')
zhibiao_2016_zhaoming = pd.read_clipboard()
zhibiao_2016_zhaoming.to_csv('data_output/zhengtai_csv/zhibiao_2016_zhaoming.csv')
zhibiao_2016_paicha = pd.read_clipboard()
zhibiao_2016_paicha.to_csv('data_output/zhengtai_csv/zhibiao_2016_paicha.csv')

zhibiao_2017_diangong = pd.read_clipboard()
zhibiao_2017_diangong.to_csv('data_output/zhengtai_csv/zhibiao_2017_diangong.csv')
zhibiao_2017_zhaoming = pd.read_clipboard()
zhibiao_2017_zhaoming.to_csv('data_output/zhengtai_csv/zhibiao_2017_zhaoming.csv')
zhibiao_2017_paicha = pd.read_clipboard()
zhibiao_2017_paicha.to_csv('data_output/zhengtai_csv/zhibiao_2017_paicha.csv')

all_client = pd.read_csv('data_output/zhengtai_csv/jingxiaoshang.csv', encoding='gbk').drop('Unnamed: 0', 1)

zhibiao_2016_temp = pd.merge(zhibiao_2016_diangong, zhibiao_2016_zhaoming, on='经销商名称', how='outer')
zhibiao_2016 = pd.merge(zhibiao_2016_temp, zhibiao_2016_paicha, on='经销商名称', how='outer')
zhibiao_2016 = zhibiao_2016.fillna(0)
zhibiao_2016['2016年销售指标'] = zhibiao_2016['2016年电工销售指标'] + zhibiao_2016['2016年照明总指标'] + zhibiao_2016['2016年排插销售指标']
zhibiao_2016 = zhibiao_2016.drop(['2016年电工销售指标', '2016年照明总指标', '2016年排插销售指标'], 1)
zhibiao_2016.to_csv('data_output/zhengtai_csv/zhibiao_2016.csv')

zhibiao_2017_temp = pd.merge(zhibiao_2017_diangong, zhibiao_2017_zhaoming, on='经销商名称', how='outer')
zhibiao_2017 = pd.merge(zhibiao_2017_temp, zhibiao_2017_paicha, on='经销商名称', how='outer')
zhibiao_2017 = zhibiao_2017.fillna(0)
zhibiao_2017['2017年销售指标'] = zhibiao_2017['2017年电工销售指标'] + zhibiao_2017['2017年照明总指标'] + zhibiao_2017['2017年排插总指标']
zhibiao_2017 = zhibiao_2017.drop(['2017年电工销售指标', '2017年照明总指标', '2017年排插总指标'], 1)
zhibiao_2017.to_csv('data_output/zhengtai_csv/zhibiao_2017.csv')

zhibiao_2014['2014年销售指标'] = zhibiao_2014['2014年销售指标'] * 10000
zhibiao_2015['2015年销售指标'] = zhibiao_2015['2015年销售指标'] * 10000
zhibiao_2016['2016年销售指标'] = zhibiao_2016['2016年销售指标'] * 10000
zhibiao_2017['2017年销售指标'] = zhibiao_2017['2017年销售指标'] * 10000

whitelist = pd.read_clipboard()
whitelist.columns = ['client_name']
whitelist['whitelist'] = 1
whitelist.to_csv('data_output/zhengtai_csv/whitelist.csv')
whitelist.columns = ['经销商名称', 'whitelist']

client_credit_2014_first = pd.merge(all_client, whitelist, on='经销商名称', how='left')
client_credit_2014_first['year'] = 2014
client_credit_2014_second = pd.merge(client_credit_2014_first, zhibiao_2014, on='经销商名称', how='left')
client_credit_2014_second['credit_ratio'] = np.NaN
client_credit_2014_second['credit_amount'] = np.NaN
client_credit_2014_second['credit_memo'] = np.NaN
client_credit_2014 = pd.merge(client_credit_2014_second, shouxin_2015[['客户简称', '14信用等级']], left_on='经销商名称', right_on='客户简称', how='left')
client_credit_2014 = client_credit_2014.drop('客户简称', 1)
client_credit_2014.columns = ['client_name', 'whitelist', 'year', 'sales_goal', 'credit_ratio', 'credit_amount', 'credit_memo', 'credit_rating']
client_credit_2014.to_csv('data_output/zhengtai_csv/client_credit_2014.csv')


client_credit_2015_first = pd.merge(all_client, whitelist, on='经销商名称', how='left')
client_credit_2015_first['year'] = 2015
client_credit_2015_second = pd.merge(client_credit_2015_first, zhibiao_2015, on='经销商名称', how='left')
client_credit_2015_third = pd.merge(client_credit_2015_second, shouxin_2015[['客户简称', '15授信比例', '15信用额', '备注']], left_on='经销商名称', right_on='客户简称', how='left')
client_credit_2015_third = client_credit_2015_third.drop('客户简称', 1)
client_credit_2015 = pd.merge(client_credit_2015_third, shouxin_2016[['经销商名称', '15信用等级']], on='经销商名称', how='left')
client_credit_2015.columns = ['client_name', 'whitelist', 'year', 'sales_goal', 'credit_ratio', 'credit_amount', 'credit_memo', 'credit_rating']
client_credit_2015.to_csv('data_output/zhengtai_csv/client_credit_2015.csv')



client_credit_2015_first = pd.merge(all_client, whitelist, on='经销商名称', how='left')
client_credit_2015_first['year'] = 2015
client_credit_2015_second = pd.merge(client_credit_2015_first, zhibiao_2015, on='经销商名称', how='left')
client_credit_2015_third = pd.merge(client_credit_2015_second, shouxin_2015[['客户简称', '15授信比例', '15信用额', '备注']], left_on='经销商名称', right_on='客户简称', how='left')
client_credit_2015 = pd.merge(client_credit_2015_third, shouxin_2016[['经销商名称', '15信用等级']], on='经销商名称', how='left')
client_credit_2015.columns = ['client_name', 'whitelist', 'year', 'sales_goal', 'credit_ratio', 'credit_amount', 'credit_memo', 'credit_rating']
client_credit_2015.to_csv('data_output/zhengtai_csv/client_credit_2015.csv')

client_credit_2016_first = pd.merge(all_client, whitelist, on='经销商名称', how='left')
client_credit_2016_first['year'] = 2016
client_credit_2016_second = pd.merge(client_credit_2016_first, zhibiao_2016, on='经销商名称', how='left')
client_credit_2016_third = pd.merge(client_credit_2016_second, shouxin_2016[['经销商名称', '16授信比例', '16年信用额', '2016年授信表备注']], on='经销商名称', how='left')
client_credit_2016 = pd.merge(client_credit_2016_third, shouxin_2017[['经销商名称', '16信用等级']], on='经销商名称', how='left')
client_credit_2016.columns = ['client_name', 'whitelist', 'year', 'sales_goal', 'credit_ratio', 'credit_amount', 'credit_memo', 'credit_rating']
client_credit_2016.to_csv('data_output/zhengtai_csv/client_credit_2016.csv')

client_credit_2017_first = pd.merge(all_client, whitelist, on='经销商名称', how='left')
client_credit_2017_first['year'] = 2017
client_credit_2017_second = pd.merge(client_credit_2017_first, zhibiao_2017, on='经销商名称', how='left')
client_credit_2017_third = pd.merge(client_credit_2017_second, shouxin_2017[['经销商名称', '17授信比例', '17年信用额', '2017年授信表备注']], on='经销商名称', how='left')
client_credit_2017_third['17年授信等级'] = np.NaN
client_credit_2017 = client_credit_2017_third
client_credit_2017.columns = ['client_name', 'whitelist', 'year', 'sales_goal', 'credit_ratio', 'credit_amount', 'credit_memo', 'credit_rating']
client_credit_2017.to_csv('data_output/zhengtai_csv/client_credit_2017.csv')

client_credit_first = pd.concat([client_credit_2014, client_credit_2015], axis=0)
client_credit_second = pd.concat([client_credit_first, client_credit_2016], axis=0)
client_credit = pd.concat([client_credit_second, client_credit_2017], axis=0)
client_credit.to_csv('data_output/zhengtai_csv/client_credit20171130V0.2.csv')



client_balance = pd.read_csv('data_output/data_for_model/client_balance1128V0.1.csv', encoding='gbk')
client_balance.groupby(by=['year', 'month']).agg({
    'receivable': np.sum,
    'payment': np.sum
}).to_clipboard()

client_balance[client_balance['document_type'] == '期初余额'].to_clipboard()

a = client_balance.groupby(by=['year', 'month']).agg({
    'receivable': np.sum,
    'payment': np.sum
}).reset_index()

import sys
sys.path.append('univariate')

from univariate.analysis import UnivariateAnalysis
obj = UnivariateAnalysis(data=a)
obj.get_variable_description()