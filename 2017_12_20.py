#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/20 9:35
# @Author  : sch
# @File    : 2017_12_20.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

feature = pd.read_csv('data_output/data_for_model/client_feature1213V0.1.csv', encoding='gbk')
mapping = {'新疆自治区': '新疆维吾尔自治区', '广西自治区': '广西壮族自治区', '浙南区': '浙江省', '浙北区': '浙江省'}
feature['省'] = feature['省'].map(mapping)
feature.to_csv('data_output/data_for_model/client_feature1220V0.2.csv')

file = 'en_data.json'
f = open(file, 'r', encoding='utf8')
result = []
for line in f:
    result.append(line)
result_json_str = ','.join(result)
result_json_str = '[' + result_json_str + ']'
data = json.loads(result_json_str)
d = pd.DataFrame(data)
d['idxtype'] = d['idxtype'].apply(np.int)
d['data'] = d['data'].apply(np.float)
d['typeid'].apply(lambda x: np.int(x) if x != 'null' else x)

d['year'] = d['datetime'].apply(lambda x: x.split('-')[0])
d['month'] = d['datetime'].apply(lambda x: x.split('-')[1])
d_sel = d[(d['idxtype'] == 2) | (d['idxtype'] == 8) | (d['idxtype'] == 11) | (d['idxtype'] == 14) | (d['idxtype'] == 15)]
# d_sel = d[d['idxtype'].isin([2, 8, 11, 14, 15])]
en_40 = d_sel.groupby(by=['idxtype', 'year']).agg({
    'data': np.mean
})
en_40.to_csv('data_output/dateorder/20171220/en40.csv')

d_111 = d[d['idxtype'] == 34]
en_111 = d_111[d_111['year'].isin(['2014', '2015', '2016'])]
en_111.to_csv('data_output/dateorder/20171220/en111.csv')

d_112_2014 = d_111[d_111['year'].isin(['2012', '2013', '2014'])]
en_112_2014 = d_112_2014.groupby(by=['typeid']).agg({
    'data': np.mean
}).reset_index()
en_112_2014['year'] = 2014

d_112_2015 = d_111[d_111['year'].isin(['2013', '2014', '2015'])]
en_112_2015 = d_112_2015.groupby(by=['typeid']).agg({
    'data': np.mean
}).reset_index()
en_112_2015['year'] = 2015

en_112 = pd.concat([en_112_2014, en_112_2015], axis=0)
en_112.to_csv('data_output/dateorder/20171220/en112.csv')


data = pd.read_clipboard()
province = data['地区']
province = pd.DataFrame(province)
province.columns = ['province']
year = pd.DataFrame([2014, 2015, 2016, 2017], columns=['year'])
new_df = pd.DataFrame(columns=['year', 'province'])
# for p_index, p_row in province.iterrows():
#     for y_index, y_row in year.iterrows():
#         p_data = p_row['地区']
#         y_data = y_row['year']
#
#         row = pd.DataFrame([dict(province=p_data, year=y_data), ])
#         new_df = new_df.append(row, ignore_index=True)

def getMergeAB(A,B):
    newDf = pd.DataFrame(columns=['year', 'province'])
    for _, A_row in A.iterrows():
        for _, B_row in B.iterrows():
            AData = A_row['province']
            BData = B_row['year']
            row = pd.DataFrame([dict(province=AData, year=BData)])
            newDf = newDf.append(row, ignore_index=True)
    return newDf
# 另一种生成法
# province['test'] = 1
# year['test'] = 1
# new_df = pd.merge(province, year, on='test', how='left')
# new_df = new_df.drop('test', 1)
new_df = getMergeAB(province, year)

en40 = pd.read_csv('data_output/dateorder/20171220/en40.csv', encoding='gbk')
new_en40 = pd.merge(new_df, en40[en40['idxtype'] == 2][['data', 'year']], on='year', how='left')
new_en40 = new_en40.rename(columns={'data': 'en40'})
new_en92 = pd.merge(new_en40, en40[en40['idxtype'] == 8][['data', 'year']], on='year', how='left')
new_en92 = new_en92.rename(columns={'data': 'en92'})
new_en12 = pd.merge(new_en92, en40[en40['idxtype'] == 11][['data', 'year']], on='year', how='left')
new_en12 = new_en12.rename(columns={'data': 'en12'})
new_en62 = pd.merge(new_en12, en40[en40['idxtype'] == 14][['data', 'year']], on='year', how='left')
new_en62 = new_en62.rename(columns={'data': 'en62'})
new_en52 = pd.merge(new_en62, en40[en40['idxtype'] == 15][['data', 'year']], on='year', how='left')
new_en52 = new_en52.rename(columns={'data': 'en52'})

en_111 = pd.read_csv('data_output/dateorder/20171220/en111.csv', encoding='gbk')
mapping = data.set_index('编号').to_dict()
en_111['province'] = en_111['typeid'].map(mapping['地区'])
en_111 = en_111[['province', 'year', 'data']]
new_en111 = pd.merge(new_en52, en_111, on=['year', 'province'], how='left')
new_en111 = new_en111.rename(columns={'data': 'en111'})

en_112 = pd.read_csv('data_output/dateorder/20171220/en112.csv', encoding='gbk')
en_112['province'] = en_112['typeid'].map(mapping['地区'])
en_112 = en_112[['province', 'year', 'data']]
new_en112 = pd.merge(new_en111, en_112, on=['year', 'province'], how='left')
new_en112 = new_en112.rename(columns={'data': 'en112'})

new_en112.to_csv('data_output/dateorder/20171220/en1220V0.2.csv')

