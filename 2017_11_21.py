
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/21 9:15
# @Author  : sch
# @File    : 2017_11_21.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# client_credit
data_shouxin_2016 = pd.read_clipboard()
data_xinyong_2016 = pd.read_clipboard()
data_xinyong_2016 = data_xinyong_2016.drop('15信用等级', 1)
data_temp = pd.merge(data_shouxin_2016, data_xinyong_2016, on='经销商名称', how='left')

data_1 = pd.read_clipboard()
data_2 = pd.read_clipboard()
data_3 = pd.read_clipboard()
data_1 = data_1.drop(['经销商编码', '结算单位名称', '结算单位编码'], 1)
data_2 = data_2.drop(['经销商编码', '结算单位名称', '结算单位编码'], 1)
data_3 = data_3.drop(['经销商编码', '结算单位名称', '结算单位编码'], 1)
data_12 = pd.merge(data_1, data_2, on='经销商名称', how='outer')
data_all_offline = pd.merge(data_12, data_3, on='经销商名称', how='outer')

data_all = pd.merge(data_temp, data_all_offline, on='经销商名称', how='left')
data_all.to_csv('data_output/20171121/all_2016.csv')
####################################################
data_shouxin_2015 = pd.read_clipboard()
data_xinyong_2015 = pd.read_clipboard()
data_xinyong_2015 = data_xinyong_2015.drop('14信用等级', 1)
data_temp = pd.merge(data_shouxin_2015, data_xinyong_2015, on='经销商名称', how='left')

data_all_offline = pd.read_clipboard()
data_all_offline = data_all_offline.drop('结算单位名称', 1)

data_all = pd.merge(data_temp, data_all_offline, on='经销商名称', how='left')
data_all.to_csv('data_output/20171121/all_2015.csv')


# client_balance
data1 = pd.read_clipboard()
# 时间清理
def clean_time(data, day, month):
    data['日'] = data['日'].fillna(day)
    data['月'] = data['月'].fillna(month)

data2 = pd.read_clipboard()

data_all = data1.append(data2)









# plot
data = pd.read_clipboard()
p_float = data['16授信比例'].str.strip("%").astype(float)/100;
data['16授信比例'] = p_float
fig1 = sns.boxplot(x=data['16信用等级'], y=data['16总指标'])
fig1.set_title('信用等级-总指标')

dictionary = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
data['16信用得分'] = data['16信用等级'].map(dictionary)
sns.jointplot('16信用得分', '16总指标', data)

data_1 = pd.read_clipboard()
data_1['本期应收'] = data_1['本期应收'].apply(lambda x: re.sub(',', '', x))
data_1[data_1['本期应收'].astype(float) > 0]['本期应收'].count()
data_1['本期应收'].astype(float).sum()

data_1['余额'] = data_1['余额'].apply(lambda x: float(re.sub(',', '', x)))
data_1['余额'].astype(float).max()


d = pd.DataFrame()
d['余额'] = data_1['余额']
d.reset_index(inplace=True)

a = d[d['余额'] < -300000]
np.max(a['index'].diff())







all_name = pd.read_clipboard()
data1 = pd.read_clipboard()
data2 = pd.read_clipboard()

def get_None(x, data, group, target):
    if x in data[group].values:
        return data.loc[data[group] == x, target].iloc[0]
    else:
        return 'NA'
all_name['16信用等级'] = all_name['经销商名称'].apply(lambda x: get_None(x, data1, '经销商名称', '16信用等级'))
all_name['16线上总指标'] = all_name['经销商名称'].apply(lambda x:get_None(x, data2, '经销商名称', '16线上总指标'))
all_name['16授信比例'] = all_name['经销商名称'].apply(lambda x:get_None(x, data2, '经销商名称', '16授信比例'))
all_name['2016年授信表备注'] = all_name['经销商名称'].apply(lambda x:get_None(x, data2, '经销商名称', '2016年授信表备注'))

offline_1 = pd.read_clipboard()
all_name['2016年线下电工'] = all_name['经销商名称'].apply(lambda x:get_None(x, offline_1, '经销商名称', '2016年电工销售指标'))
offline_2 = pd.read_clipboard()
all_name['2016年线下照明'] = all_name['经销商名称'].apply(lambda x:get_None(x, offline_2, '经销商名称', '2016年照明总指标'))
offline_3 = pd.read_clipboard()
all_name['2016年线下排插'] = all_name['经销商名称'].apply(lambda x:get_None(x, offline_3, '经销商名称', '2016年排插销售指标'))

all_name.to_csv('data_output/20171121/2016shouxin_new.csv')





all_name = pd.read_clipboard()
xiaoshou_2014 = pd.read_clipboard()
# xiaoshou_2014 = xiaoshou_2014.drop('结算单位名称', 1)
all_name['14线下指标'] = all_name['经销商名称'].apply(lambda x: get_None(x, xiaoshou_2014, '经销商名称', '2014年销售指标'))
shouxin_2015 = pd.read_clipboard()
all_name['14授信'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2015, '经销商名称', '14信用等级'))
xiaoshou_2015 = pd.read_clipboard()
all_name['15线下指标'] = all_name['经销商名称'].apply(lambda x: get_None(x, xiaoshou_2015, '经销商名称', '2015年销售指标'))
shouxin_2015 = pd.read_clipboard()
all_name['15总指标'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2015, '经销商名称', '15总指标'))
all_name['15授信比例'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2015, '经销商名称', '15授信比例'))
all_name['15信用额'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2015, '经销商名称', '15信用额'))
all_name['15备注'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2015, '经销商名称', '15年备注'))
all_name_confirm = all_name.copy()

shouxin_2016 = pd.read_clipboard()
all_name['15信用等级'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2016, '经销商名称', '15信用等级'))
offline_diangong_2016 = pd.read_clipboard()
all_name['16线下电工'] = all_name['经销商名称'].apply(lambda x: get_None(x, offline_diangong_2016, '经销商名称', '2016年电工销售指标'))
offline_zhaoming_2016 = pd.read_clipboard()
all_name['16线下照明'] = all_name['经销商名称'].apply(lambda x: get_None(x, offline_zhaoming_2016, '经销商名称', '2016年照明总指标'))
offline_paicha_2016 = pd.read_clipboard()
all_name['16线下排插'] = all_name['经销商名称'].apply(lambda x: get_None(x, offline_paicha_2016, '经销商名称', '2016年排插销售指标'))
all_name_confirm2 = all_name.copy()

shouxin_2016 = pd.read_clipboard()
all_name['16线上总指标'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2016, '经销商名称', '16总指标'))
all_name['16授信比例'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2016, '经销商名称', '16授信比例'))
all_name['16信用额'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2016, '经销商名称', '16年信用额'))
all_name['16备注'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2016, '经销商名称', '16年备注'))
all_name_confirm3 = all_name.copy()

shouxin_2017 = pd.read_clipboard()
all_name['16信用等级'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2017, '经销商名称', '16信用等级'))
offline_diangong_2017 = pd.read_clipboard()
all_name['17线下电工'] = all_name['经销商名称'].apply(lambda x: get_None(x, offline_diangong_2017, '经销商名称', '2017年电工销售指标'))
offline_zhaoming_2017 = pd.read_clipboard()
all_name['17线下照明'] = all_name['经销商名称'].apply(lambda x: get_None(x, offline_zhaoming_2017, '经销商名称', '2017年照明总指标'))
offline_paicha_2017 = pd.read_clipboard()
all_name['17线下排插'] = all_name['经销商名称'].apply(lambda x: get_None(x, offline_paicha_2017, '经销商名称', '2017年排插总指标'))
all_name_confirm4 = all_name.copy()

shouxin_2017 = pd.read_clipboard()
all_name['17线上总指标'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2017, '经销商名称', '17总指标'))
all_name['17授信比例'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2017, '经销商名称', '17授信比例'))
all_name['17信用额'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2017, '经销商名称', '17年信用额'))
all_name['17备注'] = all_name['经销商名称'].apply(lambda x: get_None(x, shouxin_2017, '经销商名称', '17年备注'))
all_name_confirm5 = all_name.copy()

all_name.to_csv('data_output/20171121/all_2016_shouxin.csv')



company_2014 = pd.read_clipboard()


