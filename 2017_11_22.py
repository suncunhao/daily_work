#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/22 9:30
# @Author  : sch
# @File    : 2017_11_22.py

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

whitelist = pd.read_clipboard()
all_name = pd.read_clipboard()
new_data = pd.merge(all_name, whitelist, how='left', on='经销商名称')
new_data.to_clipboard()


yingshou_2014_1 = pd.read_clipboard()
# 时间清理
def clean_time(data, day, month):
    data['日'] = data['日'].fillna(day)
    data['月'] = data['月'].fillna(month)
clean_time(yingshou_2014_1, 1, 1)

yingshou_2014_2 = pd.read_clipboard()
yingshou_2014 = yingshou_2014_1.append(yingshou_2014_2)

yingshou_2014['客户简称'] = yingshou_2014['客户简称'].str.split('-')
yingshou_2014['客户简称'] = yingshou_2014['客户简称'].apply(lambda x: x[0] if type(x) == list else x)
yingshou_2014.to_csv('data_output/zhengtai_csv/yingshou_2015.csv')

#####################################
yingshou_2016_1 = pd.read_clipboard()
# 时间清理
def clean_time(data, day, month):
    data['日'] = data['日'].fillna(day)
    data['月'] = data['月'].fillna(month)
clean_time(yingshou_2016_1, 1, 1)

yingshou_2016_2 = pd.read_clipboard()
yingshou_2016_temp = yingshou_2016_1.append(yingshou_2014_2)

yingshou_2016_3 = pd.read_clipboard()
yingshou_2016 = yingshou_2016_temp.append(yingshou_2016_3)


yingshou_2016['客户简称'] = yingshou_2016['客户简称'].str.split('-')
yingshou_2016['客户简称'] = yingshou_2016['客户简称'].apply(lambda x: x[0] if type(x) == list else x)
yingshou_2016.to_csv('data_output/zhengtai_csv/yingshou_2016.csv')

shouxin_all = pd.read_clipboard()
shouxin_all.to_csv('data_output/zhengtai_csv/shouxin_all.csv')

jingxiaoshang = pd.read_clipboard()
jingxiaoshang.to_csv('data_output/zhengtai_csv/jingxiaoshang.csv')

jiandang_2014 = pd.read_clipboard()
jiandang_2015 = pd.read_clipboard()
jiandang_2016 = pd.read_clipboard()
jiandang_2016['客户简称'] = jiandang_2016['客户简称'].str.split('-')
jiandang_2016['客户简称'] = jiandang_2016['客户简称'].apply(lambda x: x[0] if type(x) == list else x)

data_all_2014 = pd.merge(jingxiaoshang, jiandang_2014, on='客户简称', how='left')
data_all_2015 = pd.merge(jingxiaoshang, jiandang_2015, on='客户简称', how='left')
data_all_2016 = pd.merge(jingxiaoshang, jiandang_2016, on='客户简称', how='left')

data_all_2014.to_csv('data_output/zhengtai_csv/company_2014.csv')
data_all_2015.to_csv('data_output/zhengtai_csv/company_2015.csv')
data_all_2016.to_csv('data_output/zhengtai_csv/company_2016.csv')



level_14 = pd.read_clipboard()
draw_level_14 = level_14[level_14['15授信比例'].isnull() == False]
ax1 = sns.boxplot(draw_level_14['14信用等级'], draw_level_14['15授信比例'])
ax1.set_title('14信用等级-15授信比例')

level_15 = pd.read_clipboard()
draw_level_15 = level_15[level_15['16授信比例'].isnull() == False]
ax2 = sns.boxplot(draw_level_15['15信用等级'], draw_level_15['16授信比例'])
ax2.set_title('15信用等级-16授信比例')

level_16 = pd.read_clipboard()
draw_level_16 = level_16[level_16['17授信比例'].isnull() == False]
ax3 =sns.boxplot(draw_level_16['16信用等级'], draw_level_16['17授信比例'])
ax3.set_title('16信用等级-17授信比例')

shouxin = pd.read_clipboard()
calculate_14 = shouxin.groupby(by='14信用等级', axis=0)['15授信比例'].value_counts()
calculate_15 = shouxin.groupby(by='15信用等级', axis=0)['16授信比例'].value_counts()
calculate_16 = shouxin.groupby(by='16信用等级', axis=0)['17授信比例'].value_counts()
a = pd.concat([calculate_14, calculate_15,calculate_16],axis=1)
a.to_csv('data_output/20171122/level_stat.csv')

level_all = pd.read_clipboard()
draw_level_all = level_all[level_all['授信比例'].isnull() == False]
ax4 = sns.boxplot(draw_level_all['信用等级'], draw_level_all['授信比例'])
ax4.set_title('全部信用等级-授信比例')


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
data = pd.read_clipboard(parse_dates=['start_date'])
# data = pd.read_clipboard()
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
data_new.to_csv('data_output/20171122/company_time.csv')

data_new = pd.read_clipboard()
information = pd.read_clipboard()
data_alll = pd.merge(data_new, information, on='client_name', how='left')
data_alll = data_alll.drop('经销商编码', 1)
data_alll['main_business_zhengtai'] = data_alll['是否主营正泰'].apply(lambda x: 1 if x=='是' else 0)
data_alll = data_alll.drop('是否主营正泰', 1)
data_alll.to_csv('data_output/20171122/client_feature.csv')


# client_balance
jingxiaoshang = pd.read_clipboard()

yingshou_2014 = pd.read_clipboard()
yingshou_2014_merge = pd.merge(jingxiaoshang, yingshou_2014, how='left', on='客户简称')
yingshou_2014_merge.to_csv('data_output/20171122/yingshou_2016.csv')


import re

yingshou_2014_merge = pd.read_clipboard()
yingshou_2014_merge['receivable'] = yingshou_2014_merge['receivable'].apply(
    lambda x: float(re.sub(',', '' ,x))
)

month_sum = yingshou_2014_merge.groupby('month').agg({
    'receivable': lambda x: np.sum(x > 0)
}).reset_index()


date = pd.DataFrame(pd.date_range('2014-1-1', '2014-12-31'))

date_range = pd.date_range('2014-1-1', '2014-12-31', freq='M')
date_range.month

data = pd.DataFrame(date_range.month, columns=['month'])


new = pd.merge(data, month_sum, on='month', how='left')
new.fillna(0)


