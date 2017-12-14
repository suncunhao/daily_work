#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/13 9:35
# @Author  : sch
# @File    : 2017_12_13.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy import stats

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

data = pd.read_excel('data_output/ZTmodule/rp.xlsx', encoding='gbk')
y = pd.read_csv('data_output/ZTmodule/CS_index_nan20171210V0.4.csv', encoding='gbk')
y = y[['client_name', 'year', 'credit_ratio']]
rp_data = pd.merge(data, y, left_on=['year', 'name'], right_on=['year', 'client_name'], how='left')
rp_data_useful = rp_data.dropna()
X_rp = rp_data_useful[rp_data_useful.columns[2:-3]]
y_rp = rp_data_useful['credit_ratio']
X_rp_train, X_rp_test, y_rp_train, y_rp_test = train_test_split(X_rp, y_rp, test_size=0.3, random_state=0)

X2 = sm.add_constant(X_rp_train)
est = sm.OLS(y_rp_train, X2)
est2 = est.fit()
pvalue = est2.pvalues

clf1 = LinearRegression()
clf1.fit(X_rp_train, y_rp_train)

final = pd.merge(pd.concat([pd.concat([pd.Series(['const']), pd.Series(clf1.intercept_)], axis=1), pd.concat([pd.Series(X_rp_train.columns), pd.Series(clf1.coef_)], axis=1)], axis=0), pd.DataFrame(pvalue).reset_index().rename(columns={'index':0, 0 :1}), on=0)
final.columns = ['name', 'Value', 'p-value']
final.to_csv('data_output/dateorder/20171213/coef.csv')
metrics.r2_score(y_rp_train, clf1.predict(X_rp_train))
metrics.mean_squared_error(y_rp_test, clf1.predict(X_rp_test))

plt.plot(y_rp_test.reset_index()['credit_ratio'], label='y')
plt.plot(clf1.predict(X_rp_test), label='y_hat')
plt.legend()
plt.title('Prediction of credit_ratio in testdata(RP)')


#######################
client_feature = pd.read_csv('data_output/dateorder/20171122/client_feature.csv', encoding='gbk')
whitelist = pd.read_csv('data_output/zhengtai_csv/whitelist.csv', encoding='gbk')
new_client_feature = pd.merge(client_feature, whitelist, on='client_name', how='left')
new_client_feature = new_client_feature.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], 1)
new_client_feature['whitelist'] = new_client_feature['whitelist'].fillna(0)
new_client_feature.to_csv('data_output/dateorder/20171213/client_feature1213V0.1.csv')







###################
# MDP
R = np.array([-2, -2, -2, 10, 1, -1, 0]).reshape(-1, 1)
P = [[0, 0.5, 0, 0, 0, 0.5, 0],
    [0, 0, 0.5, 0, 0, 0, 0.5],
    [0, 0, 0, 0.5, 0.5, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0.2, 0.4, 0.4, 0, 0, 0, 0],
    [0.5, 0, 0, 0, 0, 0.5, 0],
    [0, 0, 0, 0, 0, 0, 1]]
V = R + np.dot(P, R)

for i in range(1, 500000):
    V = R + np.dot(P, V)
print(V)


R = np.array([-2, -2, -2, 10, 1, -1, 0]).reshape(-1, 1)
P2 = [[0, 0.5, 0, 0, 0, 0.5, 0],
    [0, 0, 0.8, 0, 0, 0, 0.2],
    [0, 0, 0, 0.6, 0.4, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0.2, 0.4, 0.4, 0, 0, 0, 0],
    [0.1, 0, 0, 0, 0, 0.9, 0],
    [0, 0, 0, 0, 0, 0, 1]]
V = R + np.dot(P2, R)
V = R + np.dot(P2, V)

for i in range(1, 5000):
    V = R + np.dot(P2, V)
print(V)



##################################
cs = pd.read_csv('data_output/ZTmodule/CS_index_nan20171210V0.4.csv', encoding='gbk')
from variable_analysis.univariate import analysis
from variable_analysis.univariate.common import graph
obj = graph.boxplot(y1=cs['credit_ratio'], x1=cs['credit_rating'])
obj = graph.density_hist_curve(x=cs['credit_ratio'].dropna())
obj = graph.scatter_continuous(x1=cs['credit_ratio'], y1=cs['CS1'])
obj = graph.scatter_category(x1=cs['credit_rating'], y1=cs['credit_ratio'])
obj = graph.line_chart(x1=cs['credit_ratio'], y1=cs['CS1'], z1=cs['credit_rating'])
