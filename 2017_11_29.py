#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/29 10:03
# @Author  : sch
# @File    : 2017_11_29.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

data = pd.read_csv('data_output/20171123/shouxinzhibiao.csv', encoding='gbk')
data = data.drop(['Unnamed: 0', '财务总指标', '年初信用额'], 1)
data.columns = ['year', 'whitelist', 'client_name', 'sales_goal', 'credit_ratio', 'credit_amount', 'credit_memo', 'credit_rating']
data['sales_goal'] = data['sales_goal'] * 10000
data['credit_amount'] = data['credit_amount'] * 10000
data.to_csv('data_output/20171129/client_credit1129V0.1.csv')

#######################

index_season = pd.read_csv('data_output/ZTmodule/index_season.csv', encoding='gbk')
client_credit = pd.read_csv('data_output/data_for_model/client_credit1129V0.1.csv', encoding='gbk')
client_credit = client_credit[['year', 'client_name', 'credit_ratio', 'credit_rating']]
all_data = pd.merge(index_season, client_credit, on=['year', 'client_name'], how='left')

def fill_inf(x):
    if x == np.inf:
        return np.nan
    elif x == -np.inf:
        return np.nan
    else:
        return x

for i in all_data.columns:
    all_data[i] = all_data[i].apply(fill_inf)

X = all_data[['rev_avg', 'rev_avg_c', 'rev_log',
              'rev_log_c', 'rev_incr', 'rev_change', 'rev_incr_c', 'rev_change_c',
              'rev_max', 'rev_max_c', 'rev_max_log', 'rev_max_log_c', 'rev_min',
              'rev_min_c', 'rev_min_log', 'rev_min_log_c', 'rev_max_p', 'rev_max_p_c',
              'rev_min_p', 'rev_min_p_c', 'rev_std', 'rev_std_c', 'rev_std_log',
              'rev_std_log_c', 'rev_incr_yoy', 'rev_incr_yoy_c', 'rev_change_yoy',
              'rev_max_yoy', 'rev_min_yoy', 'rev_std_yoy', 'rev_change_yoy_c',
              'rev_max_yoy_c', 'rev_min_yoy_c', 'rev_std_yoy_c']]
y_1 = all_data['credit_ratio']
y_2 = all_data['credit_rating']
model_1 = pd.concat([X, y_1], axis=1)
model_2 = pd.concat([X, y_2], axis=1)
model_1 = model_1.dropna()
model_2 = model_2.dropna()
model_1['is_train'] = np.random.uniform(0, 1, len(model_1)) <= 0.75
model_2['is_train'] = np.random.uniform(0, 1, len(model_2)) <= 0.75
# 分离出训练集与测试集
X1_train = model_1[model_1['is_train'] == True]
X1_test = model_1[model_1['is_train'] == False]
X_1_train = model_1[model_1['is_train'] == True].drop(['credit_ratio', 'is_train'], 1)
X_1_test =model_1[model_1['is_train'] == False].drop(['credit_ratio', 'is_train'], 1)
y_1_train = model_1[model_1['is_train'] == True]['credit_ratio']
y_1_test = model_1[model_1['is_train'] == False]['credit_ratio']
clf1 = RandomForestRegressor(n_estimators=20, random_state=0)
clf1.fit(X_1_train, y_1_train)
clf1.predict(X_1_train)
clf1.predict(X_1_test)
model_1_train = pd.concat([X1_train.reset_index().drop('index', 1), pd.DataFrame(clf1.predict(X_1_train))], axis=1)
model_1_test = pd.concat([X1_test.reset_index().drop('index', 1), pd.DataFrame(clf1.predict(X_1_test))], axis=1)
model_1_train.to_csv('data_output/20171129/model_regressor_train.csv')
model_1_test.to_csv('data_output/20171129/model_regressor_test.csv')


model_2['credit_rating'] = model_2['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
X2_train = model_2[model_2['is_train'] == True]
X2_test = model_2[model_2['is_train'] == False]
X_2_train = model_2[model_2['is_train'] == True].drop(['credit_rating', 'is_train'], 1)
X_2_test =model_2[model_2['is_train'] == False].drop(['credit_rating', 'is_train'], 1)
y_2_train = model_2[model_2['is_train'] == True]['credit_rating']
y_2_test = model_2[model_2['is_train'] == False]['credit_rating']
clf2 = RandomForestClassifier(n_estimators=20, random_state=0)
clf2.fit(X_2_train, y_2_train)
clf2.predict(X_2_train)
clf2.predict(X_2_test)
model_2_train = pd.concat([X2_train.reset_index().drop('index', 1), pd.DataFrame(clf2.predict(X_2_train))], axis=1)
model_2_test = pd.concat([X2_test.reset_index().drop('index', 1), pd.DataFrame(clf2.predict(X_2_test))], axis=1)
model_2_train.to_csv('data_output/20171129/model_classifier_train.csv')
model_2_test.to_csv('data_output/20171129/model_classifier_test.csv')

# 评估预测效果
from sklearn import metrics
metrics.mean_squared_error(y_1_train, clf1.predict(X_1_train))
metrics.mean_squared_error(y_1_test, clf1.predict(X_1_test))

metrics.confusion_matrix(y_2_train, clf2.predict(X_2_train))
metrics.confusion_matrix(y_2_test, clf2.predict(X_2_test))

train = pd.read_clipboard()
fit = plt.plot(train['y'])
fit = plt.plot(train['y_hat'])
plt.legend()
plt.title('训练集表现')
test = pd.read_clipboard()
fit = plt.plot(test['y'])
fit = plt.plot(test['y_hat'])
plt.legend()
plt.title('测试集表现')

plt.matshow(metrics.confusion_matrix(y_2_train, clf2.predict(X_2_train)), cmap=plt.cm.Blues)
plt.colorbar()
plt.legend()
plt.title('训练集表现')
plt.matshow(metrics.confusion_matrix(y_2_test, clf2.predict(X_2_test)), cmap=plt.cm.Blues)
plt.colorbar()
plt.legend()
plt.title('测试集表现')



OSmodule = pd.read_clipboard()
OS_1 = OSmodule
OS_2 = pd.merge(OSmodule, client_credit, on=['year', 'client_name'], how='left')

OS_1['credit_rating'] = OS_1['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)

OS_1['is_train'] = np.random.uniform(0, 1, len(OS_1)) <= 0.75

X1_train = OS_1[OS_1['is_train'] == True]
X1_test = OS_1[OS_1['is_train'] == False]
X_1_train = OS_1[OS_1['is_train'] == True].drop(['credit_rating', 'is_train', 'client_name', 'year'], 1)
X_1_test =OS_1[OS_1['is_train'] == False].drop(['credit_rating', 'is_train', 'client_name', 'year'], 1)
y_1_train = OS_1[OS_1['is_train'] == True]['credit_rating']
y_1_test = OS_1[OS_1['is_train'] == False]['credit_rating']
clf1 = RandomForestClassifier(n_estimators=20, random_state=0)
clf1.fit(X_1_train, y_1_train)
clf1.predict(X_1_train)
clf1.predict(X_1_test)
OS_classifier_train = pd.concat([X1_train.reset_index().drop('index', 1), pd.DataFrame(clf1.predict(X_1_train))], axis=1)
OS_classifier_train.to_csv('data_output/20171129/OS_classifier_train.csv')
metrics.confusion_matrix(y_1_train, clf1.predict(X_1_train))
OS_classifier_test = pd.concat([X1_test.reset_index().drop('index', 1), pd.DataFrame(clf1.predict(X_1_test))], axis=1)
OS_classifier_test.to_csv('data_output/20171129/OS_classifier_test.csv')
metrics.confusion_matrix(y_1_test, clf1.predict(X_1_test))



OS_2['is_train'] = np.random.uniform(0, 1, len(OS_1)) <= 0.75

OS_2 = OS_2.drop(['credit_rating_x', 'client_name', 'year', 'credit_rating_y'], 1)

OS_2 = OS_2.dropna()
X2_train = OS_2[OS_2['is_train'] == True]
X2_test = OS_2[OS_2['is_train'] == False]
X_2_train = OS_2[OS_2['is_train'] == True].drop(['credit_ratio', 'is_train'], 1)
X_2_test =OS_2[OS_2['is_train'] == False].drop(['credit_ratio', 'is_train'], 1)
y_2_train = OS_2[OS_2['is_train'] == True]['credit_ratio']
y_2_test = OS_2[OS_2['is_train'] == False]['credit_ratio']
clf2 = RandomForestRegressor(n_estimators=20, random_state=0)
clf2.fit(X_2_train, y_2_train)
clf2.predict(X_2_train)
clf2.predict(X_2_test)
OS_regressor_train = pd.concat([X2_train.reset_index().drop('index', 1), pd.DataFrame(clf2.predict(X_2_train))], axis=1)
OS_regressor_train.to_csv('data_output/20171129/OS_regressor_train.csv')
OS_regressor_test = pd.concat([X2_test.reset_index().drop('index', 1), pd.DataFrame(clf2.predict(X_2_test))], axis=1)
OS_regressor_test.to_csv('data_output/20171129/OS_regressor_test.csv')
metrics.mean_squared_error(y_2_train, clf2.predict(X_2_train))
metrics.mean_squared_error(y_2_test, clf2.predict(X_2_test))




