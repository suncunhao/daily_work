#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/1 9:36
# @Author  : sch
# @File    : 2017_12_01.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

season_module = pd.read_csv('data_output/ZTmodule/index_season20171201v0.5.csv', encoding='gbk')
season_module_valid = season_module[['client_name', 'year', 'rev_log', 'rev_incr', 'rev_std_log', 'rev_max_log', 'rev_min_log', 'rev_max_p', 'rev_min_p', 'rev_incr_yoy', 'rev_change_yoy', 'rev_max_yoy', 'rev_min_yoy', 'rev_std_yoy', 'rev_log_c', 'rev_incr_c', 'rev_change_c', 'rev_std_log_c', 'rev_max_log_c', 'rev_min_log_c', 'rev_max_p_c', 'rev_min_p_c', 'rev_incr_yoy_c', 'rev_change_yoy_c', 'rev_max_yoy_c', 'rev_min_yoy_c', 'rev_std_yoy_c', 'credit_rating', 'credit_ratio']]
season_module_valid = season_module_valid[season_module_valid['year'] == 2015].append(season_module_valid[season_module_valid['year'] == 2016])

# 全25指标，全样本随机森林回归
season_module_valid_ratio = season_module_valid.drop('credit_rating', 1).dropna()
X_all_ratio = season_module_valid_ratio[['rev_log', 'rev_incr', 'rev_std_log', 'rev_max_log', 'rev_min_log', 'rev_max_p', 'rev_min_p', 'rev_incr_yoy', 'rev_change_yoy', 'rev_max_yoy', 'rev_min_yoy', 'rev_std_yoy', 'rev_log_c', 'rev_incr_c', 'rev_change_c', 'rev_std_log_c', 'rev_max_log_c', 'rev_min_log_c', 'rev_max_p_c', 'rev_min_p_c', 'rev_incr_yoy_c', 'rev_change_yoy_c', 'rev_max_yoy_c', 'rev_min_yoy_c', 'rev_std_yoy_c']]
y_all_ratio = season_module_valid_ratio['credit_ratio']
clf_ratio = RandomForestRegressor(n_estimators=20, random_state=0)
clf_ratio.fit(X_all_ratio, y_all_ratio)
ratio_predict = pd.DataFrame(clf_ratio.predict(X_all_ratio))
ratio_predict.columns = ['ratio_hat']
clf_ratio_result = pd.concat([season_module_valid_ratio.reset_index().drop('index', 1), ratio_predict], axis=1)
clf_ratio_result.to_csv('data_output/20171201/season_module_all_ratio.csv')
metrics.mean_squared_error(y_all_ratio, clf_ratio.predict(X_all_ratio))

# 全25指标，全样本随机森林分类
season_module_valid_rating = season_module_valid.drop('credit_ratio', 1).dropna()
X_all_rating = season_module_valid_rating[['rev_log', 'rev_incr', 'rev_std_log', 'rev_max_log', 'rev_min_log', 'rev_max_p', 'rev_min_p', 'rev_incr_yoy', 'rev_change_yoy', 'rev_max_yoy', 'rev_min_yoy', 'rev_std_yoy', 'rev_log_c', 'rev_incr_c', 'rev_change_c', 'rev_std_log_c', 'rev_max_log_c', 'rev_min_log_c', 'rev_max_p_c', 'rev_min_p_c', 'rev_incr_yoy_c', 'rev_change_yoy_c', 'rev_max_yoy_c', 'rev_min_yoy_c', 'rev_std_yoy_c']]
y_all_rating = season_module_valid_rating['credit_rating']
clf_rating = RandomForestClassifier(n_estimators=20, random_state=0)
clf_rating.fit(X_all_rating, y_all_rating)
rating_predict = pd.DataFrame(clf_rating.predict(X_all_rating))
rating_predict.columns = ['rating_hat']
clf_rating_result = pd.concat([season_module_valid_rating.reset_index().drop('index', 1), rating_predict], axis=1)
clf_rating_result.to_csv('data_output/20171201/season_module_all_rating.csv')
metrics.confusion_matrix(y_all_rating, clf_rating.predict(X_all_rating))

# 全25指标，有测试集随机森林回归
season_module_valid_ratio = season_module_valid.drop('credit_rating', 1).dropna()
season_module_valid_ratio['is_train'] = np.random.uniform(0, 1, len(season_module_valid_ratio)) <= 0.7
season_module_valid_ratio_train = season_module_valid_ratio[season_module_valid_ratio['is_train'] == True]
season_module_valid_ratio_test = season_module_valid_ratio[season_module_valid_ratio['is_train'] == False]
X_ratio_train = season_module_valid_ratio_train[['rev_log', 'rev_incr', 'rev_std_log', 'rev_max_log', 'rev_min_log', 'rev_max_p', 'rev_min_p', 'rev_incr_yoy', 'rev_change_yoy', 'rev_max_yoy', 'rev_min_yoy', 'rev_std_yoy', 'rev_log_c', 'rev_incr_c', 'rev_change_c', 'rev_std_log_c', 'rev_max_log_c', 'rev_min_log_c', 'rev_max_p_c', 'rev_min_p_c', 'rev_incr_yoy_c', 'rev_change_yoy_c', 'rev_max_yoy_c', 'rev_min_yoy_c', 'rev_std_yoy_c']]
y_ratio_train = season_module_valid_ratio_train['credit_ratio']
X_ratio_test = season_module_valid_ratio_test[['rev_log', 'rev_incr', 'rev_std_log', 'rev_max_log', 'rev_min_log', 'rev_max_p', 'rev_min_p', 'rev_incr_yoy', 'rev_change_yoy', 'rev_max_yoy', 'rev_min_yoy', 'rev_std_yoy', 'rev_log_c', 'rev_incr_c', 'rev_change_c', 'rev_std_log_c', 'rev_max_log_c', 'rev_min_log_c', 'rev_max_p_c', 'rev_min_p_c', 'rev_incr_yoy_c', 'rev_change_yoy_c', 'rev_max_yoy_c', 'rev_min_yoy_c', 'rev_std_yoy_c']]
y_ratio_test = season_module_valid_ratio_test['credit_ratio']
clf_ratio2 = RandomForestRegressor(n_estimators=20, random_state=0)
clf_ratio2.fit(X_ratio_train, y_ratio_train)
ratio_train_predict = pd.DataFrame(clf_ratio2.predict(X_ratio_train))
ratio_train_predict.columns = ['ratio_hat']
ratio_test_predict = pd.DataFrame(clf_ratio2.predict(X_ratio_test))
ratio_test_predict.columns = ['ratio_hat']
clf2_ratio_train_result = pd.concat([season_module_valid_ratio_train.reset_index().drop('index', 1), ratio_train_predict], axis=1)
clf2_ratio_test_result = pd.concat([season_module_valid_ratio_test.reset_index().drop('index', 1), ratio_test_predict], axis=1)
metrics.mean_squared_error(y_ratio_test, clf_ratio2.predict(X_ratio_test))
metrics.mean_squared_error(y_ratio_train, clf_ratio2.predict(X_ratio_train))
clf2_ratio_train_result.to_csv('data_output/20171201/season_module_train_ratio.csv')
clf2_ratio_test_result.to_csv('data_output/20171201/season_module_test_ratio.csv')

# 全25指标，有测试集随机森林分类
season_module_valid_rating = season_module_valid.drop('credit_ratio', 1).dropna()
season_module_valid_rating['is_train'] = np.random.uniform(0, 1, len(season_module_valid_rating)) <= 0.7
season_module_valid_rating_train = season_module_valid_rating[season_module_valid_rating['is_train'] == True]
season_module_valid_rating_test = season_module_valid_rating[season_module_valid_rating['is_train'] == False]
X_rating_train = season_module_valid_rating_train[['rev_log', 'rev_incr', 'rev_std_log', 'rev_max_log', 'rev_min_log', 'rev_max_p', 'rev_min_p', 'rev_incr_yoy', 'rev_change_yoy', 'rev_max_yoy', 'rev_min_yoy', 'rev_std_yoy', 'rev_log_c', 'rev_incr_c', 'rev_change_c', 'rev_std_log_c', 'rev_max_log_c', 'rev_min_log_c', 'rev_max_p_c', 'rev_min_p_c', 'rev_incr_yoy_c', 'rev_change_yoy_c', 'rev_max_yoy_c', 'rev_min_yoy_c', 'rev_std_yoy_c']]
y_rating_train = season_module_valid_rating_train['credit_rating']
X_rating_test = season_module_valid_rating_test[['rev_log', 'rev_incr', 'rev_std_log', 'rev_max_log', 'rev_min_log', 'rev_max_p', 'rev_min_p', 'rev_incr_yoy', 'rev_change_yoy', 'rev_max_yoy', 'rev_min_yoy', 'rev_std_yoy', 'rev_log_c', 'rev_incr_c', 'rev_change_c', 'rev_std_log_c', 'rev_max_log_c', 'rev_min_log_c', 'rev_max_p_c', 'rev_min_p_c', 'rev_incr_yoy_c', 'rev_change_yoy_c', 'rev_max_yoy_c', 'rev_min_yoy_c', 'rev_std_yoy_c']]
y_rating_test = season_module_valid_rating_test['credit_rating']
clf_rating2 = RandomForestClassifier(n_estimators=20, random_state=0)
clf_rating2.fit(X_rating_train, y_rating_train)
rating_train_predict = pd.DataFrame(clf_rating2.predict(X_rating_train))
rating_train_predict.columns = ['ratio_hat']
rating_test_predict = pd.DataFrame(clf_rating2.predict(X_rating_test))
rating_test_predict.columns = ['ratio_hat']
clf2_rating_train_result = pd.concat([season_module_valid_rating_train.reset_index().drop('index', 1), rating_train_predict], axis=1)
clf2_rating_test_result = pd.concat([season_module_valid_rating_test.reset_index().drop('index', 1), rating_test_predict], axis=1)
metrics.confusion_matrix(y_rating_test, clf_rating2.predict(X_rating_test))
metrics.confusion_matrix(y_rating_train, clf_rating2.predict(X_rating_train))
clf2_rating_train_result.to_csv('data_output/20171201/season_module_train_rating.csv')
clf2_rating_test_result.to_csv('data_output/20171201/season_module_test_rating.csv')

clf_rating2.feature_importances_


#################

dispatchlist = pd.read_clipboard(header=None)
dispatchlist.columns = ['DLID', 'document_id']
dispatchlist.to_csv('data_output/20171201/2016_dispatchlist_new.csv')
dispatchlists = pd.read_clipboard()
dispatchlists = dispatchlists[['DLID', 'iSum']]
new_dispatchlists = dispatchlists.groupby('DLID').agg({
    'iSum':np.sum
}).reset_index()
new_dispatchlists.to_csv('data_output/20171201/2016_new_dispatchlists.csv')
all_balance = pd.read_csv('data_output/zhengtai_csv/client_balance1201V0.4.csv', encoding='gbk')
all_2015 = all_balance[all_balance['year'] == 2015]
dispatchlist = dispatchlist.drop('Unnamed: 0',1 )
dispatchlist.columns = ['DLID', 'document_id']

new_balance = pd.merge(all_2015, dispatchlist, on='document_id', how='left')
check_balance = pd.merge(new_balance, new_dispatchlists, on='DLID', how='left')
check_balance.to_csv('data_output/20171201/2016_balance_check.csv')


dispatchlist = pd.read_clipboard(header=None)
dispatchlist.columns = ['DLID', 'document_id']
dispatchlist.to_csv('data_output/20171201/2015_dispatchlist_new.csv')
dispatchlists = pd.read_clipboard()
dispatchlists = dispatchlists[['DLID', 'iSum']]
new_dispatchlists = dispatchlists.groupby('DLID').agg({
    'iSum': np.sum
}).reset_index()
new_dispatchlists.to_csv('data_output/20171201/2015_new_dispatchlists.csv')
all_balance = pd.read_csv('data_output/zhengtai_csv/client_balance1201V0.4.csv', encoding='gbk')
all_2015 = all_balance[all_balance['year'] == 2015]
# dispatchlist = dispatchlist.drop('Unnamed: 0', 1)
# dispatchlist.columns = ['DLID', 'document_id']
new_balance = pd.merge(all_2015, dispatchlist, on='document_id', how='left')
check_balance = pd.merge(new_balance, new_dispatchlists, on='DLID', how='left')
check_balance.to_csv('data_output/20171201/2015_balance_check.csv')


#################################
# 计算季度应收指标
all_balance = pd.read_clipboard()
all_balance['season'] = all_balance['month'].apply(lambda x: 1 if (x == 1.) | (x == 2) | (x == 3) else 2 if
                                                    (x == 4.) | (x == 5.) | (x == 6.) else 3 if
                                                    (x == 7.) | (x == 8.) | (x == 9.) else 4 if
                                                    (x == 10.) | (x == 11.) | (x == 12.) else np.NaN)
season_receivable = all_balance[all_balance['receivable'] > 0].groupby(['year', 'season', 'client_name']).agg({
    'receivable': {'sum': np.sum, 'mean': np.mean}
})
# season_receivable = pd.read_clipboard()
season_receivable.to_csv('data_output/20171201/season_receivable.csv')

# 计算季度余额平均值
result = []
for i in np.unique(all_balance['client_name'].values):
    for year in [2014, 2015, 2016]:
        for j in [1, 2, 3, 4]:
            rec = all_balance[((all_balance['client_name'] == i) & (all_balance['year'] == year) & (all_balance['season'] == j))]
            if len(rec) == 0:
                continue
            result.append(rec.iloc[-1])
result_df = pd.DataFrame(result)
result_df['season'] = result_df['season'] + 1
df = result_df[result_df['season'] != 5.]
df.to_csv('data_output/20171201/season_start.csv')

new_season_balance = pd.concat([all_balance, df], axis=0)
new_season_balance.to_clipboard()
new_season_balance['balance'] = new_season_balance['balance'].apply(lambda x: float(x) if (x != np.NaN) else np.NaN)
season_balance = new_season_balance.groupby(['year', 'season', 'client_name']).agg({
    'balance': {'sum': np.sum, 'mean': np.mean}
})
season_balance_reset = season_balance.reset_index()
season_balance_reset.to_csv('data_output/20171201/season_balance.csv')


##################


