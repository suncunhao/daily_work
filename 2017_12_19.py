#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/19 10:12
# @Author  : sch
# @File    : 2017_12_19.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

myList = list(range(1, 50))
neighbors = filter(lambda x: x % 2 != 0, myList)
neighbors.__next__()
neighbors = list(filter(lambda x: x % 2 != 0, myList))
neighbors

from sklearn import metrics
from sklearn.linear_model import LinearRegression
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
from sklearn.preprocessing import PolynomialFeatures
self_featurizer = PolynomialFeatures(degree=1)
quadratic_featurizer = PolynomialFeatures(degree=2)
cubic_featurizer = PolynomialFeatures(degree=3)
fourth_featurizer = PolynomialFeatures(degree=4)
fifth_featurizer = PolynomialFeatures(degree=5)
sixth_featurizer = PolynomialFeatures(degree=6)
seventh_featurizer = PolynomialFeatures(degree=7)
train_result = []
test_result = []
for i,j in enumerate([self_featurizer,quadratic_featurizer, cubic_featurizer, fourth_featurizer, fifth_featurizer, sixth_featurizer, seventh_featurizer]):
    X_train_dimension = j.fit_transform(X_train)
    X_test_dimension = j.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train_dimension, y_train)
    train_MSE = metrics.mean_squared_error(y_train, regressor.predict(X_train_dimension))
    test_MSE = metrics.mean_squared_error(y_test, regressor.predict(X_test_dimension))
    train_result.append(train_MSE)
    test_result.append(test_MSE)
    print('%s维度时的训练集MSE为:' % (i+1), metrics.mean_squared_error(y_train, regressor.predict(X_train_dimension)))
    print('%s维度时的测试集MSE为:' % (i+1), metrics.mean_squared_error(y_test, regressor.predict(X_test_dimension)))

plt.plot(train_result, label='偏差')
index = np.arange(len(train_result))
plt.xticks(index, index + 1)
plt.plot(test_result, label='方差')
index = np.arange(len(test_result))
plt.xticks(index, index + 1)
plt.xlabel('多项式次数')
plt.legend()

error = []
for i in range(len(train_result)):
    error.append(train_result[i] + test_result[i])
