#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/27 9:34
# @Author  : sch
# @File    : 2017_11_27.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

sys.path.append('univariate')
from univariate.analysis import UnivariateAnalysis

CSmodule = pd.read_csv('data_output/ZTmodule/CSmudule_index20171127V0.3.csv')
HImodule = pd.read_csv('data_output/ZTmodule/HImudule_index20171124V0.2.csv')
OSmodule = pd.read_csv('data_output/ZTmodule/OSmudule_index20171127V0.3.csv')
CSdata = CSmodule.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
HIdata = HImodule.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
OSdata = OSmodule.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
obj1 = UnivariateAnalysis(CSdata)
obj2 = UnivariateAnalysis(HIdata)
obj3 = UnivariateAnalysis(OSdata)
obj1.get_variable_description()
obj2.get_variable_description()
obj3.get_variable_description()

a = CSdata.isnull()
b = HIdata.isnull()
c = OSdata.isnull()
# a['AAA'] = np.all(a, axis=1)
a['AA'] = np.sum(a, axis=1)
b['BB'] = np.sum(b, axis=1)
c['CC'] = np.sum(c, axis=1)
aa = []
bb = []
cc = []
for i in range(0, len(CSdata.columns) + 1):
    aa.append(np.sum(a['AA'] == i))
for i in range(0, len(HIdata.columns) + 1):
    bb.append(np.sum(b['BB'] == i))
for i in range(0, len(OSdata.columns) + 1):
    cc.append(np.sum(c['CC'] == i))


CSmodule2015 = CSmodule[CSmodule['year'] == 2015]
CSdata2015 = CSmodule2015.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
obj4 = UnivariateAnalysis(CSdata2015.T)
obj4.get_variable_description()
CSdata2015ana = obj4.get_variable_description()[['NaN', 'Unknown']].groupby(['NaN']).count().reset_index()
countlist = pd.DataFrame(np.arange(0, len(CSdata2015.columns) + 1))
countlist.columns = ['NaN']
CSdata2015ana_new = pd.merge(countlist, CSdata2015ana, on='NaN', how='left')
CSdata2015ana_new

HImodule2015 = HImodule[HImodule['year'] == 2015]
HIdata2015 = HImodule2015.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
obj1 = UnivariateAnalysis(HIdata2015)
obj1.get_variable_description()
obj5 = UnivariateAnalysis(HIdata2015.T)
HIdata2015ana = obj5.get_variable_description()[['NaN', 'Unknown']].groupby(['NaN']).count().reset_index()
countlist = pd.DataFrame(np.arange(0, len(HIdata2015.columns) + 1))
countlist.columns = ['NaN']
HIdata2015ana_new = pd.merge(countlist, HIdata2015ana, on='NaN', how='left')
HIdata2015ana_new = HIdata2015ana_new.fillna(0)
HIdata2015ana_new.to_clipboard()

OSmodule2015 = OSmodule[OSmodule['year'] == 2015]
OSdata2015 = OSmodule2015.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
obj1 = UnivariateAnalysis(OSdata2015)
obj1.get_variable_description()
obj5 = UnivariateAnalysis(OSdata2015.T)
OSdata2015ana = obj5.get_variable_description()[['NaN', 'Unknown']].groupby(['NaN']).count().reset_index()
countlist = pd.DataFrame(np.arange(0, len(OSdata2015.columns) + 1))
countlist.columns = ['NaN']
OSdata2015ana_new = pd.merge(countlist, OSdata2015ana, on='NaN', how='left')
OSdata2015ana_new = OSdata2015ana_new.fillna(0)
OSdata2015ana_new.to_clipboard()

#

CSmodule2015 = CSmodule[CSmodule['year'] == 2016]
CSdata2015 = CSmodule2015.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
obj1 = UnivariateAnalysis(CSdata2015)
obj1.get_variable_description().to_clipboard()
obj5 = UnivariateAnalysis(CSdata2015.T)
CSdata2015ana = obj5.get_variable_description()[['NaN', 'Unknown']].groupby(['NaN']).count().reset_index()
countlist = pd.DataFrame(np.arange(0, len(CSdata2015.columns) + 1))
countlist.columns = ['NaN']
CSdata2015ana_new = pd.merge(countlist, CSdata2015ana, on='NaN', how='left')
CSdata2015ana_new = CSdata2015ana_new.fillna(0)
CSdata2015ana_new.to_clipboard()

HImodule2015 = HImodule[HImodule['year'] == 2016]
HIdata2015 = HImodule2015.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
obj1 = UnivariateAnalysis(HIdata2015)
obj1.get_variable_description().to_clipboard()
obj5 = UnivariateAnalysis(HIdata2015.T)
HIdata2015ana = obj5.get_variable_description()[['NaN', 'Unknown']].groupby(['NaN']).count().reset_index()
countlist = pd.DataFrame(np.arange(0, len(HIdata2015.columns) + 1))
countlist.columns = ['NaN']
HIdata2015ana_new = pd.merge(countlist, HIdata2015ana, on='NaN', how='left')
HIdata2015ana_new = HIdata2015ana_new.fillna(0)
HIdata2015ana_new.to_clipboard()

OSmodule2015 = OSmodule[OSmodule['year'] == 2016]
OSdata2015 = OSmodule2015.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
obj1 = UnivariateAnalysis(OSdata2015)
obj1.get_variable_description().to_clipboard()
obj5 = UnivariateAnalysis(OSdata2015.T)
OSdata2015ana = obj5.get_variable_description()[['NaN', 'Unknown']].groupby(['NaN']).count().reset_index()
countlist = pd.DataFrame(np.arange(0, len(OSdata2015.columns) + 1))
countlist.columns = ['NaN']
OSdata2015ana_new = pd.merge(countlist, OSdata2015ana, on='NaN', how='left')
OSdata2015ana_new = OSdata2015ana_new.fillna(0)
OSdata2015ana_new.to_clipboard()

#

CSmodule2015 = CSmodule[CSmodule['year'] == 2015].append(CSmodule[CSmodule['year'] == 2016])
CSdata2015 = CSmodule2015.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
obj1 = UnivariateAnalysis(CSdata2015)
obj1.get_variable_description().to_clipboard()
obj5 = UnivariateAnalysis(CSdata2015.T)
CSdata2015ana = obj5.get_variable_description()[['NaN', 'Unknown']].groupby(['NaN']).count().reset_index()
countlist = pd.DataFrame(np.arange(0, len(CSdata2015.columns) + 1))
countlist.columns = ['NaN']
CSdata2015ana_new = pd.merge(countlist, CSdata2015ana, on='NaN', how='left')
CSdata2015ana_new = CSdata2015ana_new.fillna(0)
CSdata2015ana_new.to_clipboard()

HImodule2015 = HImodule[HImodule['year'] == 2015].append(HImodule[HImodule['year'] == 2016])
HIdata2015 = HImodule2015.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
obj1 = UnivariateAnalysis(HIdata2015)
obj1.get_variable_description().to_clipboard()
obj5 = UnivariateAnalysis(HIdata2015.T)
HIdata2015ana = obj5.get_variable_description()[['NaN', 'Unknown']].groupby(['NaN']).count().reset_index()
countlist = pd.DataFrame(np.arange(0, len(HIdata2015.columns) + 1))
countlist.columns = ['NaN']
HIdata2015ana_new = pd.merge(countlist, HIdata2015ana, on='NaN', how='left')
HIdata2015ana_new = HIdata2015ana_new.fillna(0)
HIdata2015ana_new.to_clipboard()

OSmodule2015 = OSmodule[OSmodule['year'] == 2015].append(OSmodule[OSmodule['year'] == 2016])
OSdata2015 = OSmodule2015.drop(['Unnamed: 0', 'credit_rating', 'client_name', 'year'], 1)
obj1 = UnivariateAnalysis(OSdata2015)
obj1.get_variable_description().to_clipboard()
obj5 = UnivariateAnalysis(OSdata2015.T)
OSdata2015ana = obj5.get_variable_description()[['NaN', 'Unknown']].groupby(['NaN']).count().reset_index()
countlist = pd.DataFrame(np.arange(0, len(OSdata2015.columns) + 1))
countlist.columns = ['NaN']
OSdata2015ana_new = pd.merge(countlist, OSdata2015ana, on='NaN', how='left')
OSdata2015ana_new = OSdata2015ana_new.fillna(0)
OSdata2015ana_new.to_clipboard()




#
CSmodule2015 = CSmodule2015.drop(['CS1', 'CS8'], 1)
CSmodule2015 = CSmodule2015.dropna()
X = CSmodule2015[['CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'CS9']]
y = CSmodule2015['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)

from sklearn import metrics, linear_model
import mord
print('module:CS')
clf1 = linear_model.LogisticRegression()
clf1.fit(X, y)
clf1.predict(X)
print('Mean Absolute Error of LogisticRegression: %s' %
      metrics.mean_absolute_error(clf1.predict(X), y))
clf2 = mord.LogisticAT(alpha=1.)
clf2.fit(X, y)
clf2.predict(X)
print('Mean Absolute Error of LogisticAT %s' %
      metrics.mean_absolute_error(clf2.predict(X), y))
clf3 = mord.LogisticIT(alpha=1.)
clf3.fit(X, y)
print('Mean Absolute Error of LogisticIT %s' %
      metrics.mean_absolute_error(clf3.predict(X), y))
clf4 = mord.LogisticSE(alpha=1.)
clf4.fit(X, y)
print('Mean Absolute Error of LogisticSE %s' %
      metrics.mean_absolute_error(clf4.predict(X), y))


# HImodule2015 = HImodule2015.drop('HI3', 1)
HImodule2015 = HImodule2015.dropna()
X = HImodule2015[['HI1', 'HI2', 'HI4', 'HI5', 'HI6']]
y = HImodule2015['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
print('module:HI')
clf1 = linear_model.LogisticRegression()
clf1.fit(X, y)
clf1.predict(X)
print('Mean Absolute Error of LogisticRegression: %s' %
      metrics.mean_absolute_error(clf1.predict(X), y))
clf2 = mord.LogisticAT(alpha=1.)
clf2.fit(X, y)
clf2.predict(X)
print('Mean Absolute Error of LogisticAT %s' %
      metrics.mean_absolute_error(clf2.predict(X), y))
clf3 = mord.LogisticIT(alpha=1.)
clf3.fit(X, y)
print('Mean Absolute Error of LogisticIT %s' %
      metrics.mean_absolute_error(clf3.predict(X), y))
clf4 = mord.LogisticSE(alpha=1.)
clf4.fit(X, y)
print('Mean Absolute Error of LogisticSE %s' %
      metrics.mean_absolute_error(clf4.predict(X), y))


# OSmodule2015 = OSmodule2015.drop('OS5', 1)
OSmodule2015 = OSmodule2015.dropna()
X = OSmodule2015[['OS1', 'OS2', 'OS4', 'OS6', 'OS7', 'OS8']]
y = OSmodule2015['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
print('module:OS')
clf1 = linear_model.LogisticRegression()
clf1.fit(X, y)
clf1.predict(X)
print('Mean Absolute Error of LogisticRegression: %s' %
      metrics.mean_absolute_error(clf1.predict(X), y))
clf2 = mord.LogisticAT(alpha=1.)
clf2.fit(X, y)
clf2.predict(X)
print('Mean Absolute Error of LogisticAT %s' %
      metrics.mean_absolute_error(clf2.predict(X), y))
clf3 = mord.LogisticIT(alpha=1.)
clf3.fit(X, y)
print('Mean Absolute Error of LogisticIT %s' %
      metrics.mean_absolute_error(clf3.predict(X), y))
clf4 = mord.LogisticSE(alpha=1.)
clf4.fit(X, y)
print('Mean Absolute Error of LogisticSE %s' %
      metrics.mean_absolute_error(clf4.predict(X), y))



# CSmodule2015
CSmodule = pd.read_csv('data_output/ZTmodule/CSmudule_index20171127V0.3.csv')
CSmodule2015 = CSmodule[CSmodule['year'] == 2015].append(CSmodule[CSmodule['year'] == 2016])
CSmodule2015 = CSmodule2015.drop(['CS1', 'CS8'], 1)
CSmodule2015 = CSmodule2015.dropna()
X = CSmodule2015[['CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'CS9']]
y = CSmodule2015['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
CSname = CSmodule2015[['credit_rating', 'client_name']]
clf1 = linear_model.LogisticRegression()
clf1.fit(X, y)
y_predict = clf1.predict(X)
y_predict = np.eye(3)[y_predict]
y_predict = pd.DataFrame(y_predict)
y_predict.columns = ['A_hat', 'B_hat', 'C_hat']
y_prob = clf1.predict_proba(X)
y_prob = pd.DataFrame(y_prob)
y_prob.columns = ['Pr(A)', 'Pr(B)', 'Pr(C)']
y = np.eye(3)[y]
y = pd.DataFrame(y)
y.columns = ['A', 'B', 'C']

CSname = CSname.reset_index().drop('index', 1)
CSname
new_CS = pd.concat([CSname, y_prob, y_predict, y], 1)
new_CS['Score(A)'] = new_CS['Pr(A)'] * 100
new_CS['Score(B)'] = new_CS['Pr(B)'] * 100
new_CS['Score(C)'] = new_CS['Pr(C)'] * 100
new_CS.to_csv('data_output/20171127/CSlogit.csv')


HImodule = pd.read_csv('data_output/ZTmodule/HImudule_index20171124V0.2.csv')
HImodule2015 = HImodule[HImodule['year'] == 2015].append(HImodule[HImodule['year'] == 2016])
HImodule2015 = HImodule2015.drop('HI3', 1)
HImodule2015 = HImodule2015.dropna()

X = HImodule2015[['HI1', 'HI2', 'HI4', 'HI5', 'HI6']]
y = HImodule2015['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
HIname = HImodule2015[['credit_rating', 'client_name']]
clf1 = linear_model.LogisticRegression()
clf1.fit(X, y)
y_predict = clf1.predict(X)
y_predict = np.eye(3)[y_predict]
y_predict = pd.DataFrame(y_predict)
y_predict.columns = ['A_hat', 'B_hat', 'C_hat']
y_prob = clf1.predict_proba(X)
y_prob = pd.DataFrame(y_prob)
y_prob.columns = ['Pr(A)', 'Pr(B)', 'Pr(C)']
y = np.eye(3)[y]
y = pd.DataFrame(y)
y.columns = ['A', 'B', 'C']

HIname = HIname.reset_index().drop('index', 1)
HIname
new_HI = pd.concat([HIname, y_prob, y_predict, y], 1)
new_HI['Score(A)'] = new_HI['Pr(A)'] * 100
new_HI['Score(B)'] = new_HI['Pr(B)'] * 100
new_HI['Score(C)'] = new_HI['Pr(C)'] * 100
new_HI.to_csv('data_output/20171127/HIlogit.csv')



OSmodule = pd.read_csv('data_output/ZTmodule/OSmudule_index20171127V0.3.csv')
OSmodule2015 = OSmodule[OSmodule['year'] == 2015].append(OSmodule[OSmodule['year'] == 2016])
OSmodule2015 = OSmodule2015.drop('OS5', 1)
OSmodule2015 = OSmodule2015.dropna()

X = OSmodule2015[['OS1', 'OS2', 'OS4', 'OS6', 'OS7', 'OS8']]
y = OSmodule2015['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
OSname = OSmodule2015[['credit_rating', 'client_name']]
clf1 = linear_model.LogisticRegression()
clf1.fit(X, y)
y_predict = clf1.predict(X)
y_predict = np.eye(3)[y_predict]
y_predict = pd.DataFrame(y_predict)
y_predict.columns = ['A_hat', 'B_hat', 'C_hat']
y_prob = clf1.predict_proba(X)
y_prob = pd.DataFrame(y_prob)
y_prob.columns = ['Pr(A)', 'Pr(B)', 'Pr(C)']
y = np.eye(3)[y]
y = pd.DataFrame(y)
y.columns = ['A', 'B', 'C']

OSname = OSname.reset_index().drop('index', 1)
OSname
new_OS = pd.concat([OSname, y_prob, y_predict, y], 1)
new_OS['Score(A)'] = new_HI['Pr(A)'] * 100
new_OS['Score(B)'] = new_HI['Pr(B)'] * 100
new_OS['Score(C)'] = new_HI['Pr(C)'] * 100
new_OS.to_csv('data_output/20171127/OSlogit.csv')

import seaborn as sns

graph = sns.FacetGrid(data=new_CS, hue='A')
graph.map(sns.distplot, 'Score(A)', kde=False)
plt.legend()
plt.title('CS-A')
graph = sns.FacetGrid(data=new_CS, hue='A_hat')
graph.map(sns.distplot, 'Score(A)', kde=False)
plt.legend()
plt.title('CS-A_hat')




CSmodule = pd.read_csv('data_output/ZTmodule/CSmudule_index20171127V0.3.csv')
CSmodule2015 = CSmodule[CSmodule['year'] == 2015].append(CSmodule[CSmodule['year'] == 2016])
CSmodule2015 = CSmodule2015.drop(['CS1', 'CS8'], 1)
CSmodule2015 = CSmodule2015.dropna()
X = CSmodule2015[['CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'CS9']]
y = CSmodule2015['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
clf1 = linear_model.LogisticRegression()
clf2 = mord.LogisticAT(alpha=1.)
clf3 = mord.LogisticIT(alpha=1.)
clf4 = mord.LogisticSE(alpha=1.)
clf1.fit(X, y)
LR = pd.DataFrame(clf1.predict(X))
LR.columns = ['LR']
clf2.fit(X, y)
AT = pd.DataFrame(clf2.predict(X))
AT.columns = ['AT']
clf3.fit(X, y)
IT = pd.DataFrame(clf3.predict(X))
IT.columns = ['IT']
clf4.fit(X, y)
SE = pd.DataFrame(clf4.predict(X))
SE.columns = ['SE']
y = pd.DataFrame(y).reset_index().drop('index', 1)
CSname = CSmodule2015[['credit_rating', 'client_name']]
CSname = CSname.reset_index()
CSname = CSname.drop('index', 1)
ordinal_CS = pd.concat([CSname, LR, AT, IT, SE, y], 1)


HImodule = pd.read_csv('data_output/ZTmodule/HImudule_index20171124V0.2.csv')
HImodule2015 = HImodule[HImodule['year'] == 2015].append(HImodule[HImodule['year'] == 2016])
HImodule2015 = HImodule2015.drop('HI3', 1)
HImodule2015 = HImodule2015.dropna()
X = HImodule2015[['HI1', 'HI2', 'HI4', 'HI5', 'HI6']]
y = HImodule2015['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
clf1 = linear_model.LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial')
clf2 = mord.LogisticAT(alpha=1.)
clf3 = mord.LogisticIT(alpha=1.)
clf4 = mord.LogisticSE(alpha=1.)
clf1.fit(X, y)
LR = pd.DataFrame(clf1.predict(X))
LR.columns = ['LR']
clf2.fit(X, y)
AT = pd.DataFrame(clf2.predict(X))
AT.columns = ['AT']
clf3.fit(X, y)
IT = pd.DataFrame(clf3.predict(X))
IT.columns = ['IT']
clf4.fit(X, y)
SE = pd.DataFrame(clf4.predict(X))
SE.columns = ['SE']
y = pd.DataFrame(y).reset_index().drop('index', 1)
HIname = HImodule2015[['credit_rating', 'client_name']]
HIname = HIname.reset_index()
HIname = HIname.drop('index', 1)
ordinal_HI = pd.concat([HIname, LR, AT, IT, SE, y], 1)
ordinal_HI.to_csv('data_output/20171127/ordinal_HI.csv')


OSmodule = pd.read_csv('data_output/ZTmodule/OSmudule_index20171127V0.3.csv')
OSmodule2015 = OSmodule[OSmodule['year'] == 2015].append(OSmodule[OSmodule['year'] == 2016])
OSmodule2015 = OSmodule2015.drop('OS5', 1)
OSmodule2015 = OSmodule2015.dropna()
X = OSmodule2015[['OS1', 'OS2', 'OS4', 'OS6', 'OS7', 'OS8']]
y = OSmodule2015['credit_rating'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2 if x == 'C' else 2 if x == 'D' else 2)
clf1 = linear_model.LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial')
clf2 = mord.LogisticAT(alpha=1.)
clf3 = mord.LogisticIT(alpha=1.)
clf4 = mord.LogisticSE(alpha=1.)
clf1.fit(X, y)
LR = pd.DataFrame(clf1.predict(X))
LR.columns = ['LR']
clf2.fit(X, y)
AT = pd.DataFrame(clf2.predict(X))
AT.columns = ['AT']
clf3.fit(X, y)
IT = pd.DataFrame(clf3.predict(X))
IT.columns = ['IT']
clf4.fit(X, y)
SE = pd.DataFrame(clf4.predict(X))
SE.columns = ['SE']
y = pd.DataFrame(y).reset_index().drop('index', 1)
OSname = OSmodule2015[['credit_rating', 'client_name']]
OSname = OSname.reset_index()
OSname = OSname.drop('index', 1)
ordinal_OS = pd.concat([OSname, LR, AT, IT, SE, y], 1)
ordinal_OS.to_csv('data_output/20171127/ordinal_OS.csv')

#################################

graph = sns.FacetGrid(data=new_CS, hue='A')
graph.map(sns.distplot, 'Score(A)', kde=False)
plt.legend()
plt.title('CS-A')
graph = sns.FacetGrid(data=new_CS, hue='A_hat')
graph.map(sns.distplot, 'Score(A)', kde=False)
plt.legend()
plt.title('CS-A_hat')
graph = sns.FacetGrid(data=new_CS, hue='B')
graph.map(sns.distplot, 'Score(B)', kde=False)
plt.legend()
plt.title('CS-B')
graph = sns.FacetGrid(data=new_CS, hue='B_hat')
graph.map(sns.distplot, 'Score(B)', kde=False)
plt.legend()
plt.title('CS-B_hat')
graph = sns.FacetGrid(data=new_CS, hue='C')
graph.map(sns.distplot, 'Score(C)', kde=False)
plt.legend()
plt.title('CS-C')
graph = sns.FacetGrid(data=new_CS, hue='C_hat')
graph.map(sns.distplot, 'Score(C)', kde=False)
plt.legend()
plt.title('CS-C_hat')

graph = sns.FacetGrid(data=new_HI, hue='A')
graph.map(sns.distplot, 'Score(A)', kde=False)
plt.legend()
plt.title('HI-A')
graph = sns.FacetGrid(data=new_HI, hue='A_hat')
graph.map(sns.distplot, 'Score(A)', kde=False)
plt.legend()
plt.title('HI-A_hat')
graph = sns.FacetGrid(data=new_HI, hue='B')
graph.map(sns.distplot, 'Score(B)', kde=False)
plt.legend()
plt.title('HI-B')
graph = sns.FacetGrid(data=new_HI, hue='B_hat')
graph.map(sns.distplot, 'Score(B)', kde=False)
plt.legend()
plt.title('HI-B_hat')
graph = sns.FacetGrid(data=new_HI, hue='C')
graph.map(sns.distplot, 'Score(C)', kde=False)
plt.legend()
plt.title('HI-C')
graph = sns.FacetGrid(data=new_HI, hue='C_hat')
graph.map(sns.distplot, 'Score(C)', kde=False)
plt.legend()
plt.title('HI-C_hat')

graph = sns.FacetGrid(data=new_OS, hue='A')
graph.map(sns.distplot, 'Score(A)', kde=False)
plt.legend()
plt.title('OS-A')
graph = sns.FacetGrid(data=new_OS, hue='A_hat')
graph.map(sns.distplot, 'Score(A)', kde=False)
plt.legend()
plt.title('OS-A_hat')
graph = sns.FacetGrid(data=new_OS, hue='B')
graph.map(sns.distplot, 'Score(B)', kde=False)
plt.legend()
plt.title('OS-B')
graph = sns.FacetGrid(data=new_OS, hue='B_hat')
graph.map(sns.distplot, 'Score(B)', kde=False)
plt.legend()
plt.title('OS-B_hat')
graph = sns.FacetGrid(data=new_OS, hue='C')
graph.map(sns.distplot, 'Score(C)', kde=False)
plt.legend()
plt.title('OS-C')
graph = sns.FacetGrid(data=new_OS, hue='C_hat')
graph.map(sns.distplot, 'Score(C)', kde=False)
plt.legend()
plt.title('OS-C_hat')

