#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/7 16:49
# @Author  : sch
# @File    : 2017_11_7.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets

data = pd.read_clipboard()

kind = '经销商'
group_list = np.unique(data[kind].dropna().values)
for group in group_list:
    new_data = data[data[kind] == group]
    new_data.to_csv('data_output/jingxiao2015/%s.csv' % group)

