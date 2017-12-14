#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/7 9:30
# @Author  : sch
# @File    : 20171207.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

# 插入EG1
EG1 = pd.read_csv('data_output/20171207/EGmodule_20171206V0.2-EG1.csv', encoding='gbk')
EG = pd.read_csv('data_output/20171206/EGmodule1206V0.2.csv', encoding='gbk')

EG1 = EG1[['year', 'client_name', 'EG1']]
EG = EG.drop('Unnamed: 0', 1)

new_EG = pd.merge(EG1, EG, on=['year', 'client_name'], how='outer')
new_EG.to_csv('data_output/20171207/EGmodule1207V0.3.csv')



# 遗传算法
yhat_CS = pd.read_csv('data_output/20171207/yhat_20171207/yhat_CS.csv', encoding='gbk')
yhat_OS = pd.read_csv('data_output/20171207/yhat_20171207/yhat_OS.csv', encoding='gbk')
yhat_EG = pd.read_csv('data_output/20171207/yhat_20171207/yhat_EG.csv', encoding='gbk')

y_true = pd.read_csv('data_output/ZTmodule/OSmodule_index20171205NEW.csv', encoding='gbk')
y_true = y_true[['credit_ratio', 'year', 'client_name']]

new_temp1 = pd.merge(yhat_CS, yhat_OS, on=['year', 'client_name'], how='outer')
new_temp2 = pd.merge(new_temp1, yhat_EG, on=['year', 'client_name'], how='outer')
y_final = pd.merge(new_temp2, y_true, on=['year', 'client_name'], how='outer')
y_final = y_final.dropna()




# 遗传算法
from deap import base, creator, tools

np.random.seed(12)
X = np.random.random(30).reshape((10, 3))
y = np.random.randint(0, 2, (10,))
# X = y_final[['yhat_CS', 'yhat_OS', 'yhat_EG']].values
# y = y_final['credit_ratio'].values
print(X)
print(y)

# #### 设置遗传算法的参数，比如种群数量(population), 基因交换的概率(cross_prob), 变异系数(mutation_prob), 进化迭代次数(generation), 提前终止的参数(epsilon)，即当目标函数值变化小于一定值时，程序提前停止
population = 1000
cross_prob = 0.5
mutation_prob = 0.2
generation = 30
epsilon = 0.000000001

# threshold_max 用于限制权重最大值
# threshold_min 用于限制权重最小值
threshold_max = 0.5
threshold_min = 0.15

n = X.shape[1]
# #### 首先用deap包中的creator函数生成遗传算法的类，其中'GA'为自定的遗传算法名称，可以取其它名称，base.Fitness为生成GA类的基类，weights代表该优化问题有多少个目标函数，必须为Python tuple格式，weights=(-1,)代表有一个目标函数，并且遗传算法的目标是去最小化目标函数，而当weights=(-1,1)时，代表遗传算法有两个目标函数，第一个目标函数为最小化问题，第二个目标函数为最大化问题，weights参数的取值只用符号来判定到底是最大化还是最小化问题
creator.create('GA', base.Fitness, weights=(-1,))

# #### 生成遗传算法中个体的底层储存的数据结构，该初用的是list作为储存结构
creator.create('Individual', list, fitness=creator.GA)

# #### 生成toolbox，后面将会用toolbox来描述遗传算法
toolbox = base.Toolbox()

# #### 自己定义我们希望的种群的特征，这里种群就是我们优化问题中的参数$(w_1, w_2, w_3, w_4)$, 因为这些参数需要为正，且求和后需要为1，所以我们自己定义了下面的，生成参数的函数
# def generate_weight(num):
#     value = np.random.random(num)
#     total_sum = np.sum(value)
#     for v in value:
#         yield v / total_sum

def generate_weight(num):
    assert 1.0/num < threshold_max
    while True:
        value = np.random.random(num)
        total_sum = np.sum(value)
        if np.all(value / total_sum <= threshold_max) and np.all(value / total_sum >= threshold_min):
            break
    for v in value:
        yield v / total_sum

# 目标函数的返回值一定要用tuple格式，所以当优化问题只有一个目标函数时，函数返回值需要在后面加上'，'
# def objective_function(individual):
#     y_hat = np.matmul(X, individual)
#     return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)),
def objective_function(individual):
    if np.any(np.array(individual) > threshold_max) | np.any(np.array(individual) < threshold_min):
        return 10000,
    y_hat = np.matmul(X, individual)
    return np.average((y - y_hat) ** 2, axis=0)*10000,

# #### 定义基因互换的函数，这里由于我们需要保证种群中每个individual的和为1，且各个元素的值在0到1之前，由于包里自带的函数无法满足这个要求，所以我们需要自己定义基因互换的规则
# def cxTwoPoint(ind1, ind2):
#     size = min(len(ind1), len(ind2))
#     cxpoint1 = np.random.randint(1, size)
#     cxpoint2 = np.random.randint(1, size - 1)
#     if cxpoint2 >= cxpoint1:
#         cxpoint2 += 1
#     else:  # Swap the two cx points
#         cxpoint1, cxpoint2 = cxpoint2, cxpoint1
#
#     ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
#
#     total_ind1 = sum(ind1)
#     total_ind2 = sum(ind2)
#     for i, v in enumerate(ind1):
#         ind1[i] = v / total_ind1
#     for i, v in enumerate(ind2):
#         ind2[i] = v / total_ind2
#     return ind1, ind2

# 加了阈值限制的基因互换函数
def cxTwoPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = np.random.randint(1, size)
    cxpoint2 = np.random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    # 转换成numpy array，方便后续计算
    ind1_np = np.array(ind1)
    ind2_np = np.array(ind2)
    while True:
        # 首先计算基因互换后的函数值
        total_ind1 = sum(ind1_np)
        total_ind2 = sum(ind2_np)

        # 找出权重大于阈值的位置
        pos_ind1 = (ind1_np / total_ind1) > threshold_max
        pos_ind2 = (ind2_np / total_ind2) > threshold_max

        # 如果对于第一个个体，存在权重大于阈值的位置
        if np.any(pos_ind1):
            # 则替换对应的值为阈值
            ind1_np[pos_ind1] = threshold_max
            # 重新计算和
            total_ind1 = np.sum(ind1_np)
            # 标准化权重
            ind1_np /= total_ind1

        pos_min_ind1 = (ind1_np / total_ind1) < threshold_min
        # 如果对于第一个个体，存在权重小于阈值的位置
        if np.any(pos_min_ind1):
            # 则替换对应的值为阈值
            ind1_np[pos_min_ind1] = threshold_min
            # 重新计算和
            total_ind1 = np.sum(ind1_np)
            # 标准化权重
            ind1_np /= total_ind1

        # 如果对于第二个个体，存在权重大于阈值的位置
        if np.any(pos_ind2):
            # 则替换对应的值为阈值
            ind2_np[pos_ind2] = threshold_max
            # 重新计算和
            total_ind2 = np.sum(ind2_np)
            # 标准化权重
            ind2_np /= total_ind2

        pos_min_ind2 = (ind2_np / total_ind2) < threshold_min
        # 如果对于第二个个体，存在权重小于阈值的位置
        if np.any(pos_min_ind2):
            # 则替换对应的值为阈值
            ind2_np[pos_min_ind2] = threshold_min
            # 重新计算和
            total_ind2 = np.sum(ind2_np)
            # 标准化权重
            ind2_np /= total_ind2

        # 如果都没有权重大于阈值和小于阈值的位置，则退出循环，否则直到替换个体满足条件为止
        if ~np.any(pos_ind1) and ~np.any(pos_ind2) and ~np.any(pos_min_ind1) and ~np.any(pos_min_ind2):
            break
        # print('#' * 10)
        # print(ind1_np)
        # print(ind2_np)
    #     替换原始个体对应位置的值
    for i, v in enumerate(ind1_np):
        ind1[i] = v / total_ind1
    for i, v in enumerate(ind2_np):
        ind2[i] = v / total_ind2
    return ind1, ind2

# #### 定义个体individual的初始化函数，并且生成对应的种群
toolbox.register('generate_weight', generate_weight, num=n)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.generate_weight)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# #### 定义目标函数
toolbox.register('evaluate', objective_function)
# #### 定义个体基因互换(mate)的函数, 个体基因变异(mutate)的函数，还有个体选择的函数(select), 其中变异和选择函数直接用的tools里自带的函数
toolbox.register('mate', cxTwoPoint)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.1)
toolbox.register('select', tools.selTournament, tournsize=3)

# #### 遗传算法实现的详细过程
# 初始化生成所有种群
pop = toolbox.population(population)

# 用目标函数评估生成的种群中每一个个体的取值
fitnesses = list(map(toolbox.evaluate, pop))

# 将目标函数在每个个体上面的值赋予每个个体
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# 记录现在最小的目标函数的值(因为我们是最小化问题)
previous_obj = None

# 循环迭代预设的次数
for g in range(generation):
    print("-- Generation %i --" % g)

    # 选取需要迭代的种群
    offspring = toolbox.select(pop, len(pop))
    # 克隆现有的种群用于计算
    offspring = list(map(toolbox.clone, offspring))

    # 对于个体之间，两两进行是发生基因互换的判断
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # 如果随机数小于发生基因互换的预设概率，则发生互换
        if np.random.random() < cross_prob:
            toolbox.mate(child1, child2)

            # 删除对应个体的目标函数取值，因为他们是新生成的个体，后面会对新生成的个体进行重新目标函数值的计算
            del child1.fitness.values
            del child2.fitness.values

    # 对种群中每个个体进行遍历，判断是否需要进行变异
    for mutant in offspring:

        # 如果随机数小于发生变异的概率，则发生变异
        if np.random.random() < mutation_prob:
            toolbox.mutate(mutant)

            # 删除对应个体的目标函数取值，因为他们是新生成的个体，后面会对新生成的个体进行重新目标函数值的计算
            del mutant.fitness.values

    # 选出所有新生成的个体
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    # 计算对应新生成个体的目标函数的值
    fitnesses = map(toolbox.evaluate, invalid_ind)
    # 将计算后得到的目标函数的值赋予新生成的个体
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 重新更新现在所有的种群
    pop[:] = offspring

    # 得到所有个体的目标函数值
    fits = [ind.fitness.values[0] for ind in pop]

    # 记录里面最小的数值，因为我们这里是最小化问题
    current_obj = min(fits)
    # 如果是第一次计算，则直接记录数值
    if previous_obj is None:
        previous_obj = current_obj
    # 如果目标函数值没有对应的提升并且迭代次数大于预设的迭代次数的一半，则停止
    elif g > 0.5 * generation and abs(previous_obj - current_obj) < epsilon:
        break
    # 否则更新目标函数值，并准备后面的继续迭代
    else:
        previous_obj = current_obj

    # 输出一些统计量
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

# 输出最后的结果
print("-- End of (successful) evolution --")
best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

y_final['y_hat'] = y_final['yhat_CS']*best_ind[0] + y_final['yhat_OS']*best_ind[1] + y_final['yhat_EG']*best_ind[2]

# 日期插值
new_date = pd.date_range('2014-1-1', '2014-12-31')
date_df = pd.concat([pd.DataFrame(new_date), pd.DataFrame(new_date.year), pd.DataFrame(new_date.month), pd.DataFrame(new_date.day)], axis=1)
date_df.columns = ['date', 'year', 'month', 'day']
balance = pd.read_csv('data_output/data_for_model/client_balance.csv', encoding='gbk')
balance_2014 = balance[balance['year'] == 2014]


result = {}
for i in np.unique(list(balance['client_name'])):
    result[i] = pd.merge(date_df, balance_2014[balance_2014['client_name'] == i], on=['month', 'day'], how=['left'])















# FIN指标
