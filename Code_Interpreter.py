import pandas as pd
import numpy as np
import featuretools as ft
from deap import base, creator, tools, algorithms
from scipy.stats import spearmanr
import random

# 加载数据
df = pd.read_excel('make_factors_data.xlsx')

# 创建新的实体集
es = ft.EntitySet(id = 'features')

# 将整个数据框添加为一个实体
es = es.add_dataframe(dataframe_name='data', dataframe=df, 
                      index='index')

# 执行深度特征合成
features, feature_defs = ft.dfs(entityset = es, target_dataframe_name = 'data',
                                agg_primitives = ['mean', 'max', 'min', 'std', 'skew','percent_true','mode'],
                                trans_primitives = ['add_numeric', 'multiply_numeric'],max_depth = 5)

print('Total number of new features:', len(features.columns))

# 定义计算信息系数（IC）的函数
def calculate_ic(features, target):
    return spearmanr(features, target)[0]

# 定义适应度函数
def fitness(individual):
    selected_features = [feature for feature, used in zip(features.columns, individual) if used]
    if len(selected_features) == 0:
        return 0,
    if features[selected_features].shape[0] != df['收盘'].shape[0]:
        print(f"Shapes not matching: {features[selected_features].shape[0]} vs {df['收盘'].shape[0]}")
        return 0,
    # 计算选中的特征的信息系数（IC）
    ic = calculate_ic(features[selected_features], df['收盘'])
    return ic,

# 创建工具箱
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 特征生成器
toolbox.register("attr_bool", random.randint, 0, 1)

# 结构初始化器
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(features.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册操作
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 进化种群
# Evolving the population
NGEN = 40
population = toolbox.population(n=50)
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # Select the next generation individuals
    population = toolbox.select(offspring, k=len(population))


# 计算最优个体的特征和IC值
best_ind = tools.selBest(population, 1)[0]
best_features = [feature for feature, used in zip(features.columns, best_ind) if used]
best_ic = calculate_ic(features[best_features], df['收盘'])

# 将结果保存为CSV文件
result = pd.DataFrame({'Feature': best_features, 'IC': best_ic})
result.to_csv('result.csv', index=False)
