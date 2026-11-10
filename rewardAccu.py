#[4, 6, 8, 76, 77, 82, 84, 88, 110, 125, 136, 141, 166, 168, 174, 184, 185, 192, 197, 199]
import numpy as np
import utils
from sourceLocalizationClass import SingleSourceLocalization
import math
import matplotlib.pyplot as plt
import random
import time


test_data = utils.data_loader(**{
            'data_path': 'data/train_data/',  # 目前训练数据和测试数据是同一个路径？？后续需要分开
            'graph_type': 'BA',  # 图类型
            'n_nodes': 200,  # 节点数量
            'num_graphs': 100,  # 每个规模下生成图的数量
            'no_sd': True  # 是否取消连边延迟的标准差
        })
# 抽取0号图
G, G_matrix_diag, G_distance, G_distance_delay = test_data['G'][0], test_data['G_matrix_diag'][0], \
                        test_data['G_distance'][0], test_data['G_distance_delay'][0]
SL_Model = SingleSourceLocalization(G, G_matrix_diag)
# place_obs = [4, 6, 8, 76, 77, 82, 84, 88, 110, 125, 136, 141, 166, 168, 174, 184, 185, 192, 197, 199]
# place_obs = [4, 6, 8, 76, 77, 82, 84, 88, 110, 125]
place_obs = random.sample(range(len(G)), 10)
reward = {'Precision': [], 'CSSa': [], 'AveRank': []}

source_actual_lst = SL_Model.sample_source_lst(n_source=200, sample_method='order')
# print(source_actual_lst)

tic = time.time()

for i in range(1, len(place_obs)):
    place_obs_lst = place_obs[0:i+1]
    # print('place_obs_lst', place_obs_lst)
    #计算推断源可能性排序
    rank_lst_nodes = []
    rank_values = []
    for source_actual in source_actual_lst:
        inform = SL_Model.get_inform(source_actual, place_obs_lst, G_distance_delay, spread_method='snowball')
        source_actual_rank, source_actual_rank_values = SL_Model.locate_source(source_actual, inform, G_distance, locate_strategy='Rank_List')
        rank_lst_nodes.append(source_actual_rank)
        rank_values.append(source_actual_rank_values)   
    rank_lst_nodes = np.array(rank_lst_nodes)
    rank_values = np.array(rank_values)
    # print('rank_lst_nodes:', rank_lst_nodes)
    # print('rank_values:', rank_values)

    small_equal = []
    smaller_count = 0
    equal_count = 0
    # 计算每行 0 的个数
    count_zeros_per_row = (rank_values == 0).sum(axis=1)
    # print(count_zeros_per_row)

    # 计算precision
    precision = 1/count_zeros_per_row
    # print('Precision:', precision)
    reward['Precision'].append(np.average(precision))

    # 计算CSSa, a=0.95
    cssa = np.percentile(count_zeros_per_row, 95)
    sorted_data = np.sort(count_zeros_per_row)
    cssa = sorted_data[math.ceil(200*0.95)-1] # -1是因为是从0开始计数的
    reward['CSSa'].append(cssa)

    # 计算AveRank
    ave_rank = count_zeros_per_row/2
    reward['AveRank'].append(np.average(ave_rank))

print(reward)
# 归一化函数
def min_max_normalize(lst):
    lst = np.array(lst)
    min_val = np.min(lst)
    max_val = np.max(lst)
    if max_val == min_val:  # 防止除以零
        return np.zeros_like(lst)
    return (lst - min_val) / (max_val - min_val)

# 对 reward 中的每个指标进行归一化
reward = {
    'Precision': min_max_normalize(reward['Precision']).tolist(),
    'CSSa': min_max_normalize(reward['CSSa']).tolist(),
    'AveRank': min_max_normalize(reward['AveRank']).tolist()
}

print("归一化后的 reward：")
print(reward)
toc = time.time()
print("运行时间：", toc - tic)

# 生成 x 轴数据（传感器数量变化）
x = range(1, len(reward['Precision']) + 1)

# 创建图表
plt.figure(figsize=(12, 6))

# Precision
plt.subplot(1, 3, 1)
plt.plot(x, reward['Precision'], marker='o', color='b')
plt.title('Precision')
plt.xlabel('Number of Observers')
plt.ylabel('Value')

# CSSa
plt.subplot(1, 3, 2)
plt.plot(x, reward['CSSa'], marker='o', color='g')
plt.title('CSSa (0.95-Percentile)')
plt.xlabel('Number of Observers')
plt.ylabel('Value')

# AveRank
plt.subplot(1, 3, 3)
plt.plot(x, reward['AveRank'], marker='o', color='r')
plt.title('AveRank')
plt.xlabel('Number of Observers')
plt.ylabel('Value')

# 自动调整子图间距
plt.tight_layout()
plt.show()


