# -*- coding: utf-8 -*-
"""
本项目中用到的工具函数
"""

from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import networkx as nx
import os
import re
# import pickle
from scipy.stats import truncnorm
from functools import reduce
import copy
import numpy as np
import random
# import math
import numpy as np
import torch
import pandas as pd
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from graph_indexing import (
    canonical_nodelist,
    validate_adjacency_hop_alignment,
    validate_delay_distance,
)

# from torch.nn import Parameter
# from layers import GraphConvolution, GraphAttention

import numpy as np

GRAPH_FEATURE_GROUPS = {
    'centrality_core': [
        'degree_norm',
        'betweenness_norm',
        'closeness_norm',
        'k_core_norm',
    ],
    'local_structure': [
        'clustering_norm',
        'two_hop_ratio',
        'eccentricity_norm',
    ],
    'global_context': [
        'graph_density',
        'diameter_ratio',
    ],
    'constant_bias': [
        'bias',
    ],
}

GRAPH_FEATURE_NAMES = [
    feature_name
    for feature_group in GRAPH_FEATURE_GROUPS.values()
    for feature_name in feature_group
]

# Cumulative feature sets used by the feature-ablation experiments.  F3 is
# deliberately the default so commands and checkpoints created by the current
# experiment framework keep their existing 12-dimensional PPO input.
GRAPH_FEATURE_SETS = {
    'f0': [
        'degree_norm',
        'betweenness_norm',
        'closeness_norm',
        'bias',
    ],
    'f1': [
        'degree_norm',
        'betweenness_norm',
        'closeness_norm',
        'k_core_norm',
        'bias',
    ],
    'f2': [
        'degree_norm',
        'betweenness_norm',
        'closeness_norm',
        'k_core_norm',
        'clustering_norm',
        'two_hop_ratio',
        'eccentricity_norm',
        'bias',
    ],
    'f3': list(GRAPH_FEATURE_NAMES),
}
DEFAULT_GRAPH_FEATURE_SET = 'f3'

STATIC_GRAPH_FEATURE_DIM = len(GRAPH_FEATURE_NAMES)
DYNAMIC_NODE_FEATURE_DIM = 2
AUGMENTED_NODE_FEATURE_DIM = STATIC_GRAPH_FEATURE_DIM + DYNAMIC_NODE_FEATURE_DIM


def normalize_feature_set(feature_set=DEFAULT_GRAPH_FEATURE_SET):
    """Return a validated, lower-case feature-set identifier."""
    normalized = str(feature_set).strip().lower()
    if normalized not in GRAPH_FEATURE_SETS:
        choices = ', '.join(GRAPH_FEATURE_SETS)
        raise ValueError(
            f'Unknown graph feature set {feature_set!r}; expected one of: {choices}'
        )
    return normalized


def get_graph_feature_names(feature_set=DEFAULT_GRAPH_FEATURE_SET):
    """Return a copy of the ordered static feature names for an ablation."""
    return list(GRAPH_FEATURE_SETS[normalize_feature_set(feature_set)])


def get_static_graph_feature_dim(feature_set=DEFAULT_GRAPH_FEATURE_SET):
    return len(get_graph_feature_names(feature_set))


def get_augmented_node_feature_dim(feature_set=DEFAULT_GRAPH_FEATURE_SET):
    return get_static_graph_feature_dim(feature_set) + DYNAMIC_NODE_FEATURE_DIM

def normalize_column(arr: np.ndarray, col: int, method: str = "min-max") -> np.ndarray:
    """
    对二维 NumPy 数组的指定列按列做归一化。

    参数
    ----
    arr : np.ndarray
        原始二维数组，形状 (n_samples, n_features)。
    col : int
        要归一化的列索引（0 <= col < n_features）。
    method : str
        归一化方法，支持 "min-max" 或 "z_score"。
        - "min-max": (x - min) / (max - min) 使取值映射到 [0, 1] 范围 :contentReference[oaicite:0]{index=0}  
        - "z_score": (x - mean) / std   使归一化后满足均值为 0、标准差为 1 :contentReference[oaicite:1]{index=1}

    返回
    ----
    np.ndarray
        归一化后的新数组，原数组不被修改。
    """
    # 复制数组，避免原地修改
    out = arr.astype(float).copy()
    col_vals = out[:, col]

    if method == "min-max":
        min_val = col_vals.min()
        max_val = col_vals.max()
        # 如果 max == min，直接置零以避免除 0
        if np.isclose(max_val, min_val):
            out[:, col] = 0.0
        else:
            out[:, col] = (col_vals - min_val) / (max_val - min_val)
        return out

    elif method == "z-score":
        mean = col_vals.mean()
        std = col_vals.std(ddof=0)
        # 如果 std 接近 0，直接置零
        if np.isclose(std, 0.0):
            out[:, col] = 0.0
        else:
            out[:, col] = (col_vals - mean) / std
        return out

    else:
        raise ValueError(f"Unknown normalization method: {method!r}. "
                         "Use 'min-max' or 'z_score'.")

def calc_features(G): #根据网络计算网络中各个节点在训练全程不变的特征（fixed features）
    # 加入全程不变的特征
    features = [[] for _ in range(len(G.nodes))] #初始化特征列表，长度为节点数

    DC_centrality = nx.degree_centrality(G)
    BC_centrality = nx.betweenness_centrality(G)
    CC_centrality = nx.closeness_centrality(G)
    all_ones = np.ones(len(G.nodes))
    # c_centrality = nx.closeness_centrality(G)
    for i in G.nodes:
        # print('node:', i)
        features[i].append(DC_centrality[i])
        features[i].append(BC_centrality[i])
        features[i].append(CC_centrality[i])
        features[i].append(all_ones[i]) #初始全为1的特征
    features = np.array(features) #将特征列表转换为numpy数组
    features = normalize_column(features, col=0, method="min-max")
    features = normalize_column(features, col=1, method="min-max")
    features = normalize_column(features, col=2, method="min-max")

# 特征指示：决策变量（尽量给01，不给+-1）、初始全为1的特征（为了做邻域汇聚，获得结构信息）、没有点权、中心性指标一般不是、奖励（类似于maxcut、ising）、度中心性就OK
# 奖励惩罚项（放预算值，和准确度相反，可以设计成单目标）、特征归一化问题、有无concat(检查数据是否存在指数差异，f.normalize)、全局信息（问题本身还有哪些特征）、动力学过程的一、二阶展开

    return features #返回各个节点的特征构成的特征矩阵(n_nodes*n_features)

#定义经验回放池类，主要包含加入数据、采样数据两大函数
def _min_max_vector(values):
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values
    min_val = values.min()
    max_val = values.max()
    if np.isclose(max_val, min_val):
        return np.zeros_like(values, dtype=float)
    return (values - min_val) / (max_val - min_val)


def _mapping_to_vector(mapping, nodes):
    return np.array([mapping.get(node, 0.0) for node in nodes], dtype=float)


def _distance_matrix_from_graph(G, nodes):
    n_nodes = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    distance = np.full((n_nodes, n_nodes), fill_value=float(n_nodes), dtype=float)
    np.fill_diagonal(distance, 0.0)
    shortest_lengths = dict(nx.all_pairs_shortest_path_length(G))
    for src, lengths in shortest_lengths.items():
        src_idx = node_to_idx[src]
        for dst, value in lengths.items():
            distance[src_idx, node_to_idx[dst]] = value
    return distance


def _local_density_vector(G, nodes):
    density = []
    for node in nodes:
        ego_nodes = [node] + list(G.neighbors(node))
        density.append(nx.density(G.subgraph(ego_nodes)) if len(ego_nodes) > 1 else 0.0)
    return np.array(density, dtype=float)


def _two_hop_size_vector(G, nodes):
    sizes = []
    for node in nodes:
        two_hop_nodes = nx.single_source_shortest_path_length(G, node, cutoff=2)
        sizes.append(max(0, len(two_hop_nodes) - 1))
    return np.array(sizes, dtype=float)


def _spectral_position_matrix(G, nodes, dim=3):
    n_nodes = len(nodes)
    spectral = np.zeros((n_nodes, dim), dtype=float)
    if n_nodes <= 1:
        return spectral

    adjacency = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
    degrees = adjacency.sum(axis=1)
    inv_sqrt_degrees = np.zeros_like(degrees, dtype=float)
    nonzero = degrees > 0
    inv_sqrt_degrees[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    normalized_adjacency = inv_sqrt_degrees[:, None] * adjacency * inv_sqrt_degrees[None, :]
    laplacian = np.eye(n_nodes) - normalized_adjacency

    try:
        _, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        return spectral

    usable_dim = min(dim, max(0, n_nodes - 1))
    spectral[:, :usable_dim] = eigenvectors[:, 1:1 + usable_dim]
    for col in range(usable_dim):
        values = spectral[:, col]
        anchor = np.argmax(np.abs(values))
        if values[anchor] < 0:
            spectral[:, col] = -values
        spectral[:, col] = _min_max_vector(spectral[:, col])
    return spectral


# Compact topology encoder for no-SD source-localization experiments.
def calc_features(G, distance_matrix=None, feature_set=DEFAULT_GRAPH_FEATURE_SET):
    feature_names = get_graph_feature_names(feature_set)
    # Feature row i must describe the same node as adjacency row i and PPO
    # action i; never inherit NetworkX insertion order here.
    nodes = canonical_nodelist(G)
    n_nodes = len(nodes)
    if n_nodes == 0:
        return np.zeros((0, len(feature_names)), dtype=float)

    degree = _mapping_to_vector(nx.degree_centrality(G), nodes)
    betweenness = _mapping_to_vector(nx.betweenness_centrality(G), nodes)
    closeness = _mapping_to_vector(nx.closeness_centrality(G), nodes)
    core_number = _mapping_to_vector(nx.core_number(G), nodes)
    clustering = _mapping_to_vector(nx.clustering(G), nodes)

    if distance_matrix is None:
        distance = _distance_matrix_from_graph(G, nodes)
    else:
        distance = np.asarray(distance_matrix, dtype=float)
        if distance.shape != (n_nodes, n_nodes):
            raise ValueError(
                f'distance_matrix must have shape {(n_nodes, n_nodes)}, '
                f'got {distance.shape}'
            )
    rank_span = max(n_nodes - 1, 1)
    two_hop_ratio = np.count_nonzero(
        (distance > 0) & (distance <= 2), axis=1
    ).astype(float) / rank_span
    eccentricity = np.max(distance, axis=1)
    diameter = max(float(np.max(eccentricity)), 1.0)
    eccentricity_norm = eccentricity / diameter

    graph_density = float(nx.density(G))
    diameter_ratio = min(diameter / rank_span, 1.0)

    feature_columns = {
        'degree_norm': _min_max_vector(degree),
        'betweenness_norm': _min_max_vector(betweenness),
        'closeness_norm': _min_max_vector(closeness),
        'k_core_norm': _min_max_vector(core_number),
        'clustering_norm': np.clip(clustering, 0.0, 1.0),
        'two_hop_ratio': np.clip(two_hop_ratio, 0.0, 1.0),
        'eccentricity_norm': np.clip(eccentricity_norm, 0.0, 1.0),
        'graph_density': np.full(n_nodes, graph_density, dtype=float),
        'diameter_ratio': np.full(n_nodes, diameter_ratio, dtype=float),
        'bias': np.ones(n_nodes, dtype=float),
    }

    features = np.column_stack([feature_columns[name] for name in feature_names])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) #队列，先进先出

    def add(self, state, action, reward, next_state, done, G_index): #将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done, G_index))
        # print((state, action, reward, next_state, done, G_index))

    def sample(self, batch_size): #从buffer中采样数据，数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        return transitions

    def size(self): #目前buffer中数据的数量
        return len(self.buffer)

# def get_index(lst=None, item=''):
#     return [i for i in range(len(lst)) if lst[i] == item]

class SumTree:
    '''
    SumTree 是一种专门用于优先经验回放（PER）的二叉树数据结构，它将每个样本的优先级存储在叶子节点，其父节点值是子节点值之和，
    并在内部节点维护子节点优先级之和，从而实现 O(log n) 的高效加权采样和更新
    '''
    def __init__(self, capacity):
        '''
        二叉树上一共(2*capacity-1)个节点，其中capacity个叶子节点存储数据，capacity-1个非叶子节点存储累计优先级值
        所以对于每个叶子节点来说，有两套索引，一套是idx树索引，一套是dataIdx数据索引
        '''
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # 树节点数量有2*capacity-1个，其中capacity个叶子节点存储数据，capacity-1个非叶子节点存储累计优先级值
        self.data = np.zeros(capacity, dtype=object)  # 经验数据只有capacity个
        self.n_entries = 0 # 已存储数据的叶子节点的数量
        self.write = 0 # 数据写入指针，如果满载，后续从头开始覆盖

    def _propagate(self, idx, change):
        # 将下一层的节点值变化向上传播一层（递归实现叶子节点到根节点的传播）
        parent = (idx - 1) // 2 #逐层编号时，子节点和父节点之间索引的函数关系
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        '''
        给定一个值s∈[0, total_priority]，快速找到s落在哪个叶子节点上
        idx: 当前节点的索引.初始调用的时候，一般是0（根节点）
        s: 需要定位的累计优先级值
        return: idx: 累计优先级值所在的叶子节点的索引
        '''
        left = 2 * idx + 1 #idx的左子节点
        right = left + 1 #idx的右子节点
        if left >= len(self.tree): #超出二叉树最大索引，说明idx是叶子节点
            return idx
        if s <= self.tree[left]: # 如果s小于等于左子节点的值，说明s在左子树中
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self): 
        #返回根节点的值，即所有优先级之和
        return self.tree[0]

    def add(self, priority, data):
        '''
        将数据添加到树中，同时更新累计优先级值
        priority: 数据的优先级
        data: (s_0, a, r, s_1, done)这些经验数据本身
        '''
        idx = self.write + self.capacity - 1 #其中self.capacity - 1为非叶子节点数量
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0 #满载后从头覆盖
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        '''
        更新数据的优先级值，一并更新其所有上层节点的累计优先级值
        '''
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        '''
        给定一个值s∈[0, total_priority]，快速找到s落在哪个叶子节点上
        return: idx: 累计优先级值所在的叶子节点的索引
        return: self.tree[idx]: 目标叶子节点上的优先级值
        return: dataIdx: 目标叶子节点上的数据索引
        '''
        idx = self._retrieve(0, s)
        dataIdx = idx - (self.capacity - 1)
        # print('idx, dataIdx', idx, dataIdx)
        # return (idx, self.tree[idx], self.data[dataIdx])
        return (idx, self.tree[idx], self.data[dataIdx]) #idx为sumTree索引，self.tree[idx]为优先级值，self.data[dataIdx]为经验数据

# class SGBSTree:
#     '''
#     SGBSTree 主要用于完成SGBS算法寻优，主要包含三个步骤：Expansion, Simulation, Pruning
#     '''
#     def __init__(self, n_actions, n_steps, beta, gamma):
#         self.capacity = capacity
#         self.tree = np.zeros(2 * capacity - 1)
#         self.data = np.zeros(capacity, dtype=object)
#         self.n_entries = 0 # 已存储数据的叶子节点的数量
#         self.write = 0 # 数据写入指针，如果满载，后续从头开始覆盖

# 经验回放池，PER抽样，将经验数据存储在sumtree中
class PriorityReplayBuffer:
    e = 0.01 #确保每个样本都有被采样的可能
    a = 0.6 #a∈[0,1]优先级的指数衰减因子，越大越偏向于高优先级样本
    beta = 0.4 #beta∈[0,1]，校正强度的指数衰减因子，参数越大校正强度越大
    beta_increment_per_sampling = 0.001 # 随着训练进行，𝛽通常从较小值线性升至 1，以平衡早期的学习速度与后期的无偏性

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = SumTree(capacity)
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, data):
        p = self._get_priority(error)
        self.tree.add(p, data)
    def sample(self, n): #从sumtree中分段采样n个数据
        batch = []
        idxs = []
        segements = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segements * i
            b = segements * (i + 1)

            s = random.uniform(a, b) #分段采样。如果直接抽样，可能会出现高优先级被反复命中、batch内采样分布方差大等问题
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx) # 采样经验在整个sumTree中的索引

        sampling_probabilities = priorities / self.tree.total() # 采样概率

        # 校正TD误差的学习过程
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max() 
        # 归一化。避免梯度爆炸（如果某个P(i)非常小但是被抽到，is_weight将会很大，从而放大TD误差，导致梯度更新剧烈甚至发散。归一化是线性操作，不影响样本之间的相对权重）

        return batch, idxs, is_weight #返回采样的经验数据、对应的sumTree索引、校正强度

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries

##定义Mish激活函数
def mish(x): #Mish激活函数
    return x * (torch.tanh(F.softplus(x)))

##定义relu激活函数
def relu(x): #Mish激活函数
    return F.relu(x, inplace=False)

##定义relu激活函数与图卷积操作类
def relu(x): #relu激活函数
    return F.relu(x)
def getAllIndex(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]

def index2Onehot(index, n_nodes):
    """
    将索引元组转换为 one-hot 编码列表。

    参数：
    index (tuple): 包含索引的元组，例如 (2, 1, 4)。
    n_nodes (int): one-hot 编码的长度。

    返回：
    list: one-hot 编码的列表，每个元素是一个 NumPy 数组。
    """
    return [np.eye(n_nodes, dtype=int)[i] for i in index]

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def tensor2D_diag_cat(tensor2D_lst, begin_ind, end_ind):
    '''
    concatenate several 2D tensor(square) to a bigger tensor(square) along the main diagonal.
    :param tensor2D_lst: iterable object, every element of the iterable object is a tensor(square)
    :return: tensor concatenated
    '''
    shape_lst = [i.shape[0] for i in tensor2D_lst]
    # if not ind_lst:
    #     ind_lst = [sum(shape_lst[0:i]) for i in range(n_square)]
    shape_sum = sum(shape_lst)
    tensor_cat = torch.zeros((shape_sum, shape_sum))
    for tensor_i, begin_i, end_i in zip(tensor2D_lst, begin_ind, end_ind):
        tensor_cat[begin_i:end_i + 1, begin_i:end_i + 1] = tensor_i

    return tensor_cat

def split_and_agg(q_values_lst, split_point, actions):
    '''
    split a list according split_point, then for every list split, aggregate with agg_method
    :param q_values_lst: a list
    :param split_point: start points for every list
    :param actions: how to aggregate every list, max or with a certain number
    :return: list aggregated
    '''
    q_values_split = []
    for start_point, end_point in zip(split_point[:-2],split_point[1:-1]):
        q_values_split.append(q_values_lst[start_point:end_point])
    q_values_split.append(q_values_lst[split_point[-1]:])
    if actions == 'max':
        return [max(lst_i) for lst_i in q_values_split]
    else:
        return [lst_i[actions_i] for lst_i, actions_i in zip(q_values_split, actions)]


def _sparse_to_tensor(self, sparse_matrix):
    values = sparse_matrix.data
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col))
    i = torch.as_tensor(indices, dtype=torch.int64)
    v = torch.as_tensor(values, dtype=torch.float32)
    shape = sparse_matrix.shape
    tensor = torch.sparse_coo_tensor(i, v, shape, device=self.device)
    return tensor

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def data_generator(train_data_path, graph_type, n_nodes, G_index, distribution='Gauss'):
    pass
    # # G_distance = pickle.load('data/'+args.G_index+'.json')
    #     G_distance = np.loadtxt(G_distance_path)
    #     # with open(G_distance_path, 'rb') as fp:
    #     #     G_distance = pickle.load(fp)
    # else:
    #     print('you need to run geneTrainData.py to generate data first!')
    #     # G_distance = {} # 存储网络中每对节点之间的距离（连边权值为常数1）
    #     # for node_i in G.nodes:
    #     #     G_distance[node_i] = np.array(
    #     #         [[nx.shortest_path_length(G, source=node_i, target=node_j, weight=1) for node_j in G.nodes]])
    #     # with open(G_distance_path, "wb") as fp:
    #     #     pickle.dump(G_distance, fp)

    # if os.path.exists(G_distance_delay_path):
    #     # G_distance = pickle.load('data/'+args.G_index+'.json')
    #     G_distance_delay = np.loadtxt(G_distance_delay_path)
    #     # with open(G_distance_delay_path, 'rb') as fp:
    #     #     G_distance_delay = pickle.load(fp)
    # else:
    #     print('you need to run geneTrainData.py to generate data first!')
    #     # G_distance_delay = {} # 存储网络中每对节点之间的距离（连边权值为delay）
    #     # G = addDelay(G, distribution) #只有需要更新的时候才需要重新赋权
    #     # for node_i in G.nodes:
    #     #     G_distance_delay[node_i] = np.array(
    #     #         [[nx.shortest_path_length(G, source=node_i, target=node_j, weight='delay') for node_j in G.nodes]])
    #     # with open(G_distance_delay_path, "wb") as fp:
    #     #     pickle.dump(G_distance_delay, fp)

def _numbered_delay_files(graph_dir, graph_index):
    """Return sorted (realization_index, delay_path, sd_path) tuples."""
    pattern = re.compile(
        rf'^{re.escape(str(graph_index))}_GDistDelay_(\d+)\.npy$'
    )
    numbered = []
    for filename in os.listdir(graph_dir):
        match = pattern.fullmatch(filename)
        if match is None:
            continue
        realization_index = int(match.group(1))
        delay_path = os.path.join(graph_dir, filename)
        sd_path = os.path.join(
            graph_dir,
            f'{graph_index}_sd_{realization_index}.npy',
        )
        if not os.path.exists(sd_path):
            raise FileNotFoundError(
                f'Missing SD metadata for delay realization: {sd_path}'
            )
        numbered.append((realization_index, delay_path, sd_path))

    numbered.sort(key=lambda item: item[0])
    if numbered:
        indices = [item[0] for item in numbered]
        expected = list(range(len(indices)))
        if indices != expected:
            raise ValueError(
                f'Delay realization indices for graph {graph_index} in '
                f'{graph_dir} must be contiguous from 0; got {indices}'
            )
    return numbered


def _load_delay_realizations(graph_dir, graph_index, n_nodes, load_all,
                             hop_distance, lazy=False):
    """Load numbered delay matrices, with legacy single-file fallback."""
    numbered = _numbered_delay_files(graph_dir, graph_index)
    if not numbered:
        legacy_delay_path = os.path.join(
            graph_dir, f'{graph_index}_GDistDelay.npy'
        )
        legacy_sd_path = os.path.join(graph_dir, f'{graph_index}_sd.npy')
        if not os.path.exists(legacy_delay_path):
            raise FileNotFoundError(
                f'No numbered or legacy delay matrix found for graph '
                f'{graph_index} in {graph_dir}'
            )
        numbered = [(0, legacy_delay_path, legacy_sd_path)]

    selected = numbered if load_all else numbered[:1]
    matrices, indices, sd_values, paths = [], [], [], []
    for realization_index, delay_path, sd_path in selected:
        if not lazy:
            matrix = np.asarray(np.loadtxt(delay_path), dtype=float)
            if matrix.shape != (n_nodes, n_nodes):
                raise ValueError(
                    f'{delay_path} has shape {matrix.shape}; expected '
                    f'{(n_nodes, n_nodes)}'
                )
            if not np.isfinite(matrix).all():
                raise ValueError(f'Non-finite delay distance in {delay_path}')
            if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-9):
                raise ValueError(f'Asymmetric delay distance matrix: {delay_path}')
        if os.path.exists(sd_path):
            metadata = np.asarray(np.loadtxt(sd_path), dtype=float).reshape(-1)
            if metadata.size != 3 or not np.isfinite(metadata).all():
                raise ValueError(
                    f'{sd_path} must contain [delay_sd, graph_param, n_nodes]'
                )
            if not np.isclose(metadata[2], n_nodes):
                raise ValueError(
                    f'{sd_path} reports {metadata[2]} nodes; expected {n_nodes}'
                )
            sd_value = float(metadata[0])
        else:
            # Legacy datasets may contain only the delay-distance file.
            sd_value = float('nan')

        if not lazy:
            validate_delay_distance(
                matrix,
                hop_distance,
                sd=None if not np.isfinite(sd_value) else sd_value,
                context=delay_path,
            )
            matrices.append(matrix)

        indices.append(realization_index)
        sd_values.append(sd_value)
        paths.append(delay_path)
    return matrices, indices, sd_values, len(numbered), paths


def data_loader(data_path, graph_type, n_nodes, num_graphs, no_sd,
                load_features=True, feature_set=DEFAULT_GRAPH_FEATURE_SET,
                sample_delay_realizations=False,
                load_all_delay_realizations=False,
                lazy_delay_realizations=False):
    '''
    加载训练数据
    '''
    data = {
        'G': [],
        'G_matrix': [],
        'G_matrix_diag': [],
        'G_distance': [],
        'G_distance_delay': [],
        'G_distance_delay_realizations': [],
        'G_distance_delay_realization_paths': [],
        'delay_realization_indices': [],
        'delay_sd_values': [],
        'features': [],
    }
    for G_index in range(num_graphs): #每个规模下加载图的数量
        graph_dir = os.path.join(data_path, graph_type, f'{n_nodes}_nodes')
        G_path = os.path.join(graph_dir, f'{G_index}.npy')
        G_distance_path = os.path.join(graph_dir, f'{G_index}_GDist.npy')

        G_matrix = np.loadtxt(G_path)
        G = nx.from_numpy_array(G_matrix)
        G_matrix_diag = G_matrix + np.eye(G_matrix.shape[0])
        G_distance = np.loadtxt(G_distance_path)
        validate_adjacency_hop_alignment(
            G_matrix,
            G_distance,
            context=f'{graph_type}/{n_nodes}_nodes/graph_{G_index}',
        )
        if no_sd:
            G_distance_delay = G_distance.copy()
            delay_realizations = None
            delay_realization_paths = []
            realization_indices = []
            sd_values = []
        else:
            load_all = (
                sample_delay_realizations or load_all_delay_realizations
            )
            matrices, realization_indices, sd_values, available_count, delay_paths = (
                _load_delay_realizations(
                    graph_dir,
                    G_index,
                    n_nodes,
                    load_all=load_all,
                    hop_distance=G_distance,
                    lazy=lazy_delay_realizations,
                )
            )
            if lazy_delay_realizations:
                G_distance_delay = np.asarray(
                    np.loadtxt(delay_paths[0]), dtype=float
                )
                validate_delay_distance(
                    G_distance_delay,
                    G_distance,
                    sd=(
                        sd_values[0]
                        if np.isfinite(sd_values[0]) else None
                    ),
                    context=delay_paths[0],
                )
                delay_realizations = None
                delay_realization_paths = delay_paths
            else:
                G_distance_delay = matrices[0]
                delay_realizations = matrices if load_all else None
                delay_realization_paths = delay_paths if load_all else []
            if sample_delay_realizations and available_count < 2:
                raise ValueError(
                    f'SD training requires multiple delay realizations per '
                    f'topology, but graph {G_index} in {graph_dir} has '
                    f'{available_count}'
                )

        data['G'].append(G)
        data['G_matrix'].append(G_matrix)
        data['G_matrix_diag'].append(G_matrix_diag)
        data['G_distance'].append(G_distance)
        data['G_distance_delay'].append(G_distance_delay)
        data['G_distance_delay_realizations'].append(delay_realizations)
        data['G_distance_delay_realization_paths'].append(
            delay_realization_paths
        )
        data['delay_realization_indices'].append(realization_indices)
        data['delay_sd_values'].append(sd_values)
        if load_features:
            data['features'].append(calc_features(
                G,
                distance_matrix=G_distance,
                feature_set=feature_set,
            ))

    return data

def plot_training_results(train_log, save_path, experiment_record=None, prompt_record=True):
    # 绘制训练图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return_list_all, loss_list_all = train_log
    np.savetxt(save_path+'returns.txt', return_list_all)
    np.savetxt(save_path+'loss.txt', loss_list_all)

    mv_return = moving_average(return_list_all, 15) # moving_average函数后面的window_size参数必须是奇数
    episodes_list = list(range(len(mv_return)))
    plt.plot(episodes_list, mv_return, label='ma_return')
    # plt.plot(episodes_list, return_list_all)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Returns')

    plt.savefig(save_path + 'Returns.png', dpi=300)
    plt.close()

    loss_index = list(range(len(loss_list_all)))
    plt.figure(figsize=(8,5))
    plt.plot(loss_index, loss_list_all, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    # 在右上角添加一个插图显示后期loss细节
    axins = inset_axes(plt.gca(), width="40%", height="30%", loc='upper right')
    axins.plot(loss_index, loss_list_all, label='mini-Loss')
    xlim_index_scale = [int(len(loss_list_all)*0.4), int(len(loss_list_all)*1)]
    axins.set_xlim(xlim_index_scale[0], xlim_index_scale[1])  # 后期训练的epoch范围
    axins.set_ylim(min(loss_list_all[xlim_index_scale[0]:xlim_index_scale[1]]), max(loss_list_all[xlim_index_scale[0]:xlim_index_scale[1]]))
    axins.set_title('Zoomed In', fontsize=10)
    plt.legend()
    plt.savefig(save_path + 'train_loss.png', dpi=300)
    plt.close()
    
    
    if experiment_record is None and prompt_record:
        experiment_record = input("请写下你的观察记录，这些记录将会被保存下来：")
    if experiment_record is None:
        experiment_record = ""
    with open(save_path+"experiment_record.txt", "w", encoding="utf-8") as file:
        file.write(experiment_record)
    

# def addDelay(G, distribution):
#     # 添加边并设置高斯分布的时间延迟作为边的属性
#     if distribution == 'Gauss' or 'Normal':
#         for u, v in G.edges():
#             # 生成符合高斯分布的时间延迟
#             delay = random.gauss(args.miu, args.var)  # 由于delay的非负性要求，原文要求标准差远小于均值
#             while True:
#                 if delay > 0 and delay < 2:
#                     break
#                 else:
#                     delay = random.gauss(args.miu, args.var)
#             # print(delay)
#             G[u][v]['delay'] = delay
#
#     return G

def addDelay(G, distribution, mu, sd):
    # 添加边并设置高斯分布的时间延迟作为边的属性
    if distribution == 'Gauss' or 'Normal':
        # 设置截断区间
        lower, upper = mu - 2 * sd, mu + 2 * sd
        # 生成截断正态分布
        delays = truncnorm.rvs((lower - mu) / sd, (upper - mu) / sd, loc=mu, scale=sd, size=len(G.edges()))
        # print(delays)
        i = 0
        for u, v in G.edges():
            G[u][v]['delay'] = delays[i]
            i += 1

    return G

def minusMinTime(inform):
    # print(inform)
    t_min = min(inform.values())
    for k, v in inform.items():
        inform[k] -= t_min

    return inform


def calc_distance(G, V1, V2):
    # 基于图G，计算节点集V1和节点集V2中每对节点之间的距离
    data = np.array([[nx.dijkstra_path_length(G, i, j) for j in V2] for i in V1])
    d = pd.DataFrame(data, index=V1, columns=V2)
    # print('data', data, data.shape)

    # for i in V1:
    #     for j in V2:
    #         d.loc[i, j] = nx.dijkstra_path_length(G, i, j)
            # 加float是为了后面idxmin函数的使用
    # for j in V2:
    #     d[j] = d[j].apply(pd.to_numeric, errors='coerce', axis=0)

    return d


def calc_BC_prop(G, shortest_paths_temp):
    BC_dict = {node: 0 for node in G.nodes()}
    for k, v in shortest_paths_temp.items():
        for sub_k, sub_v in v.items():
            count_nodes = reduce(lambda x, y: x + y, [i[1:-1] for i in sub_v])
            for node in list(set(count_nodes)):
                BC_dict[node] += count_nodes.count(node) / len(sub_v)

    return BC_dict


def _shortest_path_predecessor_cache(G):
    """Build compact unweighted shortest-path DAGs for every source.

    The cache stores predecessor lists and BFS order only.  Unlike
    ``all_pairs_all_shortest_paths``, it never materializes every equal-length
    path, whose count can grow exponentially.
    """
    n = len(G)
    adjacency = [list(G.neighbors(node)) for node in range(n)]
    cache = []
    for source in range(n):
        predecessors = [[] for _ in range(n)]
        distance = np.full(n, -1, dtype=np.int32)
        distance[source] = 0
        order = []
        queue = deque([source])
        while queue:
            node = queue.popleft()
            order.append(node)
            next_distance = distance[node] + 1
            for neighbor in adjacency[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = next_distance
                    queue.append(neighbor)
                if distance[neighbor] == next_distance:
                    predecessors[neighbor].append(node)
        cache.append((order, predecessors))
    return cache


def conditional_betweenness_scores(G, selected, predecessor_cache=None):
    """Score nodes using shortest paths that avoid selected internal nodes.

    Selected nodes may still be path endpoints, matching the former CB
    implementation, but they cannot be used as internal transit nodes.
    """
    n = len(G)
    selected = set(selected)
    cache = predecessor_cache or _shortest_path_predecessor_cache(G)
    scores = np.zeros(n, dtype=np.float64)

    for source, (order, predecessors) in enumerate(cache):
        path_count = np.zeros(n, dtype=np.float64)
        dependency = np.zeros(n, dtype=np.float64)
        path_count[source] = 1.0

        for node in order[1:]:
            total = 0.0
            for predecessor in predecessors[node]:
                if predecessor == source or predecessor not in selected:
                    total += path_count[predecessor]
            path_count[node] = total

        for node in reversed(order):
            if path_count[node] == 0:
                continue
            coefficient = (1.0 + dependency[node]) / path_count[node]
            for predecessor in predecessors[node]:
                if predecessor == source or predecessor not in selected:
                    dependency[predecessor] += path_count[predecessor] * coefficient
            if node != source and node not in selected:
                scores[node] += dependency[node]

    return scores, cache


def select_conditional_betweenness(G, num_observers):
    """Select CB observers without enumerating or copying concrete paths."""
    selected = []
    cache = _shortest_path_predecessor_cache(G)
    for _ in range(num_observers):
        scores, _ = conditional_betweenness_scores(G, selected, cache)
        if selected:
            scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
    return selected


def _nodes_to_bitset(nodes):
    bitset = 0
    for node in nodes:
        bitset |= 1 << int(node)
    return bitset


def _bit_count(value):
    """Return the number of set bits on Python versions before 3.8."""
    return bin(value).count('1')


def _shortest_path_union_bitsets(distance, max_path_nodes):
    """Encode nodes on any eligible shortest path as compact Python bitsets."""
    distance = np.asarray(distance)
    n = distance.shape[0]
    path_bits = [[0] * n for _ in range(n)]
    max_hops = max_path_nodes - 1

    for source in range(n):
        for target in range(source + 1, n):
            hops = distance[source, target]
            if hops < 1 or hops > max_hops:
                continue
            on_shortest_path = np.flatnonzero(
                distance[source, :] + distance[:, target] == hops
            )
            bits = _nodes_to_bitset(on_shortest_path)
            path_bits[source][target] = bits
            path_bits[target][source] = bits
    return path_bits


def select_hv_obs(distance, num_observers, max_path_nodes=4):
    """Select HV-OBS sensors using cumulative coverage and bitset operations.

    Every graph node is still evaluated as the initial observer.  The bitset
    representation replaces nested shortest-path lists and repeated deep
    copies while preserving the high-coverage search scope.
    """
    distance = np.asarray(distance)
    n = distance.shape[0]
    path_bits = _shortest_path_union_bitsets(distance, max_path_nodes)

    best_score = -1
    best_observers = None
    for start in range(n):
        observers = [start]
        selected_bits = 1 << start
        coverage_count = np.zeros(n, dtype=np.int32)
        coverage_count[start] = 1
        candidate_coverage = [path_bits[node][start] for node in range(n)]

        for _ in range(num_observers - 1):
            minimum_coverage = int(coverage_count.min())
            needs_coverage = _nodes_to_bitset(
                np.flatnonzero(coverage_count == minimum_coverage)
            )

            next_observer = None
            next_gain = -1
            for candidate in range(n):
                if selected_bits & (1 << candidate):
                    continue
                gain = _bit_count(candidate_coverage[candidate] & needs_coverage)
                if gain > next_gain:
                    next_gain = gain
                    next_observer = candidate

            added_coverage = candidate_coverage[next_observer]
            remaining = added_coverage
            while remaining:
                least_bit = remaining & -remaining
                node = least_bit.bit_length() - 1
                coverage_count[node] += 1
                remaining ^= least_bit

            observers.append(next_observer)
            selected_bits |= 1 << next_observer
            for candidate in range(n):
                candidate_coverage[candidate] |= path_bits[candidate][next_observer]

        minimum_coverage = int(coverage_count.min())
        score = minimum_coverage * n + int(np.count_nonzero(coverage_count > minimum_coverage))
        if score > best_score:
            best_score = score
            best_observers = observers

    return best_observers


def del_BC_node(paths_dict, node):
    paths_dict_backup = copy.deepcopy(paths_dict)
    for k, v in paths_dict_backup.items():
        for sub_k, sub_v in v.items():
            for path in sub_v:
                if node in path[1:-1]:
                    paths_dict[k][sub_k].remove(path)
                    if len(paths_dict[k][sub_k]) == 0:  # 如果这些路径全被remove，即全部需要经过初始介数最大节点，删除空列表
                        del paths_dict[k][sub_k]

            if len(paths_dict[k]) == 0:  # 如果删除空了，那么删除上一级键值对
                del paths_dict[k]

    return paths_dict

class KMedian:
    def __init__(self, n_clusters=3, max_iter=1000, tol=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, G):
        # 随机初始化质心
        self.centroids = np.random.choice(G.nodes(), size=self.n_clusters, replace=False)
        # print('centroids', self.centroids)

        for iter in range(self.max_iter):
            # print('iter', iter)
            # 计算每个点到所有质心的距离
            distances = calc_distance(G, G.nodes, self.centroids)
            # print('distances', distances.values, type(distances))

            # 分配每个点到最近的质心
            labels_idx = np.argmin(distances.values, axis=1)
            self.labels = [self.centroids[labels_i] for labels_i in labels_idx]
            # self.labels = distances.idxmin(axis=1)
            # print('labels', self.labels, type(self.labels))

            # 更新质心（使用当前分类为每个质心的点的平均值，作为新的质心位置） ？？
            new_centroids = []
            cents_distance = 0  # 记录更新前后质心之间的距离
            for cent_i in self.centroids:
                # print(type(np.array(G.nodes())))
                cent_i_nodes = np.array(G.nodes())[np.array(self.labels) == cent_i]
                # print(type(cent_i_nodes))
                sub_graph = G.subgraph(list(cent_i_nodes))
                closeness_centrality = nx.closeness_centrality(sub_graph)
                # print('closeness_centrality', closeness_centrality)
                new_cent_i = max(closeness_centrality, key=closeness_centrality.get)
                cents_distance += nx.dijkstra_path_length(G, new_cent_i, cent_i)
                new_centroids.append(new_cent_i)
            self.centroids = np.array(new_centroids)
            # print('new_centroids', new_centroids)

            # 检查收敛条件
            if cents_distance < self.tol:
                # print('get tol', cents_distance)
                break

    def predict(self, G):
        # 计算每个点到所有质心的距离
        distances = calc_distance(G, G.nodes(), self.centroids)

        # 分配每个点到最近的质心
        labels_idx = np.argmin(distances.values, axis=1)
        labels = [self.centroids[labels_i] for labels_i in labels_idx]

        return labels

def choice_obs_greedy(G, obs_lst, node_in, shortest_paths_init): # for the case place_strategy == 'HV_OBS'
    '''
    To choose the next observer greedily and return the node_in
    :param G: networkx graph
    :param obs_lst: the list of observers
    :param node_in: 1-D array, 1 means the node is not in PLS, 0 means the node is in once, more in more negative
    :param shortest_paths_init: the shortest paths of the graph (2 <= length <= L)
    :return: obs_lst, node_in, stage
    '''
    new_node_in_dict = {}
    stage = 1
    # print('node_in', node_in)
    for node in G.nodes():
        if node in obs_lst: # 排除已布设传感器的节点
            continue
        node_in_temp = copy.deepcopy(node_in)
        for obs_i in obs_lst: # 将候选点与已有传感器之间的最短路径上的所有节点统计频数
            # print('node', node, 'obs_i', obs_i)
            try:
                node_add = [node_i for path_i in shortest_paths_init[node][obs_i] for node_i in path_i]
            except KeyError: #不存在这两点之间的最短路径，才会弹出KeyError
                continue
            for node_i in node_add:
                node_in_temp[node_i] -= 1
            # print(node_in_temp)
        # node_in_temp[node_in_temp < 0] = 0
        new_node_in_dict[node] = np.sum(node_in[node_in > 0]) - np.sum(node_in_temp > 0)
    # print(new_node_in_dict)
    max_obs = max(new_node_in_dict, key=new_node_in_dict.get) #选定下一步布设点

    for obs_i in obs_lst:  # 针对下一步布设点与已有传感器之间的最短路径上的所有节点，统计频数
        try:
            node_add = [node_i for path_i in shortest_paths_init[max_obs][obs_i] for node_i in path_i] #双重推导式
        except KeyError:  # 不存在这两点之间的最短路径，才会弹出KeyError
            continue
        for node_i in node_add:
            node_in[node_i] -= 1
    obs_lst.append(max_obs)

    if all(node_in <= 0): #如果加入本节点后，所有节点都在这一轮被覆盖，进入下一覆盖阶段
        stage += 1
        node_in += 1
        # print('进入下一阶段')

    return obs_lst, node_in, stage

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, out, dropout):
#         #print(nhid)
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, out)
#         self.dropout = dropout

#     def _mask(self):
#         return self.mask

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return x
    
# class GCN_one(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         #print(nhid)
#         super(GCN_one, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.dropout = dropout

#     def _mask(self):
#         return self.mask

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         return x

# class GAT(nn.Module): #在聚合neighbor的情况下只需一层GAT
#     def __init__(self, nfeat, nhid, dropout, alpha, nheads):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout

#         self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = GraphAttention(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return x

# class Attention(nn.Module):
#     def __init__(self, in_size, hidden_size):
#         super(Attention, self).__init__()

#         self.project = nn.Sequential(
#             nn.Linear(in_size, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, 1, bias=False)
#         )

#     def forward(self, z):
#         w = self.project(z)
#         beta = torch.softmax(w, dim=1)
#         return (beta * z).sum(1), beta

# class FeatureFusionGate(nn.Module):
#     def __init__(self, embedding_dim):
#         super(FeatureFusionGate, self).__init__()
#         self.linear1 = nn.Linear(embedding_dim*2, embedding_dim)
#         self.linear2 = nn.Linear(embedding_dim*2, embedding_dim)

#     def forward(self, bundle_like, item_like):
#         concat_feature = torch.cat((bundle_like, item_like), dim=1)
#         fusion = F.elu(self.linear1(concat_feature))
#         gate = F.sigmoid(self.linear2(concat_feature))
#         out = torch.mul(gate, fusion) + torch.mul((1-gate), bundle_like)
#         return out



# class ListGCN(nn.Module):
#     def __init__(self, dropout_gnn, dropout_pre, num_user, num_item, dim_latent, dim_feat, dim_ori, alpha, nheads, hidden_size, active_fun, pred_method):
#         super(ListGCN, self).__init__()

#         #self.reg_weight = reg_weight
#         self.dropout1 = dropout_gnn
#         self.dropout2 = dropout_pre
#         self.alpha = alpha
#         self.concat = Parameter(torch.ones(2, 1))
#         self.softmax = nn.Softmax(dim=0)
#         # self.preference = nn.init.xavier_normal_(torch.rand((num_user, dim_feat), requires_grad=True)).cuda()
#         self.preference = nn.init.xavier_normal_(torch.rand((num_user, dim_feat), requires_grad=True))
#         self.MLP = nn.Linear(dim_ori, dim_feat)

#         self.MGCN1 = GCN(dim_feat, dim_latent, dim_latent, self.dropout1)
#         self.MGCN2 = GCN_one(dim_feat, dim_latent, self.dropout1)
#         self.MGCN3 = GCN_one(dim_feat, dim_latent, self.dropout1)
#         self.MGCN4 = GCN_one(dim_feat, dim_latent, self.dropout1)
#         self.MGCN5 = GCN_one(dim_feat, dim_latent, self.dropout1)
#         self.attention = Attention(dim_latent, hidden_size)
#         self.featureFusionGate = FeatureFusionGate(dim_latent)
#         self.tanh = nn.Tanh()
#         self.result = None

        
#         #复杂的预测模块所需
#         self.active_fun_str = active_fun
#         if self.active_fun_str == 'relu':
#             self.active_fun = nn.ReLU()
#         else:
#             self.active_fun = nn.LeakyReLU()
            
#         self.pred_method = pred_method
#         if self.pred_method == 'mlp':
#             self.predict_l_1 = nn.Linear(in_features=dim_latent * 2, out_features=dim_latent)
#             self.predict_l_2 = nn.Linear(in_features=dim_latent, out_features=dim_latent)
#             self.predict_l_3 = nn.Linear(in_features=dim_latent, out_features=1)
#         elif self.pred_method == 'joint':
#             self.predict_l_1 = nn.Linear(in_features=dim_latent * 2, out_features=dim_latent)
#             self.predict_l_2 = nn.Linear(in_features=dim_latent, out_features=dim_latent)
#             self.predict_ul = nn.Linear(in_features=dim_latent, out_features=dim_latent)
#             self.predict_il = nn.Linear(in_features=dim_latent, out_features=dim_latent)
#             self.fusion_l = nn.Linear(in_features=dim_latent * 2, out_features=1)
        
#         ####### 分界线


#     def forward(self, features, features1, fadj, sadj1, sadj2, sadj3, special):
#         ori_features = torch.matmul(features,self.softmax(self.concat))
#         item_features = ori_features.reshape((195283,-1))
#         item_features = self.MLP(item_features)

#         bundle_it = self.MGCN5(features1, special)
#         item_final = self.featureFusionGate(item_features, bundle_it)
#         x = torch.cat((self.preference, item_final), dim=0)
        
#         emb1 = self.MGCN1(x, fadj)# basic GCN out1 -- fadj ui biparite graph
#         emb2 = self.MGCN2(x, sadj1) # Special_GAT out2 -- sadj1 neighbor graph 1
#         emb3 = self.MGCN3(x, sadj2)  # Special_GAT out3 -- sadj2 neighbor graph 2
#         emb4 = self.MGCN4(x, sadj3)  # Special_GAT out3 -- sadj3 neighbor graph 3
        
#         emb = torch.stack([emb1, emb2, emb3, emb4], dim=1)
#         rep, att = self.attention(emb)
#         self.result = rep
#         return self.result
    
#     def lookup_emb(self, emb_index):
#         if self.result is not None:
#             return self.result[emb_index]
#         else:
#             return False

#     def predict(self, u_emb, i_emb):
#         ###### 复杂的预测模块
#         if self.pred_method == 'mlp':
#             drop_layer = nn.Dropout(self.dropout2)
#             if self.active_fun_str != 'none':
#                 layer_1 = self.active_fun(drop_layer(
#                     self.predict_l_1(torch.cat((u_emb, i_emb), dim=1))))
#                 layer_2 = self.active_fun(drop_layer(self.predict_l_2(layer_1)))
#                 predict_value = self.active_fun(self.predict_l_3(layer_2))
#             else:
#                 layer_1 = drop_layer(
#                     self.predict_l_1(torch.cat((u_emb, i_emb), dim=1)))
#                 layer_2 = drop_layer(self.predict_l_2(layer_1))
#                 predict_value = self.predict_l_3(layer_2)
#         elif self.pred_method == 'joint':
#             drop_layer = nn.Dropout(self.dropout2)
#             if self.active_fun_str != 'none':
#                 layer_1 = self.active_fun(drop_layer(
#                     self.predict_l_1(torch.cat((u_emb, i_emb), dim=1))))
#                 layer_2 = self.active_fun(drop_layer(self.predict_l_2(layer_1)))
#                 mat_vec = layer_2
#                 u_layer = self.active_fun(drop_layer(self.predict_ul(u_emb)))
#                 i_layer = self.active_fun(drop_layer(self.predict_il(i_emb)))
#                 rep_vec = torch.mul(u_layer, i_layer)
#                 predict_value = self.active_fun(self.fusion_l(
#                     torch.cat((mat_vec, rep_vec), dim=1)))
#             else:
#                 layer_1 = drop_layer(
#                     self.predict_l_1(torch.cat((u_emb, i_emb), dim=1)))
#                 layer_2 = drop_layer(self.predict_l_2(layer_1))
#                 mat_vec = layer_2
#                 u_layer = drop_layer(self.predict_ul(u_emb))
#                 i_layer = drop_layer(self.predict_il(i_emb))
#                 rep_vec = torch.mul(u_layer, i_layer)
#                 predict_value = self.fusion_l(
#                     torch.cat((mat_vec, rep_vec), dim=1))
#         else:
#             embd = torch.mm(u_emb, i_emb.t()).sum(0)
#             predict_value = embd.flatten()
#         return predict_value


