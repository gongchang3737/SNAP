# -*- coding: utf-8 -*-
"""对比 basic_method + 多模型方法的曲线，通过配置列表管理，方便增减"""
import json
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 0. 配置区：在此增删对比曲线
# ============================================

# ---- basic methods ----
basic_configs = [
    {'name': 'random',             'label': 'random',              'color': '#7f7f7f', 'marker': 'o', 'linestyle': '-'},
    {'name': 'BC',                 'label': 'BC',                  'color': '#d62728', 'marker': 's', 'linestyle': '-'},
    {'name': 'High_Coverage_Rate', 'label': 'High Coverage Rate',  'color': '#ff7f0e', 'marker': 'D', 'linestyle': '-'},
    {'name': 'CB',                 'label': 'CB',                  'color': '#9467bd', 'marker': '^', 'linestyle': '-'},
    {'name': 'HV_OBS',             'label': 'HV-OBS',              'color': '#8c564b', 'marker': 'v', 'linestyle': '-'},
]

# ---- model methods ----
# 每项: model_num, model_method, search_methods, label_prefix, color, markers[对应search_methods]
model_configs = [
    # {'model_num': 30, 'model_method': 'GCN_DoubleDQN', 'search_methods': ['SGBS', 'one_shot'],
    #  'label_prefix': 'DQN-30', 'color': '#1f77b4', 'markers': ['o', 's']},
    {'model_num': 31, 'model_method': 'GCN_DoubleDQN', 'search_methods': ['SGBS', 'one_shot'],
     'label_prefix': 'DQN-31', 'color': '#2ca02c', 'markers': ['o', 's']},
    {'model_num': 32, 'model_method': 'PPOAgent',       'search_methods': ['SGBS', 'one_shot'],
     'label_prefix': 'PPO-32', 'color': '#e377c2', 'markers': ['o', 's']},
    {'model_num': 33, 'model_method': 'PPOAgent',       'search_methods': ['SGBS', 'one_shot'],
     'label_prefix': 'PPO-33', 'color': '#17becf', 'markers': ['o', 's']},
    {'model_num': 34, 'model_method': 'PPOAgent',       'search_methods': ['SGBS', 'one_shot'],
     'label_prefix': 'PPO-34', 'color': "#3cdb30", 'markers': ['o', 's']}
]

# ============================================
# 1. 数据加载
# ============================================
n_nodes_range = list(range(50, 210, 10))

# basic methods
data_basic = {}
for cfg in basic_configs:
    data_basic[cfg['name']] = {}
    for n in n_nodes_range:
        with open(f'result/basic_method/basic_method_{n}.json') as f:
            d = json.load(f)
            data_basic[cfg['name']][n] = d[cfg['name']]['ave_rank']

# model methods
data_models = {}
for cfg in model_configs:
    mn = cfg['model_num']
    data_models[mn] = {sk: {} for sk in cfg['search_methods']}
    for n in n_nodes_range:
        path = f'result/model_{mn}/model_{mn}_{n}.json'
        try:
            with open(path) as f:
                d = json.load(f)
            for sk in cfg['search_methods']:
                data_models[mn][sk][n] = d[cfg['model_method']][sk]['ave_rank']
        except FileNotFoundError:
            print(f'  [WARN] 文件不存在，跳过: {path}')

# ============================================
# 2. 构建 curves / colors / linestyles / markers
# ============================================
def to_vec(d, nodes):
    return np.array([d[n] for n in nodes])

curves = {}
colors = []
linestyles = []
markers = []

# basic
for cfg in basic_configs:
    curves[cfg['label']] = to_vec(data_basic[cfg['name']], n_nodes_range)
    colors.append(cfg['color'])
    linestyles.append(cfg['linestyle'])
    markers.append(cfg['marker'])

# models
for cfg in model_configs:
    mn = cfg['model_num']
    for i, sk in enumerate(cfg['search_methods']):
        label = f"{cfg['label_prefix']} + {sk}"
        curves[label] = to_vec(data_models[mn][sk], n_nodes_range)
        colors.append(cfg['color'])
        linestyles.append('-' if i == 0 else '--')
        markers.append(cfg['markers'][i])

# ============================================
# 3. 绘图
# ============================================
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
})

fig, ax = plt.subplots(figsize=(12, 7))

for i, (label, y) in enumerate(curves.items()):
    ax.plot(
        n_nodes_range, y,
        color=colors[i],
        linestyle=linestyles[i],
        marker=markers[i],
        markersize=7,
        linewidth=1.8,
        label=label,
        markeredgewidth=0.5,
        markeredgecolor='white',
    )

ax.set_xlabel('Number of Nodes')
ax.set_ylabel('Average Rank')
ax.set_xticks(n_nodes_range)
ax.grid(True, alpha=0.3, linestyle='--')

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=6,
    frameon=True,
    fancybox=True,
    shadow=True,
    edgecolor='gray',
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('figure_save/comparison_11methods.png', dpi=300, bbox_inches='tight')
plt.show()
