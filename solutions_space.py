# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import torch
import networkx as nx
import itertools
import pandas as pd

from sourceLocalizationClass import SingleSourceLocalization
import utils

config = {
    'train_data_path': 'data/train_data/',
    'graph_type': 'BA',
    'dens_obs': 0.10,          
    'place_strategy': 'one_shot', 
    'node_range': [50, 60],     
    'model_num': 33,            
    'spread_method': 'snowball',
    'locate_strategy': 'TRBS',  
    'sample_method': 'order',   
    'L': 4,                     
    'no_sd': True               
}

def get_neighbor_solutions(G, solution):
    node_in = solution.copy()
    node_out = list(set(G.nodes()) - set(solution))
    neighbors = [
        [x for x in node_in if x != remove_node] + [add_node]
        for remove_node, add_node in itertools.product(node_in, node_out)
    ]
    return neighbors

def solution2value(solution, G, G_distance_delay, G_distance, source_actual_lst, SL_Model):
    rank_lst_nodes = []
    for source_actual in source_actual_lst:
        inform = SL_Model.get_inform(source_actual, solution, G_distance_delay, spread_method=config['spread_method'])
        source_actual_rank = SL_Model.locate_source(source_actual, inform, G_distance, locate_strategy=config['locate_strategy'])
        rank_lst_nodes.append(source_actual_rank)
    return sum(rank_lst_nodes) / len(rank_lst_nodes)

if __name__ == '__main__':
    place_strategy = config['place_strategy']
    dens_obs = config['dens_obs']
    train_data_path = config['train_data_path']
    graph_type = config['graph_type']
    
    nodes_num = list(range(config['node_range'][0], config['node_range'][1], 10))
    time_dict = dict()

    for n_nodes in nodes_num:
        tic = time.time()
        
        print(f"正在加载规模 n_nodes={n_nodes} 的图数据集...")
        test_data = utils.data_loader(
            data_path=train_data_path,
            graph_type=graph_type,
            n_nodes=n_nodes,
            num_graphs=100,  
            no_sd=config['no_sd']
        )

        for G_name in range(0, 1): 
            print('节点数 {0}，第 {1} 个图'.format(n_nodes, G_name))
            
            if place_strategy in ['BC', 'High_Coverage_Rate', 'CB', 'HV_OBS', 'one_shot', 'SGBS']:
                repeat = 1
            else:
                repeat = 100
                
            for _ in range(repeat):
                G = test_data['G'][G_name]
                G_matrix = test_data['G_matrix'][G_name]
                G_matrix_diag = test_data['G_matrix_diag'][G_name]
                G_distance = test_data['G_distance'][G_name]
                G_distance_delay = test_data['G_distance_delay'][G_name]
                
                SL_Model = SingleSourceLocalization(G, G_matrix_diag)
                source_actual_lst = SL_Model.sample_source_lst(n_source=n_nodes, sample_method=config['sample_method'])
                
                model_path = f'model_save/model_{config["model_num"]}/{config["model_num"]}.pt'
                
                test_init_params = {
                    'need_model': True if place_strategy in ['one_shot', 'SGBS'] else False,
                    'gamma': 4,
                    'beta': 4,
                    'HV_OBS': {'L': config['L']}
                }
                
                model_init_params = {
                    'model_path': model_path,
                    'n_features_aug': utils.AUGMENTED_NODE_FEATURE_DIM,
                    'dens_obs': dens_obs,
                    'hidden': [8, 16, 32, 64, 32, 16, 8],
                    'learning_rate': 0.0003,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_param': 0.2,
                    'ppo_epochs': 4,
                    'entropy_coef': 0.01,
                    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
                }
                
                place_obs_lst = SL_Model.place_observers(
                    dens_obs=dens_obs,
                    place_strategy=place_strategy,
                    model_method='PPOAgent',
                    test_init_params=test_init_params,
                    model_init_params=model_init_params,
                    source_actual_lst=source_actual_lst,
                    G_distance_delay=G_distance_delay,
                    spread_method=config['spread_method'],
                    G_distance=G_distance,
                    locate_strategy=config['locate_strategy']
                )
                print('当前策略生成的传感器布设点位:', place_obs_lst)
                
                neighbor_solutions = get_neighbor_solutions(G, place_obs_lst)
                print(f"生成的邻域解数量: {len(neighbor_solutions)}")
                
                values = list(map(lambda x: solution2value(x, G, G_distance_delay, G_distance, source_actual_lst, SL_Model), neighbor_solutions))
                print('邻域解评价值范围: Min={0:.4f}, Max={1:.4f}'.format(min(values), max(values)))
                
                solution_value_dict = dict(zip([tuple(s) for s in neighbor_solutions], values))
                current_val = solution2value(place_obs_lst, G, G_distance_delay, G_distance, source_actual_lst, SL_Model)
                solution_value_dict[tuple(place_obs_lst)] = current_val
                
                nodes_list = list(solution_value_dict.keys())
                values = list(solution_value_dict.values())
                
                space_G = nx.Graph()
                for nodes in nodes_list:
                    space_G.add_node(nodes)  
                
                k_sensors = int(dens_obs * n_nodes)
                for i in range(len(nodes_list)):
                    for j in range(i + 1, len(nodes_list)):
                        if len(set(nodes_list[i]) & set(nodes_list[j])) == (k_sensors - 1):
                            space_G.add_edge(nodes_list[i], nodes_list[j])
                
                pos = nx.spring_layout(space_G, seed=42)
                
                val_min, val_max = min(values), max(values)
                if np.isclose(val_max, val_min):
                    norm_values = np.zeros(len(values))
                else:
                    norm_values = (np.array(values) - val_min) / (val_max - val_min)
                
                cmap = plt.cm.viridis
                colors = [cmap(norm_val) for norm_val in norm_values]
                
                plt.figure(figsize=(11, 8))
                ax = plt.gca() 
                
                nx.draw(
                    space_G,
                    pos,
                    with_labels=False,
                    node_color=colors,
                    cmap=cmap,
                    node_size=600,
                    edge_color='gainsboro',
                    ax=ax
                )
                
                labels = {node: f"{value:.2f}" for node, value in solution_value_dict.items()}
                nx.draw_networkx_labels(space_G, pos, labels=labels, font_size=7, font_color='black', ax=ax)
                
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=val_min, vmax=val_max))
                sm.set_array([])
                plt.colorbar(sm, label='Solution Distance Error Value', ax=ax)
                plt.title(f"Optimization Fitness Landscape (Scale: {n_nodes} nodes, Strategy: {place_strategy})")
                plt.show()

        time_dict[n_nodes] = time.time() - tic
        print(f"规模 {n_nodes} 耗时: {time_dict[n_nodes]:.2f}s\n" + "-"*40)
