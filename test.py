import torch
import random
import numpy as np
from OP_Tester import OP_Tester
import utils

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_parmas(model_num, n_nodes, graph_type='BA'):
    tester_params = {
        'test_data': {
            'data_path': 'data/train_data/',  
            'graph_type': graph_type,  
            'n_nodes': n_nodes,  
            'num_graphs': 100,  
            'no_sd': False  
        },
        'results_save_path': f'result/model_{model_num}/model_{model_num}_{graph_type}_{n_nodes}.json'  
    }

    method_params = {
        'test_methods': ['SGBS', 'one_shot'],  
        'model_methods': ['PPOAgent'],  
        'test_init_params': {
            'one_shot':{
                'need_model': True  
            },
            'SGBS':{
                'beta': 4,  
                'gamma': 4,  
                'need_model': True  
            },
            'BC':{'need_model': False},
            'High_Coverage_Rate':{'need_model': False},
            'CB':{'need_model': False},
            'HV_OBS':{'need_model': False, 'L': 4},
            'K_Median':{'need_model': False},
            'random':{'need_model': False},
            'GA':{'need_model': False}
        },
        'model_init_params':{
            'PPOAgent': {
                'model_path': f'model_save/model_{model_num}/{model_num}.pt',
                'n_features_aug': utils.AUGMENTED_NODE_FEATURE_DIM,
                'dens_obs': 0.1,
                'hidden': [8, 16, 32, 64, 32, 16, 8],
                'learning_rate': 0.0001,
                'gamma': 1.0,
                'gae_lambda': 0.95,
                'clip_param': 0.2,
                'ppo_epochs': 4,
                'entropy_coef': 0.01,
                'entropy_decay': 0.999,
                'min_entropy_coef': 0.001,
                'dropout_p': 0.0,
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            },
            'GCN_DoubleDQN': {
                'model_path': f'model_save/model_{model_num}/{model_num}.pt',
                'n_features_aug': utils.AUGMENTED_NODE_FEATURE_DIM,
                'dens_obs': 0.1,
                'hidden': [8, 16, 32, 64, 32, 16, 8],
                'learning_rate': 0.001,
                'lr_gamma': 0.9999954,
                'gamma': 0.9,
                'epsilon': 0.1,
                'target_update': 1000,
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            }
        }
    }

    env_params = {
        'OP_Env_PLO':{
            'dens_obs': 0.1,  
            'L': 4  
        },
        'OP_Env_Precision':{
            'dens_obs': 0.1,  
            'sample_method': 'order',  
            'spread_method': 'snowball',  
            'locate_strategy': 'TRBS',  
            'begin_shaping': 200000 * 1.0,  
            'shaping_param': 1.0  
        }
    }

    return tester_params, method_params, env_params

if __name__ == '__main__':
    set_seed(42)
    model_num = 34  # Pre-trained model ID (e.g., trained on N=50, BA graphs)
    for graph_type in ['BA', 'ER', 'WS']:
        for n_nodes in range(50, 210, 10):
            print(f'graph_type={graph_type}, n_nodes={n_nodes}')
            tester_params, method_params, env_params = get_parmas(model_num, n_nodes, graph_type)
            tester = OP_Tester(tester_params, method_params, env_params)
            tester.run()
