import torch
from OP_Trainer import OP_Trainer
import random
import numpy as np
import utils

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

model_num = 35  # 模型编号
n_nodes = 50    # 节点数量

trainer_params = {
    'train_data': {
        'data_path': 'data/train_data/',  
        'graph_type': 'BA',  
        'n_nodes': n_nodes,  
        'num_graphs': 100,  
        'no_sd': False  
    },
    'train_model': {
        'num_episodes': 5000,  
        'G_num': 100,  
        'update_timestep': 512,  # PPO 每收集 512 步轨迹更新一次
    },
    'model_save_path': f'model_save/model_{model_num}/{model_num}.pt',  
    'results_save': {
        'save_path': f'model_save/model_{model_num}/',  
    }
}

model_params = {
    'PPOAgent': {
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

if __name__ == '__main__':
    set_seed(42)
    trainer = OP_Trainer(trainer_params, model_params, env_params)
    trainer.run()
