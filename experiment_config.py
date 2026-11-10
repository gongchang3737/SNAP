# -*- coding: utf-8 -*-
"""Shared experiment configuration for SNAP training and evaluation."""

import torch

import utils
from sensor_budget_config import SENSOR_BUDGETS, get_sensor_budget


GRAPH_TYPES = ['BA', 'ER', 'GEO', 'WS']
TRAIN_SIZES = [20, 50, 100]
TEST_SIZES = [20, 50, 100, 200, 300, 500]

TRAIN_DATA_PATH = 'data/train_data/'
TEST_DATA_PATH = 'data/test_data/'

TRAIN_GRAPHS_PER_GROUP = 1000
TEST_GRAPHS_PER_GROUP = 100

# Kept only for compatibility with existing constructor signatures.
# Sensor counts now come exclusively from SENSOR_BUDGETS.
DENS_OBS = None
HIDDEN = [8, 16, 32, 64, 32, 16, 8]


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_data_groups(data_path, graph_types, n_nodes_list, num_graphs, no_sd=False,
                      feature_set=utils.DEFAULT_GRAPH_FEATURE_SET,
                      sample_delay_realizations=False):
    feature_set = utils.normalize_feature_set(feature_set)
    groups = []
    for graph_type in graph_types:
        for n_nodes in n_nodes_list:
            groups.append({
                'data_path': data_path,
                'graph_type': graph_type,
                'n_nodes': n_nodes,
                'num_graphs': num_graphs,
                'no_sd': no_sd,
                'feature_set': feature_set,
                'sample_delay_realizations': sample_delay_realizations,
            })
    return groups


def build_env_params():
    return {
        'OP_Env_PLO': {
            'dens_obs': DENS_OBS,
            'L': 4,
        },
        'OP_Env_Precision': {
            'dens_obs': DENS_OBS,
            'sample_method': 'order',
            'spread_method': 'snowball',
            'locate_strategy': 'TRBS',
            'begin_shaping': 200000 * 1.0,
            'shaping_param': 1.0,
        },
    }


def build_model_params(model_method, model_path=None,
                       feature_set=utils.DEFAULT_GRAPH_FEATURE_SET):
    feature_set = utils.normalize_feature_set(feature_set)
    augmented_feature_dim = utils.get_augmented_node_feature_dim(feature_set)
    if model_method == 'PPOAgent':
        return {
            'PPOAgent': {
                'n_features_aug': augmented_feature_dim,
                'dens_obs': DENS_OBS,
                'hidden': HIDDEN,
                'learning_rate': 0.0001,
                # Marginal rank rewards telescope to the final placement
                # objective only when they are not discounted.
                'gamma': 1.0,
                'gae_lambda': 0.95,
                'clip_param': 0.2,
                'ppo_epochs': 4,
                'entropy_coef': 0.01,
                'entropy_decay': 0.999,
                'min_entropy_coef': 0.001,
                # PPO likelihood ratios must be evaluated by the same
                # deterministic network used to collect old log-probabilities.
                'dropout_p': 0.0,
                'device': device(),
            }
        }

    if model_method == 'GCN_DoubleDQN':
        return {
            'GCN_DoubleDQN': {
                'n_features_aug': augmented_feature_dim,
                'dens_obs': DENS_OBS,
                'hidden': HIDDEN,
                'learning_rate': 0.001,
                'lr_gamma': 0.9999954,
                'gamma': 0.9,
                'epsilon': 0.1,
                'target_update': 1000,
                'device': device(),
            }
        }

    raise ValueError(f'Unsupported model_method: {model_method}')


def add_model_path(model_params, model_method, model_path):
    params = {model_method: dict(model_params[model_method])}
    params[model_method]['model_path'] = model_path
    return params


def build_trainer_params(model_id, model_method, graph_types, n_nodes_list,
                         num_graphs=TRAIN_GRAPHS_PER_GROUP, no_sd=False,
                         num_episodes=5000, prompt_record=False,
                         feature_set=utils.DEFAULT_GRAPH_FEATURE_SET,
                         seed=None):
    feature_set = utils.normalize_feature_set(feature_set)
    feature_names = utils.get_graph_feature_names(feature_set)
    static_feature_dim = utils.get_static_graph_feature_dim(feature_set)
    augmented_feature_dim = utils.get_augmented_node_feature_dim(feature_set)
    sd_training_record = (
        '' if no_sd else
        ', delay_sampling=uniform_per_episode_across_numbered_realizations'
    )
    train_data_groups = build_data_groups(
        TRAIN_DATA_PATH,
        graph_types,
        n_nodes_list,
        num_graphs,
        no_sd=no_sd,
        feature_set=feature_set,
        sample_delay_realizations=(model_method == 'PPOAgent' and not no_sd),
    )
    train_model = {'num_episodes': num_episodes}
    if model_method == 'PPOAgent':
        train_model['update_timestep'] = 512
    elif model_method == 'GCN_DoubleDQN':
        train_model.update({
            'replay_buffer_params': {
                'buffer_size': 6000000,
                'replay_buffer_type': 'PriorityReplayBuffer',
            },
            'minimal_size': 500,
            'batch_size': 128,
            'G_num': num_graphs,
            'num_sample': 1,
        })
    else:
        raise ValueError(f'Unsupported model_method: {model_method}')

    return {
        'model_method': model_method,
        'train_data_groups': train_data_groups,
        'train_model': train_model,
        'model_save_path': f'model_save/{model_id}/{model_id}.pt',
        'results_save': {
            'save_path': f'model_save/{model_id}/',
        },
        'prompt_record': prompt_record,
        'experiment_record': (
            f'{model_id}: {model_method}, graph_types={graph_types}, '
            f'n_nodes={n_nodes_list}, num_graphs={num_graphs}, '
            f'no_sd={no_sd}, seed={seed}, '
            f'sensor_budgets={{{", ".join(f"{n}: {get_sensor_budget(n)}" for n in n_nodes_list)}}}, '
            f'feature_set={feature_set}, '
            f'feature_names={feature_names}, '
            f'static_feature_dim={static_feature_dim}, '
            f'dynamic_feature_dim={utils.DYNAMIC_NODE_FEATURE_DIM}, '
            f'augmented_feature_dim={augmented_feature_dim}, '
            f'reward_normalization=2*rank_delta/(n_nodes-1)'
            f'{sd_training_record}'
        ),
    }


def build_method_params(model_id, model_method, include_baselines=True,
                        include_neural=True, beta=4, gamma=4, fixed_k=None,
                        feature_set=utils.DEFAULT_GRAPH_FEATURE_SET,
                        neural_methods=None, sd_score_mode='hop',
                        search_scenarios=12, search_seed=42,
                        sd_prior='equal_lmh', delay_mu=1.0,
                        known_sd_level='all',
                        topology_prefix=None, run_seed=None):
    feature_set = utils.normalize_feature_set(feature_set)
    if neural_methods is None:
        neural_methods = ['one_shot', 'SGBS']
    unknown_methods = set(neural_methods) - {'one_shot', 'SGBS'}
    if unknown_methods:
        raise ValueError(
            f'Unsupported neural test methods: {sorted(unknown_methods)}'
        )
    test_methods = list(neural_methods) if include_neural else []
    if include_baselines:
        test_methods = ['random', 'BC', 'High_Coverage_Rate', 'CB', 'HV_OBS'] + test_methods

    model_path = f'model_save/{model_id}/{model_id}.pt'
    model_params = build_model_params(model_method, feature_set=feature_set)
    fixed_k_param = {} if fixed_k is None else {'fixed_k': fixed_k}
    neural_params = {
        'feature_set': feature_set,
        'run_seed': run_seed,
        **fixed_k_param,
    }

    return {
        'test_methods': test_methods,
        'model_methods': [model_method],
        'test_init_params': {
            'one_shot': {'need_model': True, **neural_params},
            'SGBS': {
                'beta': beta,
                'gamma': gamma,
                'sd_score_mode': sd_score_mode,
                'search_scenarios': search_scenarios,
                'search_seed': search_seed,
                'sd_prior': sd_prior,
                'delay_mu': delay_mu,
                'known_sd_level': known_sd_level,
                'topology_prefix': topology_prefix,
                'run_seed': run_seed,
                'need_model': True,
                **neural_params,
            },
            'BC': {'need_model': False, **fixed_k_param},
            'High_Coverage_Rate': {'need_model': False, **fixed_k_param},
            'CB': {'need_model': False, **fixed_k_param},
            'HV_OBS': {'need_model': False, 'L': 4, **fixed_k_param},
            'K_Median': {'need_model': False, **fixed_k_param},
            'random': {'need_model': False, **fixed_k_param},
            'GA': {'need_model': False, **fixed_k_param},
        },
        'model_init_params': add_model_path(model_params, model_method, model_path),
    }


def build_tester_params(model_id, graph_type, n_nodes,
                        num_graphs=TEST_GRAPHS_PER_GROUP, no_sd=False, fixed_k=None,
                        feature_set=utils.DEFAULT_GRAPH_FEATURE_SET,
                        data_path=TEST_DATA_PATH, results_save_path=None,
                        sd_score_mode='hop', search_scenarios=12,
                        search_seed=42, known_sd_level='all'):
    feature_set = utils.normalize_feature_set(feature_set)
    fixed_k_suffix = '' if fixed_k is None else f'_k{fixed_k}'
    delay_suffix = '' if no_sd else '_delay'
    if sd_score_mode == 'hop':
        score_suffix = ''
    elif sd_score_mode == 'mc':
        score_suffix = f'_mc_s{search_scenarios}_seed{search_seed}'
    else:
        score_suffix = (
            f'_pmc_{known_sd_level}_s{search_scenarios}_seed{search_seed}'
        )
    return {
        'test_data': {
            'data_path': data_path,
            'graph_type': graph_type,
            'n_nodes': n_nodes,
            'num_graphs': num_graphs,
            'no_sd': no_sd,
            'load_features': False,
            'feature_set': feature_set,
            # SD tests evaluate every numbered delay realization while
            # loading only one topology's matrices at a time in OP_Env.
            'load_all_delay_realizations': not no_sd,
            'lazy_delay_realizations': not no_sd,
        },
        'results_save_path': results_save_path or (
            f'result/{model_id}/{model_id}_{graph_type}_{n_nodes}'
            f'{fixed_k_suffix}{delay_suffix}{score_suffix}.json'
        ),
    }
