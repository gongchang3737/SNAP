# -*- coding: utf-8 -*-
import utils
import torch
import OP_Model
import OP_Env
import os
import numpy as np
from tqdm import tqdm

class _LegacyOPTrainer:
    def __init__(self, trainer_params, model_params, env_params):
        self.trainer_params = trainer_params
        self.model_params = model_params
        self.env_params = env_params

    def run(self):
        train_data = utils.data_loader(**self.trainer_params['train_data'])
        device = self.model_params['PPOAgent']['device']
        
        # 预处理图数据 Tensor
        features_tensor = torch.cat([torch.tensor(f, dtype=torch.float).unsqueeze(0) for f in train_data['features']], 0).to(device)
        G_matrix_diag_tensor = torch.cat([torch.tensor(m, dtype=torch.float).unsqueeze(0) for m in train_data['G_matrix_diag']], 0).to(device)

        model = OP_Model.PPOAgent(**self.model_params['PPOAgent'])
        env = OP_Env.OP_Env_Precision(**self.env_params['OP_Env_Precision'])

        G_num = self.trainer_params['train_model']['G_num']
        num_episodes = self.trainer_params['train_model']['num_episodes']
        update_timestep = self.trainer_params['train_model']['update_timestep']

        return_list_all, loss_list_all = [], []
        
        # PPO 临时 Rollout Buffer
        transition_dict = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': [], 'next_states': [], 'b_indices': []}
        time_step = 0

        with tqdm(total=G_num * num_episodes, desc="PPO Training") as pbar:
            for i_episode in range(G_num * num_episodes):
                G_index = np.random.randint(0, G_num)
                G = train_data['G'][G_index]
                env.get_graph(G, train_data['G_matrix_diag'][G_index], train_data['G_distance'][G_index], train_data['G_distance_delay'][G_index])

                state = env.reset()
                done = False
                episode_return = 0

                while not done:
                    time_step += 1
                    action, log_prob, value = model.take_action(state, train_data['features'][G_index], train_data['G_matrix_diag'][G_index])
                    next_state, reward, done, _ = env.step(state, action)

                    # 存入 Rollout Buffer
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['log_probs'].append(log_prob)
                    transition_dict['rewards'].append(reward)
                    transition_dict['values'].append(value)
                    transition_dict['dones'].append(done)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['b_indices'].append(G_index)

                    state = next_state
                    episode_return += reward

                    # 达到步数阈值，执行 PPO 更新
                    if time_step % update_timestep == 0:
                        loss = model.update(transition_dict, features_tensor, G_matrix_diag_tensor)
                        loss_list_all.append(loss)
                        # 更新完毕后清空 Buffer
                        transition_dict = {k: [] for k in transition_dict}

                return_list_all.append(episode_return)
                pbar.update(1)

        save_path = self.trainer_params['results_save']['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.ac_net.state_dict(), self.trainer_params['model_save_path'])
        
        train_log = (return_list_all, loss_list_all) 
        utils.plot_training_results(train_log, save_path)


class OP_Trainer:
    """Experiment-ready trainer for PPOAgent and GCN_DoubleDQN."""

    def __init__(self, trainer_params, model_params, env_params):
        self.trainer_params = trainer_params
        self.model_params = model_params
        self.env_params = env_params

    def run(self):
        model_method = self.trainer_params.get('model_method', 'PPOAgent')
        if model_method == 'PPOAgent':
            return self._run_ppo()
        if model_method == 'GCN_DoubleDQN':
            return self._run_double_dqn()
        raise ValueError(f'Unsupported model_method: {model_method}')

    def _load_train_groups(self):
        if 'train_data_groups' in self.trainer_params:
            data_groups = self.trainer_params['train_data_groups']
        else:
            data_groups = [self.trainer_params['train_data']]

        groups = []
        for group_id, data_params in enumerate(data_groups):
            data = utils.data_loader(**data_params)
            realization_counts = [
                len(realizations) if realizations is not None else 1
                for realizations in data.get('G_distance_delay_realizations', [])
            ]
            if not data_params.get('no_sd', False):
                unique_counts = sorted(set(realization_counts))
                if len(unique_counts) != 1:
                    raise ValueError(
                        f'Every topology in one training group must have the '
                        f'same number of delay realizations; got {unique_counts} '
                        f'for {data_params["graph_type"]}-{data_params["n_nodes"]}'
                    )
                delay_usage = np.zeros(unique_counts[0], dtype=np.int64)
            else:
                delay_usage = None
            groups.append({
                'id': group_id,
                'params': data_params,
                'data': data,
                'num_graphs': len(data['G']),
                'delay_usage': delay_usage,
            })
        return groups

    @staticmethod
    def _episode_delay_distance(group, graph_index):
        """Choose one SD realization without changing the no-SD RNG path."""
        train_data = group['data']
        if group['params'].get('no_sd', False):
            return train_data['G_distance_delay'][graph_index]

        realizations = train_data['G_distance_delay_realizations'][graph_index]
        if not realizations:
            raise ValueError(
                'SD PPO training requires numbered delay realizations. '
                f'No realizations loaded for graph index {graph_index}.'
            )
        realization_position = np.random.randint(0, len(realizations))
        group['delay_usage'][realization_position] += 1
        return realizations[realization_position]

    @staticmethod
    def _empty_ppo_buffer():
        return {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'next_states': [],
            'b_indices': [],
        }

    def _run_ppo(self):
        groups = self._load_train_groups()
        device = self.model_params['PPOAgent']['device']

        for group in groups:
            data = group['data']
            group['features_tensor'] = torch.cat(
                [torch.tensor(f, dtype=torch.float).unsqueeze(0) for f in data['features']], 0
            ).to(device)
            group['G_matrix_diag_tensor'] = torch.cat(
                [torch.tensor(m, dtype=torch.float).unsqueeze(0) for m in data['G_matrix_diag']], 0
            ).to(device)

        model = OP_Model.PPOAgent(**self.model_params['PPOAgent'])
        env = OP_Env.OP_Env_Precision(**self.env_params['OP_Env_Precision'])

        num_episodes = self.trainer_params['train_model']['num_episodes']
        update_timestep = self.trainer_params['train_model']['update_timestep']
        total_episodes = num_episodes * len(groups)

        return_list_all, loss_list_all = [], []
        episodes_completed = 0
        group_episode_counts = {group['id']: 0 for group in groups}

        # PPO is on-policy: collect one complete rollout batch from one graph
        # group and update immediately.  The former per-group persistent
        # buffers became stale while other groups updated the shared policy.
        with tqdm(total=total_episodes, desc='PPO Training') as pbar:
            while episodes_completed < total_episodes:
                unfinished_groups = [
                    group for group in groups
                    if group_episode_counts[group['id']] < num_episodes
                ]
                # Keep progress balanced across groups while respecting the
                # exact requested episode count for every graph type/size.
                minimum_progress = min(
                    group_episode_counts[group['id']] / num_episodes
                    for group in unfinished_groups
                )
                least_trained_groups = [
                    group for group in unfinished_groups
                    if np.isclose(
                        group_episode_counts[group['id']] / num_episodes,
                        minimum_progress,
                    )
                ]
                group = least_trained_groups[
                    np.random.randint(0, len(least_trained_groups))
                ]
                train_data = group['data']
                buffer = self._empty_ppo_buffer()

                while (
                    len(buffer['states']) < update_timestep
                    and group_episode_counts[group['id']] < num_episodes
                ):
                    G_index = np.random.randint(0, group['num_graphs'])
                    episode_delay_distance = self._episode_delay_distance(
                        group,
                        G_index,
                    )
                    env.get_graph(
                        train_data['G'][G_index],
                        train_data['G_matrix_diag'][G_index],
                        train_data['G_distance'][G_index],
                        episode_delay_distance,
                    )

                    state = env.reset()
                    done = False
                    episode_return = 0
                    while not done:
                        action, log_prob, value = model.take_action(
                            state,
                            train_data['features'][G_index],
                            train_data['G_matrix_diag'][G_index],
                        )
                        next_state, reward, done, _ = env.step(state, action)

                        buffer['states'].append(state)
                        buffer['actions'].append(action)
                        buffer['log_probs'].append(log_prob)
                        buffer['rewards'].append(reward)
                        buffer['values'].append(value)
                        buffer['dones'].append(done)
                        buffer['next_states'].append(next_state)
                        buffer['b_indices'].append(G_index)

                        state = next_state
                        episode_return += reward

                    return_list_all.append(episode_return)
                    episodes_completed += 1
                    group_episode_counts[group['id']] += 1
                    pbar.update(1)

                if buffer['states']:
                    loss = model.update(
                        buffer,
                        group['features_tensor'],
                        group['G_matrix_diag_tensor'],
                    )
                    loss_list_all.append(loss)

        print('PPO episodes per group:')
        for group in groups:
            params = group['params']
            print(
                f"  {params['graph_type']}-{params['n_nodes']}: "
                f"{group_episode_counts[group['id']]}"
            )
            if group['delay_usage'] is not None:
                usage = ', '.join(
                    f'{index}:{int(count)}'
                    for index, count in enumerate(group['delay_usage'])
                )
                print(f'    delay realization usage: {usage}')

        save_path = self.trainer_params['results_save']['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.ac_net.state_dict(), self.trainer_params['model_save_path'])

        train_log = (return_list_all, loss_list_all)
        utils.plot_training_results(
            train_log,
            save_path,
            experiment_record=self.trainer_params.get('experiment_record', ''),
            prompt_record=self.trainer_params.get('prompt_record', True),
        )
        return train_log

    def _run_double_dqn(self):
        groups = self._load_train_groups()
        model_params = self.model_params['GCN_DoubleDQN']
        device = model_params['device']
        model = OP_Model.GCN_DoubleDQN(**model_params)
        env = OP_Env.OP_Env_Precision(**self.env_params['OP_Env_Precision'])

        return_list_all, loss_list_all = [], []
        train_model_params = self.trainer_params['train_model']

        for group in groups:
            data = group['data']
            group_params = dict(train_model_params)
            group_params.setdefault('num_sample', 1)
            group_params['G_num'] = min(group_params.get('G_num', group['num_graphs']), group['num_graphs'])
            group_params['device'] = device
            train_log = model.train(env, data, **group_params)
            return_list_all.extend(train_log[0])
            loss_list_all.extend(train_log[3])

        save_path = self.trainer_params['results_save']['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), self.trainer_params['model_save_path'])

        train_log = (return_list_all, loss_list_all)
        utils.plot_training_results(
            train_log,
            save_path,
            experiment_record=self.trainer_params.get('experiment_record', ''),
            prompt_record=self.trainer_params.get('prompt_record', True),
        )
        return train_log
