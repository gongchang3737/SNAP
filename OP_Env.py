import numpy as np
import networkx as nx
from sourceLocalizationClass import SingleSourceLocalization
from utils import getAllIndex
from functools import reduce
import copy
import time
from sensor_budget_config import get_sensor_budget
from graph_indexing import (
    validate_delay_distance,
    validate_graph_index_space,
    validate_index_sequence,
)
from sd_scenarios import (
    build_search_score_context,
    conditional_delay_prior,
    delay_prior_from_preset,
)

'''
用来存储各类深度强化学习环境
'''


def _validate_available_action(state, action, n_nodes):
    action = validate_index_sequence(
        [action], n_nodes, name='PPO environment action'
    )[0]
    if len(state) != n_nodes:
        raise ValueError(
            f'PPO state length {len(state)} does not match N={n_nodes}'
        )
    if state[action] != 0:
        raise ValueError(f'PPO action {action} selects an existing sensor')
    return action

class OP_Env_PLO:
    def __init__(self, dens_obs, L):
        self.dens_obs = dens_obs # 传感器密度
        self.L = L

    def get_graph(self, G, G_matrix_diag, G_distance, G_distance_delay):
        validate_graph_index_space(
            G,
            [
                np.asarray(G_matrix_diag).shape,
                np.asarray(G_distance).shape,
                np.asarray(G_distance_delay).shape,
            ],
            context='PLO environment',
        )
        self.G = G
        self.G_matrix_diag = G_matrix_diag
        self.G_distance = G_distance
        self.G_distance_delay = G_distance_delay

        self.N = len(self.G) #节点数
        self.step_max = get_sensor_budget(self.N) # 固定传感器数量
        self.cover_pre = dict(zip(range(self.N), np.zeros(self.N))) #为了降低get_reward的计算复杂度，存储上一步状态对应的指标评估值(节点覆盖情况)
        self.return_pre = 0

    # 计算前后状态的奖励 2024121903
    def get_reward(self, now_state, next_state): # PL(O)作为奖励
        place_obs_lst_now = getAllIndex(now_state,1)
        # print('place_obs_lst_now', place_obs_lst_now)
        if place_obs_lst_now == []: #此时为第一步，布设一个传感器无法溯源
            return 0
        else:
            place_obs_lst_next = getAllIndex(next_state,1)
            # print('place_obs_lst_next', place_obs_lst_next)
            place_index = list(set(place_obs_lst_next).difference(set(place_obs_lst_now))) #获取这一步布设节点的序号

            new_nodes = [nx.dijkstra_path(self.G, place_index[0], obs_i) for obs_i in place_obs_lst_now]
            new_nodes = [path_i for path_i in new_nodes if len(path_i) <= self.L]
            # print('new_nodes', new_nodes)
            if new_nodes == []: #如果没有一条新的符合条件的路径，那么直接返回reward=0,不修改cover_pre
                return 0
            # 如果存在新的符合条件的路径，计算reward，并且更新cover_pre
            new_nodes = reduce(lambda x,y:x+y,new_nodes)
            cover_now = copy.deepcopy(self.cover_pre)
            for node_i in new_nodes:
                cover_now[node_i] += 1

            min_cover_now = min(cover_now.values())
            if min_cover_now == 0: #如果最小值是0，那么直接计算有多少值不为0即可（降低低密度传感器下的计算时间）
                return_now = np.sum(np.array(list(cover_now.values()), dtype=bool))
            else: #如果最小值不是0，计算大于最小值的数量，再加上节点数量*最小值
                return_now = np.sum(np.array(list(cover_now.values()))>min_cover_now) + len(cover_now)*min_cover_now

            reward = return_now - self.return_pre
            # print('reward', reward)
            self.cover_pre = cover_now
            self.return_pre = return_now

            # key = input('输入1继续，0退出:')
            # if key == '0':
            #     exit(0)

            return reward

    # 走一步函数，这个是最关键的了
    def step(self, state, action):
        action = _validate_available_action(state, action, self.N)
        now_state = state   # 获取当前状态
        # print('now_state:', now_state)
        next_state = now_state.copy()
        next_state[action] = 1
        reward = self.get_reward(now_state, next_state)  # reward function
        # print('reward', reward)
        done = 0.0    # done用来表示传感器是否布设完毕
        if sum(next_state) >= self.step_max: #这里是大于还是大于等于？？
            done = 1.0

        return next_state, reward, done

    def reset(self):
        state = [0]*self.N
        self.cover_pre = dict(zip(range(len(self.G)), np.zeros(len(self.G))))
        self.return_pre = 0

        return state

    def update(self,state): #什么作用？？
        # self.G = from_state_to_network(state)
        return state


class OP_Env_Precision:

    def __init__(self, dens_obs, sample_method, spread_method, locate_strategy, begin_shaping, shaping_param):
        self.dens_obs = dens_obs # 传感器密度
        self.sample_method = sample_method
        self.spread_method = spread_method
        self.locate_strategy = locate_strategy
        self.begin_shaping = begin_shaping
        self.shaping_param = shaping_param

    def get_graph(self, G, G_matrix_diag, G_distance, G_distance_delay): # 针对特定图的初始化
        validate_graph_index_space(
            G,
            [
                np.asarray(G_matrix_diag).shape,
                np.asarray(G_distance).shape,
                np.asarray(G_distance_delay).shape,
            ],
            context='precision environment',
        )
        self.G = G
        self.G_matrix_diag = G_matrix_diag
        self.G_distance = G_distance
        self.G_distance_delay = G_distance_delay

        self.episode = 0 #该图环境已经训练的轮数
        self.episode_return_raw = 0
        self.best_return_raw = 0 #当前最优的总奖励，每一张图训练结束后清零
        self.N = len(self.G) #节点数
        self.step_max = get_sensor_budget(self.N) # 固定传感器数量
        self.cover_pre = dict(zip(range(self.N), np.zeros(self.N))) #为了降低get_reward的计算复杂度，存储上一步状态对应的指标评估值(节点覆盖情况)
        self.return_pre = 0
        self.value_pre = self.N/2 #为了降低get_reward的计算复杂度，存储上一步状态对应的指标评估值，初始值为图上所有节点排序的均值（即随机选择时的期望排序）

    # 计算前后状态的奖励 2024121902
    def get_reward(self, now_state, next_state): # average index作为奖励
        n_nodes = self.N
        place_obs_lst_now = getAllIndex(now_state,1)
        place_obs_lst_next = getAllIndex(next_state,1)
        if place_obs_lst_now == []: #此时为第一步，布设一个传感器无法溯源
            return 0
        else:
            SL_Model_train = SingleSourceLocalization(self.G, self.G_matrix_diag)
            source_actual_lst = SL_Model_train.sample_source_lst(n_source=n_nodes, sample_method=self.sample_method)
            index_lst_next = SL_Model_train.rank_all_sources(
                source_actual_lst,
                place_obs_lst_next,
                self.G_distance_delay,
                self.spread_method,
                self.G_distance,
                self.locate_strategy,
                chunk_size=min(64, n_nodes),
            )
            value_now = np.average(index_lst_next)
            # Normalize mixed graph sizes while preserving the final-rank
            # objective when gamma == 1.
            rank_span = max(n_nodes - 1, 1)
            reward = 2.0 * (self.value_pre - value_now) / rank_span
            self.value_pre = value_now

            return reward

    # 走一步函数，这个是最关键的了
    def step(self, state, action):
        action = _validate_available_action(state, action, self.N)
        now_state = state   # 获取当前状态
        next_state = now_state.copy()
        next_state[action] = 1
        reward = self.get_reward(now_state, next_state)  # reward function
        shaping = 0
        self.episode_return_raw += reward #每一步的奖励累加
        done = 0.0    # done用来表示传感器是否布设完毕
        if sum(next_state) >= self.step_max: #这里是大于还是大于等于？？
            done = 1.0
            self.episode += 1
            opt_gap = self.episode_return_raw - self.best_return_raw
            # print('opt_gap', opt_gap)

            if opt_gap > 0:
                # print('最优解更新**********************')
                self.best_return_raw = self.episode_return_raw #更新当前最优的总奖励
            if self.episode > self.begin_shaping: #要求执行一定轮数，才开始加入reward shaping，避免过早加入reward shaping导致初始抽样的经验的奖励值过高
                # print('reward shaping')
                shaping = self.shaping_param * opt_gap # reward shaping，引导agent寻找最优解
                

        return next_state, reward, done, shaping

    def reset(self):
        state = [0]*self.N
        self.episode_return_raw = 0
        self.value_pre = len(self.G)/2

        return state

    @staticmethod
    def _sd_group(sd_value):
        if sd_value is None or not np.isfinite(sd_value):
            return None
        if 0.0 < sd_value <= 0.1:
            return 'low'
        if 0.1 < sd_value <= 0.3:
            return 'medium'
        if 0.3 < sd_value <= 0.5:
            return 'high'
        return 'out_of_range'

    @staticmethod
    def _delay_realizations_for_graph(test_data, graph_index):
        """Materialize one topology's delay realizations for evaluation."""
        path_groups = test_data.get('G_distance_delay_realization_paths')
        index_groups = test_data.get('delay_realization_indices')
        sd_groups = test_data.get('delay_sd_values')
        paths = path_groups[graph_index] if path_groups else []
        indices = index_groups[graph_index] if index_groups else []
        sd_values = sd_groups[graph_index] if sd_groups else []

        if paths:
            matrices = [
                np.asarray(np.loadtxt(path), dtype=float)
                for path in paths
            ]
        else:
            realization_groups = test_data.get('G_distance_delay_realizations')
            loaded = (
                realization_groups[graph_index]
                if realization_groups else None
            )
            if loaded:
                matrices = loaded
            else:
                matrices = [test_data['G_distance_delay'][graph_index]]

        if not indices:
            indices = [None] * len(matrices)
        if not sd_values:
            sd_values = [None] * len(matrices)
        if not (len(matrices) == len(indices) == len(sd_values)):
            raise ValueError(
                f'Delay realization metadata mismatch for graph {graph_index}: '
                f'{len(matrices)} matrices, {len(indices)} indices, '
                f'{len(sd_values)} SD values'
            )

        hop_distance = test_data['G_distance'][graph_index]
        expected_shape = hop_distance.shape
        for position, (matrix, sd_value) in enumerate(zip(matrices, sd_values)):
            if matrix.shape != expected_shape:
                raise ValueError(
                    f'Delay matrix for graph {graph_index} has shape '
                    f'{matrix.shape}; expected {expected_shape}'
                )
            validate_delay_distance(
                matrix,
                hop_distance,
                sd=(
                    float(sd_value)
                    if sd_value is not None and np.isfinite(sd_value)
                    else None
                ),
                context=(
                    paths[position]
                    if position < len(paths)
                    else f'graph_{graph_index}/delay_{position}'
                ),
            )
        return list(zip(indices, sd_values, matrices))

    def for_test(self, test_data, test_methods, model_methods, test_init_params, model_init_params):
        test_log = {}
        G_num = len(test_data['G'])
        for model_method in model_methods:
            test_log[model_method] = {}
            for place_strategy in test_methods:
                print('place_strategy', place_strategy)
                method_test_params = test_init_params[place_strategy]
                sd_score_mode = (
                    method_test_params.get('sd_score_mode', 'hop')
                    if place_strategy == 'SGBS' else 'not_applicable'
                )
                known_sd_level = method_test_params.get(
                    'known_sd_level', 'all'
                )
                conditional_levels = (
                    ['low', 'medium', 'high']
                    if sd_score_mode == 'pmc' and known_sd_level == 'all'
                    else [known_sd_level]
                    if sd_score_mode == 'pmc'
                    else [None]
                )
                test_log[model_method][place_strategy] = {
                    'ave_rank': 0,
                    'ave_time': 0,
                    'rank': [],
                    'time': [],
                    # Backward-compatible detailed timing.  ``time`` and
                    # ``ave_time`` remain the original end-to-end metrics.
                    'model_load_time': [],
                    'feature_time': [],
                    'search_time': [],
                    'search_scenario_generation_time': [],
                    'search_scoring_time': [],
                    'deployment_time': [],
                    'placement_time': [],
                    'evaluation_time': [],
                    'ave_model_load_time': 0,
                    'ave_feature_time': 0,
                    'ave_search_time': 0,
                    'ave_search_scenario_generation_time': 0,
                    'ave_search_scoring_time': 0,
                    'ave_deployment_time': 0,
                    'ave_placement_time': 0,
                    'ave_evaluation_time': 0,
                    # One entry per topology, containing one rank per delay
                    # realization after averaging stochastic placements.
                    'rank_by_realization': [],
                    'realization_records': [],
                    'delay_realizations_per_topology': [],
                    'num_topologies': G_num,
                    'num_delay_evaluations': 0,
                    'placement_information': (
                        'topology_and_independent_delay_prior'
                        if sd_score_mode == 'mc'
                        else 'topology_and_known_sd_level_prior'
                        if sd_score_mode == 'pmc'
                        else 'topology_and_hop_only'
                    ),
                    'sd_score_mode': sd_score_mode,
                    'known_sd_level': (
                        known_sd_level if sd_score_mode == 'pmc' else None
                    ),
                    'search_config': (
                        {
                            'scenarios': method_test_params['search_scenarios'],
                            'seed': method_test_params['search_seed'],
                            'prior': (
                                method_test_params['sd_prior']
                                if sd_score_mode == 'mc'
                                else 'conditional_by_known_sd_level'
                            ),
                            'mu': method_test_params['delay_mu'],
                            'uses_test_delay_or_metadata': False,
                            'uses_actual_test_sd': False,
                            'known_sd_level': (
                                known_sd_level
                                if sd_score_mode == 'pmc' else None
                            ),
                            'conditional_intervals': (
                                {
                                    'low': '(0, 0.1]',
                                    'medium': '(0.1, 0.3]',
                                    'high': '(0.3, 0.5]',
                                }
                                if sd_score_mode == 'pmc' else None
                            ),
                            'sources': 'all_nodes',
                        }
                        if sd_score_mode in ('mc', 'pmc') else None
                    ),
                    'search_pool_fingerprints': [],
                    'search_pool_fingerprints_by_sd_level': [],
                    'placements_by_topology': [],
                    'placements_by_sd_level': [],
                    # For pmc/all, generic timing fields are averages over
                    # one real deployment (one known level).  These fields
                    # separately record the cost of executing all requested
                    # conditional deployments for the benchmark.
                    'conditional_total_deployment_time': [],
                    'conditional_total_evaluation_time': [],
                    'conditional_total_time': [],
                    'ave_conditional_total_deployment_time': None,
                    'ave_conditional_total_evaluation_time': None,
                    'ave_conditional_total_time': None,
                    'pmc_metrics_by_sd_level': {
                        level: {
                            'deployment_time': [],
                            'evaluation_time': [],
                            'time': [],
                            'ave_deployment_time': None,
                            'ave_evaluation_time': None,
                            'ave_time': None,
                        }
                        for level in ('low', 'medium', 'high')
                    },
                    'sd_groups': {
                        'low': {
                            'range': '(0, 0.1]', 'ave_rank': None, 'rank': []
                        },
                        'medium': {
                            'range': '(0.1, 0.3]', 'ave_rank': None, 'rank': []
                        },
                        'high': {
                            'range': '(0.3, 0.5]', 'ave_rank': None, 'rank': []
                        },
                        'out_of_range': {
                            'range': 'outside (0, 0.5]',
                            'ave_rank': None,
                            'rank': [],
                        },
                    },
                }
                for G_index in range(G_num):
                    print('第{0}个图'.format(G_index))
                    G = test_data['G'][G_index]
                    G_matrix = test_data['G_matrix'][G_index]
                    G_matrix_diag = test_data['G_matrix_diag'][G_index]
                    G_distance = test_data['G_distance'][G_index]
                    # Initialize only from topology/hops.  Final test delay
                    # matrices are deliberately materialized after every
                    # deployment for this topology has completed.
                    self.get_graph(
                        G,
                        G_matrix_diag,
                        G_distance,
                        G_distance,
                    )
                    repeat = 1 if place_strategy in ['BC', 'High_Coverage_Rate', 'CB', 'HV_OBS','one_shot','SGBS'] else 100
                    placements = []
                    for deployment_level in conditional_levels:
                        for _ in range(repeat):
                            tic = time.perf_counter()
                            search_score_context = None
                            if (
                                place_strategy == 'SGBS'
                                and sd_score_mode in ('mc', 'pmc')
                            ):
                                prior = (
                                    delay_prior_from_preset(
                                        method_test_params['sd_prior'],
                                        mu=method_test_params['delay_mu'],
                                    )
                                    if sd_score_mode == 'mc'
                                    else conditional_delay_prior(
                                        deployment_level,
                                        mu=method_test_params['delay_mu'],
                                    )
                                )
                                topology_prefix = (
                                    method_test_params.get('topology_prefix')
                                    or f'graph_size_{self.N}'
                                )
                                level_suffix = (
                                    '' if deployment_level is None
                                    else f'/pmc_{deployment_level}'
                                )
                                search_score_context = build_search_score_context(
                                    G_matrix,
                                    G_distance,
                                    topology_id=(
                                        f'{topology_prefix}/graph_{G_index}'
                                        f'{level_suffix}'
                                    ),
                                    count=method_test_params['search_scenarios'],
                                    seed=method_test_params['search_seed'],
                                    prior=prior,
                                    spread_method=self.spread_method,
                                    locate_strategy=self.locate_strategy,
                                )
                            SL_Model = SingleSourceLocalization(G, G_matrix_diag)
                            source_actual_lst = SL_Model.sample_source_lst(
                                n_source=self.N,
                                sample_method=self.sample_method,
                            )
                            # MC/pMC receive independent search contexts that
                            # contain no final-test realization or actual SD.
                            place_obs_lst = SL_Model.place_observers(
                                self.dens_obs,
                                place_strategy,
                                model_method,
                                test_init_params[place_strategy],
                                model_init_params[model_method],
                                source_actual_lst,
                                G_distance,
                                self.spread_method,
                                G_distance,
                                self.locate_strategy,
                                search_score_context,
                            )
                            deployment_wall_elapsed = time.perf_counter() - tic
                            placements.append((
                                SL_Model,
                                source_actual_lst,
                                place_obs_lst,
                                deployment_wall_elapsed,
                                search_score_context,
                                deployment_level,
                            ))

                    delay_realizations = self._delay_realizations_for_graph(
                        test_data, G_index
                    )
                    rank_by_realization = [
                        [] for _ in delay_realizations
                    ]
                    for (
                        SL_Model,
                        source_actual_lst,
                        place_obs_lst,
                        deployment_wall_elapsed,
                        search_score_context,
                        deployment_level,
                    ) in placements:
                        evaluation_start = time.perf_counter()
                        for realization_position, (_, sd_value, delay_matrix) in enumerate(
                            delay_realizations
                        ):
                            realization_level = self._sd_group(sd_value)
                            if (
                                sd_score_mode == 'pmc'
                                and realization_level != deployment_level
                            ):
                                continue
                            rank_lst_nodes = SL_Model.rank_all_sources(
                                source_actual_lst,
                                place_obs_lst,
                                delay_matrix,
                                self.spread_method,
                                G_distance,
                                self.locate_strategy,
                            )
                            rank_by_realization[realization_position].append(
                                float(np.average(rank_lst_nodes))
                            )
                        evaluation_elapsed = time.perf_counter() - evaluation_start
                        test_log[model_method][place_strategy]['time'].append(
                            deployment_wall_elapsed + evaluation_elapsed
                        )
                        test_log[model_method][place_strategy]['model_load_time'].append(
                            SL_Model.last_model_load_time
                        )
                        test_log[model_method][place_strategy]['placement_time'].append(
                            SL_Model.last_placement_time
                        )
                        test_log[model_method][place_strategy]['feature_time'].append(
                            SL_Model.last_feature_time
                        )
                        test_log[model_method][place_strategy]['search_time'].append(
                            SL_Model.last_search_time
                        )
                        test_log[model_method][place_strategy]['search_scenario_generation_time'].append(
                            SL_Model.last_search_scenario_generation_time
                        )
                        test_log[model_method][place_strategy]['search_scoring_time'].append(
                            SL_Model.last_search_scoring_time
                        )
                        test_log[model_method][place_strategy]['deployment_time'].append(
                            deployment_wall_elapsed
                        )
                        test_log[model_method][place_strategy]['evaluation_time'].append(
                            evaluation_elapsed
                        )
                        if sd_score_mode == 'pmc':
                            level_metrics = test_log[model_method][place_strategy][
                                'pmc_metrics_by_sd_level'
                            ][deployment_level]
                            level_metrics['deployment_time'].append(
                                deployment_wall_elapsed
                            )
                            level_metrics['evaluation_time'].append(
                                evaluation_elapsed
                            )
                            level_metrics['time'].append(
                                deployment_wall_elapsed + evaluation_elapsed
                            )
                        if search_score_context is not None:
                            test_log[model_method][place_strategy]['search_pool_fingerprints'].append(
                                search_score_context.scenario_pool.pool_fingerprint
                            )
                    realization_means = [
                        float(np.average(values)) if values else None
                        for values in rank_by_realization
                    ]
                    method_log = test_log[model_method][place_strategy]
                    method_log['placements_by_topology'].append([
                        list(placement[2]) for placement in placements
                    ])
                    method_log['placements_by_sd_level'].append({
                        placement[5]: list(placement[2])
                        for placement in placements
                        if placement[5] is not None
                    })
                    method_log['search_pool_fingerprints_by_sd_level'].append({
                        placement[5]: placement[4].scenario_pool.pool_fingerprint
                        for placement in placements
                        if placement[5] is not None and placement[4] is not None
                    })
                    if sd_score_mode == 'pmc':
                        deployment_total = float(sum(
                            placement[3] for placement in placements
                        ))
                        # Each conditional placement is evaluated only on its
                        # matching final-test SD group.
                        evaluation_total = float(sum(
                            method_log['pmc_metrics_by_sd_level'][level][
                                'evaluation_time'
                            ][-1]
                            for level in conditional_levels
                        ))
                        method_log['conditional_total_deployment_time'].append(
                            deployment_total
                        )
                        method_log['conditional_total_evaluation_time'].append(
                            evaluation_total
                        )
                        method_log['conditional_total_time'].append(
                            deployment_total + evaluation_total
                        )
                    method_log['rank_by_realization'].append(realization_means)
                    method_log['delay_realizations_per_topology'].append(
                        len(delay_realizations)
                    )
                    evaluated_ranks = [
                        value for value in realization_means
                        if value is not None
                    ]
                    if not evaluated_ranks:
                        raise ValueError(
                            f'No final-test realizations matched pmc level '
                            f'{known_sd_level!r} for graph {G_index}'
                        )
                    method_log['rank'].append(float(np.average(evaluated_ranks)))
                    for realization_position, (
                        realization_index, sd_value, _
                    ) in enumerate(delay_realizations):
                        rank_value = realization_means[realization_position]
                        if rank_value is None:
                            continue
                        sd_group = self._sd_group(sd_value)
                        record = {
                            'graph_index': G_index,
                            'realization_index': realization_index,
                            'sd': sd_value,
                            'sd_group': sd_group,
                            'rank': rank_value,
                        }
                        method_log['realization_records'].append(record)
                        if sd_group is not None:
                            method_log['sd_groups'][sd_group]['rank'].append(
                                rank_value
                            )

                method_log = test_log[model_method][place_strategy]
                method_log['ave_rank'] = float(np.average(method_log['rank']))
                method_log['ave_time'] = float(np.average(method_log['time']))
                method_log['ave_model_load_time'] = float(np.average(
                    test_log[model_method][place_strategy]['model_load_time']
                ))
                method_log['ave_placement_time'] = float(np.average(
                    test_log[model_method][place_strategy]['placement_time']
                ))
                method_log['ave_feature_time'] = float(np.average(
                    test_log[model_method][place_strategy]['feature_time']
                ))
                method_log['ave_search_time'] = float(np.average(
                    test_log[model_method][place_strategy]['search_time']
                ))
                method_log['ave_search_scenario_generation_time'] = float(np.average(
                    method_log['search_scenario_generation_time']
                ))
                method_log['ave_search_scoring_time'] = float(np.average(
                    method_log['search_scoring_time']
                ))
                method_log['ave_deployment_time'] = float(np.average(
                    method_log['deployment_time']
                ))
                method_log['ave_evaluation_time'] = float(np.average(
                    test_log[model_method][place_strategy]['evaluation_time']
                ))
                method_log['num_delay_evaluations'] = len(
                    method_log['realization_records']
                )
                for group_data in method_log['sd_groups'].values():
                    if group_data['rank']:
                        group_data['ave_rank'] = float(
                            np.average(group_data['rank'])
                        )
                if sd_score_mode == 'pmc':
                    method_log['ave_conditional_total_deployment_time'] = float(
                        np.average(method_log['conditional_total_deployment_time'])
                    )
                    method_log['ave_conditional_total_evaluation_time'] = float(
                        np.average(method_log['conditional_total_evaluation_time'])
                    )
                    method_log['ave_conditional_total_time'] = float(
                        np.average(method_log['conditional_total_time'])
                    )
                    for level, level_metrics in method_log[
                        'pmc_metrics_by_sd_level'
                    ].items():
                        if level_metrics['time']:
                            level_metrics['ave_deployment_time'] = float(
                                np.average(level_metrics['deployment_time'])
                            )
                            level_metrics['ave_evaluation_time'] = float(
                                np.average(level_metrics['evaluation_time'])
                            )
                            level_metrics['ave_time'] = float(
                                np.average(level_metrics['time'])
                            )

        return test_log
    def update(self,state): #什么作用？？
        # self.G = from_state_to_network(state)
        return state

class OP_Env_Precision_2:
    '''
    相比OP_Env_Precision，奖励只在每个episode的最后一步计算，其他步骤的奖励为0
    '''
    def __init__(self, G, G_matrix_diag, G_distance, G_distance_delay, dens_obs):
        validate_graph_index_space(
            G,
            [
                np.asarray(G_matrix_diag).shape,
                np.asarray(G_distance).shape,
                np.asarray(G_distance_delay).shape,
            ],
            context='precision-2 environment',
        )
        self.G = G
        self.G_matrix_diag = G_matrix_diag
        self.G_distance = G_distance
        self.G_distance_delay = G_distance_delay
        self.N = len(self.G)
        self.episode = 0 #该图环境已经训练的轮数
        self.step_max = get_sensor_budget(self.N) # 固定传感器数量
        self.value_pre = self.N/2 #为了降低get_reward的计算复杂度，存储上一步状态对应的指标评估值，初始值为图上所有节点排序的均值（即随机选择时的期望排序）
        self.episode_return_raw = 0 #每一轮的总奖励，每一轮结束后清零，需要写入reset
        self.best_return_raw = 0 #当前最优的总奖励，每一张图训练结束后清零
    def test(self):
        print(self.nodelist)

    # 计算前后状态的奖励
    def get_reward(self, now_state, next_state): # average index作为奖励
        n_nodes = self.N
        place_obs_lst_now = getAllIndex(now_state,1)
        place_obs_lst_next = getAllIndex(next_state,1)
        if place_obs_lst_now == []: #此时为第一步，布设一个传感器无法溯源
            return 0
        else:
            SL_Model_train = SingleSourceLocalization(self.G, self.G_matrix_diag)
            source_actual_lst = SL_Model_train.sample_source_lst(n_source=n_nodes, sample_method=args.sample_method)
            index_lst_next = []
            for source_actual in source_actual_lst:
                inform_next = SL_Model_train.get_inform(source_actual, place_obs_lst_next, self.G_distance_delay, spread_method=args.spread_method)
                source_actual_rank = SL_Model_train.locate_source(source_actual, inform_next, self.G_distance, locate_strategy=args.locate_strategy)
                index_lst_next.append(source_actual_rank)
            value_now = np.average(index_lst_next)
            reward = self.value_pre - value_now
            self.value_pre = value_now

            return reward

    # 走一步函数，这个是最关键的了
    def step(self, state, action):
        action = _validate_available_action(state, action, self.N)
        now_state = state   # 获取当前状态
        next_state = now_state.copy()
        next_state[action] = 1
        reward = 0          
        shaping = 0
        self.episode_return_raw += reward #每一步的奖励累加
        done = 0.0    # done用来表示传感器是否布设完毕
        if sum(next_state) >= self.step_max: #这里是大于还是大于等于？？
            done = 1.0
            reward = self.get_reward(now_state, next_state)  # reward function
            self.episode += 1
            opt_gap = self.episode_return_raw - self.best_return_raw
            print('opt_gap', opt_gap)

            if opt_gap > 0:
                print('最优解更新**********************')
                self.best_return_raw = self.episode_return_raw #更新当前最优的总奖励
            if self.episode > args.begin_shaping: #要求执行一定轮数，才开始加入reward shaping，避免过早加入reward shaping导致初始抽样的经验的奖励值过高
                # print('reward shaping')
                shaping = args.shaping_param * opt_gap # reward shaping，引导agent寻找最优解
                

        return next_state, reward, done, shaping

    def reset(self):
        state = [0]*self.N
        self.episode_return_raw = 0
        self.value_pre = len(self.G)/2

        return state

    def update(self,state): #什么作用？？
        # self.G = from_state_to_network(state)
        return state


# if __name__ == '__main__':
#
#     G = nx.from_numpy_array(G_matrix)
#     env = OP_Env_Precision(G_matrix, 10)
