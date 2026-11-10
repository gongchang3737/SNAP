import networkx as nx
import numpy as np
import scipy
import torch
import copy
import utils
import time
from sensor_budget_config import get_sensor_budget
from graph_indexing import (
    validate_graph_index_space,
    validate_index_sequence,
)
from sd_scenarios import SearchScoreContext

class SingleSourceLocalization:
    def __init__(self, G, G_matrix_diag):
        self.G = G
        self.G_matrix_diag = G_matrix_diag
        validate_graph_index_space(
            G,
            [np.asarray(G_matrix_diag).shape],
            context='source-localization graph',
        )

    def sample_source_lst(self, n_source, sample_method):
        if sample_method == 'random':
            source_actual_lst = np.random.choice(self.G.nodes, (n_source,))
        if sample_method == 'order':
            node_list = list(self.G.nodes)
            source_actual_lst = np.array(node_list[:n_source])

        return source_actual_lst

    def score_complete_placement(
        self,
        sensor_indices,
        *,
        sd_score_mode,
        source_actual_lst,
        G_distance_delay,
        spread_method,
        G_distance,
        locate_strategy,
        search_score_context=None,
    ):
        """Score one complete sensor set under legacy hop or independent MC."""
        n_nodes = len(self.G)
        sensors = validate_index_sequence(
            sensor_indices,
            n_nodes,
            name='SGBS completed sensor set',
            unique=True,
        )
        if sd_score_mode == 'hop':
            ranks = self.rank_all_sources(
                source_actual_lst,
                sensors,
                G_distance_delay,
                spread_method,
                G_distance,
                locate_strategy,
                chunk_size=min(64, n_nodes),
            )
            return float(np.average(ranks))
        if sd_score_mode not in ('mc', 'pmc'):
            raise ValueError(f'Unknown SGBS SD score mode {sd_score_mode!r}')
        if not isinstance(search_score_context, SearchScoreContext):
            raise ValueError(
                f'SGBS sd_score_mode={sd_score_mode!r} requires a '
                'SearchScoreContext'
            )

        scenario_ranks = []
        for scenario in search_score_context.scenario_pool.delay_distances:
            ranks = self.rank_all_sources(
                search_score_context.source_indices,
                sensors,
                scenario,
                search_score_context.spread_method,
                search_score_context.hop_distance,
                search_score_context.locate_strategy,
                chunk_size=min(64, n_nodes),
            )
            scenario_ranks.append(float(np.average(ranks)))
        return float(np.average(scenario_ranks))

    def place_observers(self, dens_obs, place_strategy, model_method, test_init_params=None, 
                        model_init_params=None, source_actual_lst=None, G_distance_delay=None,
                        spread_method='snowball', G_distance=None, locate_strategy='TRBS',
                        search_score_context=None):
        place_obs_lst = []

        # Use a CLI override when present; otherwise use the shared fixed map.
        fixed_k = test_init_params.get('fixed_k') if test_init_params else None
        num_obs = get_sensor_budget(len(self.G.nodes), override=fixed_k)
        feature_set = (
            test_init_params.get('feature_set', utils.DEFAULT_GRAPH_FEATURE_SET)
            if test_init_params else utils.DEFAULT_GRAPH_FEATURE_SET
        )

        # 选择性加载 PPO 模型
        model_load_start = time.perf_counter()
        if test_init_params['need_model']:
            if model_method == 'PPOAgent':
                from OP_Model import PPOAgent
                temp_model_init_params = copy.deepcopy(model_init_params) 
                model_path, init_params = temp_model_init_params.pop('model_path'), temp_model_init_params 
                
                # 动态分配设备，解决 CPU 环境下推理报错
                target_device = init_params.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                model = PPOAgent(**init_params)
                try:
                    model.ac_net.load_state_dict(torch.load(
                        model_path,
                        map_location=target_device,
                        weights_only=True,
                    ))
                except RuntimeError as exc:
                    raise RuntimeError(
                        f'Checkpoint {model_path!r} is incompatible with '
                        f'feature_set={feature_set!r}. Use the same --feature-set '
                        'for training and testing.'
                    ) from exc
                model.ac_net.sensor_budget_override = num_obs
            if model_method == 'GCN_DoubleDQN':
                from OP_Model import GCN_DoubleDQN
                # model_path参数被pop出来了，因此init_params中不再包含model_path，
                # 因为model_init_params变量在这个函数中只会使用一次，pop后model_init_params变量变化也没问题
                temp_model_init_params = copy.deepcopy(model_init_params) # pop后model_init_params会产生永久变化，需要备份
                model_path, init_params = temp_model_init_params.pop('model_path'), temp_model_init_params 
                model = GCN_DoubleDQN(**init_params)
                # 【修复核心】：从传入的参数中获取指定的 target device，若不存在则做安全平替
                target_device = init_params.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                # 在 torch.load 中显式指定 map_location
                model.load_state_dict(torch.load(model_path, map_location=target_device, weights_only=True))
                model.q_net.sensor_budget_override = num_obs
                model.target_q_net.sensor_budget_override = num_obs
        self.last_model_load_time = (
            time.perf_counter() - model_load_start
            if test_init_params['need_model'] else 0.0
        )
        selection_start = time.perf_counter()
        self.last_feature_time = 0.0
        self.last_search_scoring_time = 0.0
        self.last_search_scenario_generation_time = (
            search_score_context.scenario_pool.generation_seconds
            if search_score_context is not None else 0.0
        )


        if place_strategy == 'random':
            place_obs_lst = list(np.random.choice(self.G.nodes, num_obs, replace=False))

        if place_strategy == 'BC':
            score = nx.betweenness_centrality(self.G)
            score_sorted = sorted(score.items(), key=lambda item: item[1], reverse=True)
            for i in range(num_obs):
                place_obs = score_sorted[i][0]
                place_obs_lst.append(place_obs)

        if place_strategy == 'High_Coverage_Rate':
            G_matrix_temp = np.array(self.G_matrix_diag, copy=True)
            for i in range(num_obs):
                cover_lst = np.sum(np.maximum(G_matrix_temp, 0), axis=0)
                cover_lst[place_obs_lst] = 0 
                ind = np.argmax(cover_lst)
                place_obs_lst.append(ind)

                mask = np.array(self.G_matrix_diag[ind], dtype=bool)
                G_matrix_temp[mask] -= self.G_matrix_diag[mask]

                if np.all(G_matrix_temp <= 0):
                    G_matrix_temp += self.G_matrix_diag

        if place_strategy == 'K_Median':
            kmedian = utils.KMedian(
                n_clusters=num_obs,
                max_iter=100,
                tol=max(1, int(0.2 * num_obs)),
            )
            kmedian.fit(self.G)
            place_obs_lst = kmedian.centroids

        if place_strategy == 'HV_OBS':
            place_obs_lst = utils.select_hv_obs(
                G_distance,
                num_obs,
                max_path_nodes=test_init_params['L'],
            )

        if place_strategy == 'CB':
            place_obs_lst = utils.select_conditional_betweenness(self.G, num_obs)

        if place_strategy == 'one_shot':
            n_nodes = len(self.G.nodes)
            state = [0] * n_nodes
            feature_start = time.perf_counter()
            features = utils.calc_features(self.G, feature_set=feature_set)
            self.last_feature_time = time.perf_counter() - feature_start
            for i in range(num_obs):
                # PPO 预测: 内部带有 Masking
                action = model.predict_topk(state, features, self.G_matrix_diag, k=1)[0]
                action = validate_index_sequence(
                    [action], n_nodes, name='one-shot action'
                )[0]
                if state[action] != 0:
                    raise ValueError(
                        f'one-shot action {action} selects an existing sensor'
                    )
                state[action] = 1
            place_obs_lst = utils.getAllIndex(state, 1) 
        
        if place_strategy == 'SGBS':
            n_nodes = len(self.G.nodes)
            sd_score_mode = test_init_params.get('sd_score_mode', 'hop')
            if sd_score_mode not in ('hop', 'mc', 'pmc'):
                raise ValueError(
                    f'Unknown SGBS SD score mode {sd_score_mode!r}'
                )
            if sd_score_mode in ('mc', 'pmc'):
                if not isinstance(search_score_context, SearchScoreContext):
                    raise ValueError(
                        f'SGBS sd_score_mode={sd_score_mode!r} requires a '
                        'SearchScoreContext built from independent scenarios'
                    )
                if search_score_context.hop_distance.shape != (n_nodes, n_nodes):
                    raise ValueError('Search-score context has the wrong graph size')
                if not np.allclose(
                    search_score_context.hop_distance,
                    np.asarray(G_distance),
                    rtol=0.0,
                    atol=1e-9,
                ):
                    raise ValueError(
                        'Search-score context hop matrix does not match the '
                        'graph being searched'
                    )
            state = [0] * n_nodes
            # Each entry stores a partial placement and its cumulative policy
            # log-probability.  The search deliberately has no delay input.
            beam_front = [(state, 0.0)]
            feature_start = time.perf_counter()
            features = utils.calc_features(self.G, feature_set=feature_set)
            self.last_feature_time = time.perf_counter() - feature_start

            completion_score_cache = {}
            for step in range(num_obs):
                expanded_beams = []
                for beam_state, prefix_score in beam_front:
                    candidates = model.predict_topk_with_log_probs(
                        beam_state,
                        features,
                        self.G_matrix_diag,
                        k=test_init_params['gamma'],
                    )
                    for action, action_log_prob in candidates:
                        action = validate_index_sequence(
                            [action], n_nodes, name='SGBS expansion action'
                        )[0]
                        if beam_state[action] != 0:
                            raise ValueError(
                                f'SGBS expansion action {action} selects an '
                                'existing sensor'
                            )
                        new_state = copy.deepcopy(beam_state)
                        new_state[action] = 1
                        expanded_beams.append((
                            new_state,
                            prefix_score + action_log_prob,
                        ))

                # Complete each branch greedily, then score the resulting
                # sensor set by the actual topology-only localization objective.
                # In no-SD experiments this uses only known hop distances and
                # averages over hypothetical sources; it does not reveal the
                # source of any particular test instance.
                scored_beams = []
                for partial_state, prefix_score in expanded_beams:
                    rollout_state = copy.deepcopy(partial_state)
                    while sum(rollout_state) < num_obs:
                        action, _ = model.predict_topk_with_log_probs(
                            rollout_state,
                            features,
                            self.G_matrix_diag,
                            k=1,
                        )[0]
                        action = validate_index_sequence(
                            [action], n_nodes, name='SGBS rollout action'
                        )[0]
                        if rollout_state[action] != 0:
                            raise ValueError(
                                f'SGBS rollout action {action} selects an '
                                'existing sensor'
                            )
                        rollout_state[action] = 1

                    sensor_key = tuple(utils.getAllIndex(rollout_state, 1))
                    cache_scope = (
                        search_score_context.cache_scope
                        if sd_score_mode in ('mc', 'pmc')
                        else ('hop', 'legacy')
                    )
                    cache_key = (cache_scope, sensor_key)
                    if cache_key not in completion_score_cache:
                        score_start = time.perf_counter()
                        score = self.score_complete_placement(
                            list(sensor_key),
                            sd_score_mode=sd_score_mode,
                            source_actual_lst=source_actual_lst,
                            G_distance_delay=G_distance_delay,
                            spread_method=spread_method,
                            G_distance=G_distance,
                            locate_strategy=locate_strategy,
                            search_score_context=search_score_context,
                        )
                        self.last_search_scoring_time += (
                            time.perf_counter() - score_start
                        )
                        completion_score_cache[cache_key] = score
                    objective_rank = completion_score_cache[cache_key]
                    scored_beams.append((
                        objective_rank,
                        -prefix_score,
                        partial_state,
                        prefix_score,
                    ))

                scored_beams.sort(key=lambda item: (item[0], item[1]))
                if step == num_obs - 1:
                    keep_count = 1
                else:
                    keep_count = min(test_init_params['beta'], len(scored_beams))
                beam_front = [
                    (partial_state, prefix_score)
                    for _, _, partial_state, prefix_score in scored_beams[:keep_count]
                ]

            state = beam_front[0][0]
            place_obs_lst = utils.getAllIndex(state, 1) 

        self.last_placement_time = time.perf_counter() - selection_start
        self.last_search_time = max(
            self.last_placement_time - self.last_feature_time,
            0.0,
        )
        self.last_deployment_time = (
            self.last_placement_time
            + self.last_search_scenario_generation_time
        )
        place_obs_lst = validate_index_sequence(
            place_obs_lst,
            len(self.G),
            name=f'{place_strategy} sensor indices',
            unique=True,
        )
        if len(place_obs_lst) != num_obs:
            raise ValueError(
                f'{place_strategy} returned {len(place_obs_lst)} sensors; '
                f'expected {num_obs}'
            )
        return place_obs_lst

    def get_inform(self, source_actual, place_obs_lst, G_distance_delay, spread_method):
        inform = {}
        if spread_method == 'snowball':  
            for obs_i in place_obs_lst:
                inform[obs_i] = G_distance_delay[obs_i][source_actual] 
        inform = utils.minusMinTime(inform) 
        return inform

    def rank_all_sources(self, source_actual_lst, place_obs_lst,
                         G_distance_delay, spread_method, G_distance,
                         locate_strategy, chunk_size=1):
        """Evaluate source ranks with array-based TRBS and bounded memory."""
        n_nodes = len(self.G)
        source_actual_lst = validate_index_sequence(
            source_actual_lst,
            n_nodes,
            name='source indices',
        )
        place_obs_lst = validate_index_sequence(
            place_obs_lst,
            n_nodes,
            name='sensor indices',
            unique=True,
        )
        validate_graph_index_space(
            self.G,
            [
                np.asarray(G_distance).shape,
                np.asarray(G_distance_delay).shape,
            ],
            context='localization matrices',
        )
        if spread_method != 'snowball' or locate_strategy != 'TRBS':
            ranks = []
            for source_actual in source_actual_lst:
                inform = self.get_inform(
                    source_actual,
                    place_obs_lst,
                    G_distance_delay,
                    spread_method,
                )
                ranks.append(self.locate_source(
                    source_actual,
                    inform,
                    G_distance,
                    locate_strategy,
                ))
            return ranks

        observers = np.asarray(place_obs_lst, dtype=np.int64)
        sources = np.asarray(source_actual_lst, dtype=np.int64)
        distance_from_observers = np.asarray(G_distance)[observers, :]
        observed_delays = np.asarray(G_distance_delay)[np.ix_(observers, sources)]
        observed_delays = observed_delays - observed_delays.min(axis=0, keepdims=True)

        ranks = []
        for begin in range(0, len(sources), chunk_size):
            chunk_sources = sources[begin:begin + chunk_size]
            chunk_delays = observed_delays[:, begin:begin + chunk_size]
            mindist = distance_from_observers[:, :, None] - chunk_delays[:, None, :]
            variance = np.var(mindist, axis=0)

            for column, source_actual in enumerate(chunk_sources):
                target_val = variance[source_actual, column]
                smaller_count = np.count_nonzero(variance[:, column] < target_val)
                equal_count = np.count_nonzero(variance[:, column] == target_val)
                ranks.append(smaller_count + equal_count / 2.0)
        return ranks

    def locate_source(self, source_actual, inform, G_distance, locate_strategy):
        if locate_strategy == 'LPTV':
            node_estimator_dict = {}
            inform_sorted = sorted(list(zip(inform.keys(), inform.values())), key=lambda item: item[1], reverse=False)
            square_size = len(inform_sorted) - 1

            d = np.array([i[1] - inform_sorted[0][1] for i in inform_sorted[1:]]).reshape(square_size, 1)

            A = np.zeros((square_size, square_size))  
            for i in range(square_size):
                for j in range(square_size):
                    if i == j:
                        A[i, j] = 1.0 * nx.shortest_path_length(self.G, inform_sorted[0][0], inform_sorted[i + 1][0], weight='weight')
                    elif i < j:
                        path_i_node = nx.shortest_path(self.G, inform_sorted[0][0], inform_sorted[i + 1][0])
                        path_j_node = nx.shortest_path(self.G, inform_sorted[0][0], inform_sorted[j + 1][0])
                        path_i_pair = list(zip(path_i_node[:-1], path_i_node[1:]))  
                        path_j_pair = list(zip(path_j_node[:-1], path_j_node[1:]))
                        A[i, j] = 1.0 * len(set(path_i_pair).intersection(set(path_j_pair)))  
                    else:  
                        A[i, j] = A[j, i]

            A_inv = scipy.linalg.inv(A)

            for s_pred in self.G.nodes:
                dist_ref = nx.shortest_path_length(self.G, s_pred, inform_sorted[0][0])
                u = np.array([nx.shortest_path_length(self.G, s_pred, i[0], weight='weight')
                              - dist_ref for i in inform_sorted[1:]]).reshape(square_size, 1)
                estimator = (u.T) @ (A_inv) @ (d - u / 2)
                node_estimator_dict[s_pred] = estimator[0][0]

            smaller_count = 0
            equal_count = 0
            target_val = node_estimator_dict[source_actual]  
            for _, val in node_estimator_dict.items():
                if val < target_val:
                    smaller_count += 1
                elif val == target_val:
                    equal_count += 1

            ave_rank = smaller_count + equal_count / 2.0
            return ave_rank

        if locate_strategy == 'TRBS':
            mindist = np.concatenate([[G_distance[obs_i] - delay_i] for obs_i, delay_i in inform.items()], axis=0)  
            variance = np.var(mindist, axis=0)

            target_val = variance[source_actual]
            smaller_count = np.count_nonzero(variance < target_val)
            equal_count   = np.count_nonzero(variance == target_val)
            ave_rank = smaller_count + equal_count / 2.0
            return ave_rank
        
        if locate_strategy == 'Rank_List':
            mindist = np.concatenate([[G_distance[obs_i] - delay_i] for obs_i, delay_i in inform.items()], axis=0)  
            variance = np.var(mindist, axis=0) 
            node_estimator_dict = dict(zip(range(len(self.G.nodes)), variance))
            rank_list = list(node_estimator_dict.keys())
            rank_values = list(node_estimator_dict.values())
            return rank_list, rank_values
