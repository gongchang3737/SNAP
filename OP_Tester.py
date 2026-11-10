# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import json
from OP_Model import * #加载GCN-DQN模型类
# from baseModel import *
import networkx as nx
import utils
import torch
import OP_Env
import os
import hashlib
import numpy as np

class OP_Tester:
    def __init__(self, tester_params, method_params, env_params):
        self.tester_params = tester_params
        self.method_params = method_params
        self.env_params = env_params #训练数据路径

    @staticmethod
    def _file_identifier(path):
        identifier = {
            'path': os.path.normpath(path) if path else None,
            'exists': bool(path and os.path.isfile(path)),
            'size_bytes': None,
            'sha256': None,
        }
        if not identifier['exists']:
            return identifier
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        identifier['size_bytes'] = os.path.getsize(path)
        identifier['sha256'] = digest.hexdigest()
        return identifier

    @staticmethod
    def _data_identifier(test_data, data_params):
        """Identify topology/index inputs without loading final delay files."""
        digest = hashlib.sha256()
        digest.update(str(data_params.get('data_path')).encode('utf-8'))
        digest.update(str(data_params.get('graph_type')).encode('utf-8'))
        digest.update(str(data_params.get('n_nodes')).encode('ascii'))
        for adjacency, hop_distance in zip(
            test_data['G_matrix'], test_data['G_distance']
        ):
            digest.update(
                np.asarray(adjacency, dtype='<f8').tobytes(order='C')
            )
            digest.update(
                np.asarray(hop_distance, dtype='<f8').tobytes(order='C')
            )
        for indices, sd_values in zip(
            test_data.get('delay_realization_indices', []),
            test_data.get('delay_sd_values', []),
        ):
            digest.update(np.asarray(indices, dtype='<i8').tobytes(order='C'))
            digest.update(np.asarray(sd_values, dtype='<f8').tobytes(order='C'))
        return {
            'data_path': os.path.normpath(data_params.get('data_path', '')),
            'graph_type': data_params.get('graph_type'),
            'n_nodes': data_params.get('n_nodes'),
            'requested_topologies': data_params.get('num_graphs'),
            'loaded_topologies': len(test_data['G']),
            'no_sd': data_params.get('no_sd'),
            'topology_and_metadata_sha256': digest.hexdigest(),
            'hop_alignment_checked_by_loader': True,
        }

    def _add_run_records(self, test_log, test_data):
        data_record = self._data_identifier(
            test_data, self.tester_params['test_data']
        )
        timing_semantics = {
            'search_scenario_generation_time': (
                'Independent search-pool generation only; already included '
                'in deployment_time.'
            ),
            'search_scoring_time': (
                'Completed-placement rank scoring only; a subset of '
                'search_time and deployment_time.'
            ),
            'search_time': (
                'Search logic excluding feature calculation and scenario '
                'generation; already included in deployment_time.'
            ),
            'deployment_time': (
                'Wall time from search-context construction through sensor '
                'selection; includes scenario generation, model loading, '
                'features, search, and search scoring.'
            ),
            'pmc_conditional_total_deployment_time': (
                'Benchmark-only sum of all requested low/medium/high pMC '
                'deployments. A real deployment with one known level uses '
                'only that level time.'
            ),
            'evaluation_time': (
                'Final localization evaluation after deployment; excluded '
                'from deployment_time.'
            ),
            'time': 'deployment_time + evaluation_time.',
        }
        for model_method, method_logs in test_log.items():
            model_params = self.method_params['model_init_params'].get(
                model_method, {}
            )
            checkpoint = self._file_identifier(model_params.get('model_path'))
            for method_name, method_log in method_logs.items():
                method_params = self.method_params['test_init_params'][method_name]
                score_mode = method_log.get('sd_score_mode')
                method_log['run_config'] = {
                    'model_method': model_method,
                    'inference_method': method_name,
                    'checkpoint': checkpoint,
                    'data': data_record,
                    'feature_set': method_params.get('feature_set'),
                    'run_seed': method_params.get('run_seed'),
                    'sensor_budget': method_params.get('fixed_k'),
                    'beta': method_params.get('beta'),
                    'gamma': method_params.get('gamma'),
                    'sd_score_mode': score_mode,
                    'delay_prior': (
                        method_params.get('sd_prior')
                        if score_mode == 'mc' else None
                    ),
                    'conditional_delay_priors': (
                        {
                            'low': '(0, 0.1]',
                            'medium': '(0.1, 0.3]',
                            'high': '(0.3, 0.5]',
                        }
                        if score_mode == 'pmc' else None
                    ),
                    'known_sd_level': (
                        method_params.get('known_sd_level')
                        if score_mode == 'pmc' else None
                    ),
                    'delay_mu': (
                        method_params.get('delay_mu')
                        if score_mode in ('mc', 'pmc') else None
                    ),
                    'search_scenarios': (
                        method_params.get('search_scenarios')
                        if score_mode in ('mc', 'pmc') else 0
                    ),
                    'search_seed': (
                        method_params.get('search_seed')
                        if score_mode in ('mc', 'pmc') else None
                    ),
                    'uses_final_test_delay_for_deployment': False,
                    'uses_actual_test_sd_for_deployment': False,
                    'uses_known_sd_level_for_deployment': (
                        score_mode == 'pmc'
                    ),
                    'average_rank': method_log.get('ave_rank'),
                    'average_deployment_seconds': method_log.get(
                        'ave_deployment_time'
                    ),
                    'result_path': os.path.normpath(
                        self.tester_params['results_save_path']
                    ),
                }
                method_log['timing_semantics'] = timing_semantics

    def run(self):
        # 加载数据
        # 将测试数据读取后存下来，可能会有空间占用过多的问题？
        test_data = utils.data_loader(**self.tester_params['test_data'])

        # # 加载测试环境
        env = OP_Env.OP_Env_Precision(**self.env_params['OP_Env_Precision'])

        # 测试模型
        test_log = env.for_test(test_data, **self.method_params)
        self._add_run_records(test_log, test_data)

        # 保存测试结果, 绘制测试图像并保存图像
        save_path = self.tester_params['results_save_path']

        # 【新增】自动创建目录，如果目录已存在则忽略 (exist_ok=True)
        # os.path.dirname 获取文件路径中的目录部分
        save_dir = os.path.dirname(save_path)
        if save_dir:  # 防止 save_path 只是文件名而没有目录的情况
            os.makedirs(save_dir, exist_ok=True)
        
        # 现在可以安全并美观地写入文件了
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(test_log, f, indent=4, ensure_ascii=False)
        
    
