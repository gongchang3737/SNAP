# -*- coding: utf-8 -*-
"""Focused tests for stage-3 independent-scenario SD-SGBS scoring."""

import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch

import OP_Env
import OP_Model
import experiment_config as cfg
from graph_indexing import adjacency_from_graph, all_pair_distances_from_graph
from run_experiment import build_explicit_results_path, build_parser
from sd_scenarios import build_search_score_context
from sourceLocalizationClass import SingleSourceLocalization


class SearchScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = nx.cycle_graph(6)
        cls.graph.add_edge(0, 3)
        cls.adjacency = adjacency_from_graph(cls.graph)
        cls.hop, _ = all_pair_distances_from_graph(cls.graph)
        cls.diagonal = cls.adjacency + np.eye(len(cls.graph))

    def build_context(self, seed=42, count=3):
        return build_search_score_context(
            self.adjacency,
            self.hop,
            topology_id='unit/CYCLE/6/graph_0',
            count=count,
            seed=seed,
        )

    def test_mc_score_matches_manual_scenario_and_all_source_average(self):
        localization = SingleSourceLocalization(self.graph, self.diagonal)
        context = self.build_context(count=6)
        sensors = [0, 3]
        actual = localization.score_complete_placement(
            sensors,
            sd_score_mode='mc',
            # Deliberately hostile final-test-like inputs: MC must ignore them.
            source_actual_lst=[5],
            G_distance_delay=np.full_like(self.hop, 999.0),
            spread_method='unused',
            G_distance=np.full_like(self.hop, -1.0),
            locate_strategy='unused',
            search_score_context=context,
        )
        manual = np.mean([
            np.mean(localization.rank_all_sources(
                list(range(len(self.graph))),
                sensors,
                scenario,
                'snowball',
                self.hop,
                'TRBS',
                chunk_size=6,
            ))
            for scenario in context.scenario_pool.delay_distances
        ])
        self.assertAlmostEqual(actual, float(manual), places=12)
        self.assertEqual(context.source_indices, tuple(range(len(self.graph))))
        pmc_actual = localization.score_complete_placement(
            sensors,
            sd_score_mode='pmc',
            source_actual_lst=[5],
            G_distance_delay=np.full_like(self.hop, 999.0),
            spread_method='unused',
            G_distance=np.full_like(self.hop, -1.0),
            locate_strategy='unused',
            search_score_context=context,
        )
        self.assertEqual(pmc_actual, actual)

    def test_hop_score_is_the_legacy_single_matrix_score(self):
        localization = SingleSourceLocalization(self.graph, self.diagonal)
        sources = list(range(len(self.graph)))
        sensors = [0, 3]
        expected = float(np.mean(localization.rank_all_sources(
            sources, sensors, self.hop, 'snowball', self.hop, 'TRBS'
        )))
        actual = localization.score_complete_placement(
            sensors,
            sd_score_mode='hop',
            source_actual_lst=sources,
            G_distance_delay=self.hop,
            spread_method='snowball',
            G_distance=self.hop,
            locate_strategy='TRBS',
        )
        self.assertEqual(actual, expected)

    def test_sgbs_candidates_share_pool_and_order_invariant_cache(self):
        class FakeNetwork:
            sensor_budget_override = None

            def load_state_dict(self, state):
                return None

        class FakePPOAgent:
            def __init__(self, **kwargs):
                self.ac_net = FakeNetwork()

            @staticmethod
            def predict_topk_with_log_probs(state, features, adjacency, k=1):
                available = [i for i, chosen in enumerate(state) if not chosen]
                return [
                    (node, -float(position))
                    for position, node in enumerate(available[:k])
                ]

        localization = SingleSourceLocalization(self.graph, self.diagonal)
        context = self.build_context(count=3)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'fake.pt'
            torch.save({}, checkpoint)
            with patch.object(OP_Model, 'PPOAgent', FakePPOAgent), patch.object(
                localization,
                'rank_all_sources',
                wraps=localization.rank_all_sources,
            ) as rank_mock:
                sensors = localization.place_observers(
                    None,
                    'SGBS',
                    'PPOAgent',
                    {
                        'need_model': True,
                        'fixed_k': 2,
                        'feature_set': 'f3',
                        'beta': 2,
                        'gamma': 2,
                        'sd_score_mode': 'mc',
                    },
                    {'model_path': str(checkpoint)},
                    # MC must not use this reduced source set or fake delay.
                    [5],
                    np.full_like(self.hop, 999.0),
                    'snowball',
                    self.hop,
                    'TRBS',
                    context,
                )

        self.assertEqual(len(sensors), 2)
        # Three unique completed sets are evaluated. Each is evaluated once
        # over the shared three-scenario pool, despite duplicate completion
        # paths such as 0->1 and 1->0.
        self.assertEqual(rank_mock.call_count, 9)
        sensor_sets = [tuple(call.args[1]) for call in rank_mock.call_args_list]
        self.assertEqual(set(sensor_sets), {(0, 1), (0, 2), (1, 2)})
        for sensor_set in set(sensor_sets):
            self.assertEqual(sensor_sets.count(sensor_set), 3)
        for call in rank_mock.call_args_list:
            self.assertEqual(call.args[0], context.source_indices)
        self.assertGreater(localization.last_search_scoring_time, 0.0)
        self.assertEqual(
            localization.last_search_scenario_generation_time,
            context.scenario_pool.generation_seconds,
        )


class StageThreeIntegrationTests(unittest.TestCase):
    def test_sd_sgbs_defaults_route_to_method_specific_directories(self):
        expected_directories = {
            'hop': os.path.join('result', 'hop_sgbs'),
            'mc': os.path.join('result', 'mc_sgbs'),
            'pmc': os.path.join('result', 'pmc_sgbs'),
        }
        for mode, expected_directory in expected_directories.items():
            cli = [
                'test', '--model-id', 'ppo_mix_v3_sd',
                '--neural-methods', 'SGBS',
                '--sd-score-mode', mode,
            ]
            if mode == 'pmc':
                cli.extend(['--known-sd-level', 'all'])
            args = build_parser().parse_args(cli)
            path = build_explicit_results_path(args, 'BA', 20, 3)
            self.assertEqual(
                os.path.dirname(os.path.normpath(path)),
                os.path.normpath(expected_directory),
            )

        one_shot_args = build_parser().parse_args([
            'test', '--model-id', 'ppo_mix_v3_sd',
            '--neural-methods', 'one_shot',
        ])
        self.assertIsNone(
            build_explicit_results_path(one_shot_args, 'BA', 20, 3)
        )

        no_sd_args = build_parser().parse_args([
            'test', '--model-id', 'ppo_mix_v3_no_sd',
            '--neural-methods', 'SGBS', '--no-sd', 'True',
        ])
        self.assertIsNone(
            build_explicit_results_path(no_sd_args, 'BA', 20, 3)
        )

        override_args = build_parser().parse_args([
            'test', '--model-id', 'ppo_mix_v3_sd',
            '--neural-methods', 'SGBS', '--sd-score-mode', 'mc',
            '--results-dir', 'result/custom',
        ])
        override_path = build_explicit_results_path(
            override_args, 'BA', 20, 3
        )
        self.assertEqual(
            os.path.dirname(os.path.normpath(override_path)),
            os.path.normpath(os.path.join('result', 'custom')),
        )

    def test_cli_and_config_default_to_legacy_hop(self):
        args = build_parser().parse_args([
            'test', '--model-id', 'ppo_mix_v3_sd'
        ])
        self.assertEqual(args.sd_score_mode, 'hop')
        self.assertEqual(args.search_scenarios, 12)
        self.assertIsNone(args.search_seed)

        params = cfg.build_method_params(
            'ppo_mix_v3_sd', 'PPOAgent', include_baselines=False
        )
        self.assertEqual(params['test_init_params']['SGBS']['sd_score_mode'], 'hop')
        legacy = cfg.build_tester_params(
            'ppo_mix_v3_sd', 'BA', 20, no_sd=False
        )['results_save_path']
        mc = cfg.build_tester_params(
            'ppo_mix_v3_sd',
            'BA',
            20,
            no_sd=False,
            sd_score_mode='mc',
            search_scenarios=12,
            search_seed=77,
        )['results_save_path']
        self.assertTrue(legacy.endswith('_delay.json'))
        self.assertTrue(mc.endswith('_delay_mc_s12_seed77.json'))
        self.assertNotEqual(legacy, mc)

        isolated_args = build_parser().parse_args([
            'test',
            '--model-id', 'ppo_mix_v3_sd',
            '--results-dir', 'result/sd_sgbs_stage4',
            '--result-tag', 'mc_M12_seed42001',
        ])
        isolated = build_explicit_results_path(
            isolated_args, 'BA', 20, 3
        )
        self.assertEqual(
            os.path.normpath(isolated),
            os.path.join(
                'result',
                'sd_sgbs_stage4',
                'ppo_mix_v3_sd_BA_20_k3_mc_M12_seed42001.json',
            ),
        )

        one_shot_args = build_parser().parse_args([
            'test', '--model-id', 'ppo_mix_v3_sd',
            '--results-dir', 'result/isolated',
            '--neural-methods', 'one_shot',
        ])
        hop_args = build_parser().parse_args([
            'test', '--model-id', 'ppo_mix_v3_sd',
            '--results-dir', 'result/isolated',
            '--neural-methods', 'SGBS', '--sd-score-mode', 'hop',
        ])
        mc_args = build_parser().parse_args([
            'test', '--model-id', 'ppo_mix_v3_sd',
            '--results-dir', 'result/isolated',
            '--neural-methods', 'SGBS', '--sd-score-mode', 'mc',
            '--search-scenarios', '12', '--search-seed', '42001',
        ])
        pmc_args = build_parser().parse_args([
            'test', '--model-id', 'ppo_mix_v3_sd',
            '--results-dir', 'result/isolated',
            '--neural-methods', 'SGBS', '--sd-score-mode', 'pmc',
            '--known-sd-level', 'all',
            '--search-scenarios', '12', '--search-seed', '42001',
        ])
        automatic_paths = {
            os.path.normpath(build_explicit_results_path(args, 'BA', 20, 3))
            for args in (one_shot_args, hop_args, mc_args, pmc_args)
        }
        self.assertEqual(len(automatic_paths), 4)
        self.assertTrue(any('_pmc_all_M12_' in path for path in automatic_paths))

        pmc = cfg.build_tester_params(
            'ppo_mix_v3_sd', 'BA', 20, no_sd=False,
            sd_score_mode='pmc', search_scenarios=12,
            search_seed=77, known_sd_level='all',
        )['results_save_path']
        self.assertTrue(pmc.endswith('_delay_pmc_all_s12_seed77.json'))

    def test_pmc_all_builds_three_deployments_before_final_delay_loading(self):
        graph = nx.cycle_graph(20)
        adjacency = adjacency_from_graph(graph)
        hop, _ = all_pair_distances_from_graph(graph)
        test_data = {
            'G': [graph],
            'G_matrix': [adjacency],
            'G_matrix_diag': [adjacency + np.eye(20)],
            'G_distance': [hop],
        }
        events = []
        contexts = []
        sensors_by_level = {
            'low': [0, 1, 2],
            'medium': [3, 4, 5],
            'high': [6, 7, 8],
        }

        def fake_place(instance, *args, **kwargs):
            context = args[10]
            level = context.scenario_pool.band_names[0]
            events.append(f'placement-{level}')
            contexts.append(context)
            instance.last_model_load_time = 0.0
            instance.last_placement_time = 0.0
            instance.last_feature_time = 0.0
            instance.last_search_time = 0.0
            instance.last_search_scoring_time = 0.0
            instance.last_search_scenario_generation_time = (
                context.scenario_pool.generation_seconds
            )
            return sensors_by_level[level]

        def materialize(_test_data, _graph_index):
            events.append('final-delay-materialization')
            return [
                (0, 0.05, hop.copy()),
                (1, 0.20, hop.copy()),
                (2, 0.40, hop.copy()),
            ]

        env = OP_Env.OP_Env_Precision(
            dens_obs=None,
            sample_method='order',
            spread_method='snowball',
            locate_strategy='TRBS',
            begin_shaping=200000,
            shaping_param=1.0,
        )
        with patch.object(
            SingleSourceLocalization, 'place_observers', fake_place,
        ), patch.object(
            OP_Env.OP_Env_Precision,
            '_delay_realizations_for_graph',
            side_effect=materialize,
        ):
            log = env.for_test(
                test_data,
                test_methods=['SGBS'],
                model_methods=['PPOAgent'],
                test_init_params={
                    'SGBS': {
                        'need_model': True,
                        'fixed_k': 3,
                        'feature_set': 'f3',
                        'beta': 2,
                        'gamma': 2,
                        'sd_score_mode': 'pmc',
                        'known_sd_level': 'all',
                        'search_scenarios': 2,
                        'search_seed': 42001,
                        'sd_prior': 'equal_lmh',
                        'delay_mu': 1.0,
                        'topology_prefix': 'unit/CYCLE/20',
                    }
                },
                model_init_params={'PPOAgent': {}},
            )

        self.assertEqual(events[:3], [
            'placement-low', 'placement-medium', 'placement-high'
        ])
        self.assertEqual(events[3], 'final-delay-materialization')
        self.assertEqual(len({
            context.scenario_pool.pool_fingerprint for context in contexts
        }), 3)
        method_log = log['PPOAgent']['SGBS']
        self.assertEqual(method_log['placements_by_sd_level'][0], sensors_by_level)
        self.assertEqual(len(method_log['realization_records']), 3)
        self.assertEqual(
            {item['sd_group'] for item in method_log['realization_records']},
            {'low', 'medium', 'high'},
        )
        self.assertIsNotNone(method_log['ave_conditional_total_time'])

    def test_final_delay_is_materialized_only_after_placement(self):
        graph = nx.cycle_graph(20)
        adjacency = adjacency_from_graph(graph)
        hop, _ = all_pair_distances_from_graph(graph)
        test_data = {
            'G': [graph],
            'G_matrix': [adjacency],
            'G_matrix_diag': [adjacency + np.eye(20)],
            'G_distance': [hop],
        }
        events = []
        original_place = SingleSourceLocalization.place_observers

        def recording_place(instance, *args, **kwargs):
            events.append('placement')
            return original_place(instance, *args, **kwargs)

        def materialize(_test_data, _graph_index):
            events.append('final-delay-materialization')
            return [(0, 0.05, hop.copy())]

        env = OP_Env.OP_Env_Precision(
            dens_obs=None,
            sample_method='order',
            spread_method='snowball',
            locate_strategy='TRBS',
            begin_shaping=200000,
            shaping_param=1.0,
        )
        with patch.object(
            SingleSourceLocalization,
            'place_observers',
            recording_place,
        ), patch.object(
            OP_Env.OP_Env_Precision,
            '_delay_realizations_for_graph',
            side_effect=materialize,
        ):
            log = env.for_test(
                test_data,
                test_methods=['BC'],
                model_methods=['PPOAgent'],
                test_init_params={
                    'BC': {'need_model': False, 'fixed_k': 2}
                },
                model_init_params={'PPOAgent': {}},
            )
        self.assertLess(
            events.index('placement'),
            events.index('final-delay-materialization'),
        )
        self.assertEqual(
            log['PPOAgent']['BC']['placement_information'],
            'topology_and_hop_only',
        )


if __name__ == '__main__':
    unittest.main()
