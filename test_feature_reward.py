# -*- coding: utf-8 -*-
"""Regression tests for the compact graph encoder and PPO reward scale."""

import unittest

import networkx as nx
import numpy as np

import utils
import experiment_config as cfg
from OP_Env import OP_Env_Precision
from run_experiment import build_parser


class CompactGraphFeatureTests(unittest.TestCase):
    def test_ablation_feature_sets_are_cumulative_and_well_formed(self):
        expected_dims = {'f0': 4, 'f1': 5, 'f2': 8, 'f3': 10}
        graph = nx.watts_strogatz_graph(20, 4, 0.2, seed=42)
        for feature_set, expected_dim in expected_dims.items():
            names = utils.get_graph_feature_names(feature_set)
            features = utils.calc_features(graph, feature_set=feature_set)
            self.assertEqual(len(names), expected_dim)
            self.assertEqual(features.shape, (20, expected_dim))
            self.assertEqual(
                utils.get_augmented_node_feature_dim(feature_set),
                expected_dim + 2,
            )
            self.assertTrue(np.isfinite(features).all())
            self.assertTrue((features >= 0.0).all())
            self.assertTrue((features <= 1.0).all())
            np.testing.assert_allclose(features[:, names.index('bias')], 1.0)

        self.assertEqual(
            utils.get_graph_feature_names(),
            utils.get_graph_feature_names('f3'),
        )

    def test_feature_schema_and_ranges(self):
        expected_names = [
            'degree_norm',
            'betweenness_norm',
            'closeness_norm',
            'k_core_norm',
            'clustering_norm',
            'two_hop_ratio',
            'eccentricity_norm',
            'graph_density',
            'diameter_ratio',
            'bias',
        ]
        self.assertEqual(utils.GRAPH_FEATURE_NAMES, expected_names)
        self.assertEqual(utils.STATIC_GRAPH_FEATURE_DIM, len(expected_names))
        self.assertEqual(
            utils.AUGMENTED_NODE_FEATURE_DIM,
            len(expected_names) + utils.DYNAMIC_NODE_FEATURE_DIM,
        )

        for graph in (
            nx.cycle_graph(20),
            nx.complete_graph(20),
            nx.path_graph(20),
        ):
            features = utils.calc_features(graph)
            self.assertEqual(features.shape, (20, len(expected_names)))
            self.assertTrue(np.isfinite(features).all())
            self.assertTrue((features >= 0.0).all())
            self.assertTrue((features <= 1.0).all())
            np.testing.assert_allclose(features[:, -1], 1.0)

    def test_node_permutation_equivariance(self):
        graph = nx.barbell_graph(5, 2)
        adjacency = nx.to_numpy_array(graph, nodelist=range(len(graph)))
        permutation = np.random.RandomState(42).permutation(len(graph))
        permuted_graph = nx.from_numpy_array(
            adjacency[np.ix_(permutation, permutation)]
        )

        original = utils.calc_features(graph)
        permuted = utils.calc_features(permuted_graph)
        np.testing.assert_allclose(
            permuted,
            original[permutation],
            rtol=1e-7,
            atol=1e-9,
        )

    def test_precomputed_distance_reuse_is_exact(self):
        graph = nx.watts_strogatz_graph(20, 4, 0.2, seed=42)
        distance = np.asarray(nx.floyd_warshall_numpy(graph), dtype=float)
        computed = utils.calc_features(graph)
        reused = utils.calc_features(graph, distance_matrix=distance)
        np.testing.assert_allclose(reused, computed, rtol=0.0, atol=0.0)


class RewardNormalizationTests(unittest.TestCase):
    @staticmethod
    def _rollout(graph, actions):
        adjacency = nx.to_numpy_array(graph, nodelist=range(len(graph)))
        distance = np.asarray(nx.floyd_warshall_numpy(graph), dtype=float)
        env = OP_Env_Precision(
            dens_obs=None,
            sample_method='order',
            spread_method='snowball',
            locate_strategy='TRBS',
            begin_shaping=200000,
            shaping_param=1.0,
        )
        env.get_graph(
            graph,
            adjacency + np.eye(len(graph)),
            distance,
            distance.copy(),
        )
        state = env.reset()
        rewards = []
        for action in actions:
            state, reward, _, _ = env.step(state, action)
            rewards.append(reward)
        return float(np.sum(rewards)), float(env.value_pre)

    def test_reward_telescopes_and_is_order_invariant(self):
        graph = nx.star_graph(19)
        actions = [0, 1, 2]
        reverse_actions = list(reversed(actions))

        total, final_rank = self._rollout(graph, actions)
        reverse_total, reverse_final_rank = self._rollout(
            graph, reverse_actions
        )
        expected = 2.0 * (len(graph) / 2.0 - final_rank) / (len(graph) - 1)

        self.assertAlmostEqual(total, expected, places=12)
        self.assertAlmostEqual(reverse_final_rank, final_rank, places=12)
        self.assertAlmostEqual(reverse_total, total, places=12)

        diameter_scaled = 2.0 * (
            len(graph) / 2.0 - final_rank
        ) / nx.diameter(graph)
        self.assertNotAlmostEqual(total, diameter_scaled, places=6)


class FeatureSetConfigurationTests(unittest.TestCase):
    def test_model_dimensions_follow_feature_set(self):
        expected_augmented_dims = {'f0': 6, 'f1': 7, 'f2': 10, 'f3': 12}
        for feature_set, expected_dim in expected_augmented_dims.items():
            params = cfg.build_model_params('PPOAgent', feature_set=feature_set)
            self.assertEqual(
                params['PPOAgent']['n_features_aug'],
                expected_dim,
            )

    def test_existing_commands_default_to_f3(self):
        parser = build_parser()
        train_args = parser.parse_args(['train', '--model-id', 'compat'])
        test_args = parser.parse_args(['test', '--model-id', 'compat'])
        self.assertEqual(train_args.feature_set, 'f3')
        self.assertEqual(test_args.feature_set, 'f3')
        self.assertEqual(test_args.neural_methods, 'one_shot,SGBS')

        methods = cfg.build_method_params(
            'compat', 'PPOAgent', include_baselines=False
        )['test_methods']
        self.assertEqual(methods, ['one_shot', 'SGBS'])


if __name__ == '__main__':
    unittest.main()
