# -*- coding: utf-8 -*-
"""Regression tests for SNAP's shared graph node-index contract."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch

import OP_Model
import utils
from data_generator import _save_real_graph
from generate_sd_realizations import generate_delay_distance, validate_topology
from graph_indexing import (
    adjacency_from_graph,
    all_pair_distances_from_graph,
    validate_adjacency_hop_alignment,
    validate_delay_distance,
)
from sourceLocalizationClass import SingleSourceLocalization


def insertion_order_graph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 4), (4, 1), (1, 3), (3, 2)])
    nx.set_edge_attributes(graph, 1.0, 'delay')
    return graph


class MatrixIndexContractTests(unittest.TestCase):
    def test_explicit_canonical_order_aligns_adjacency_and_distances(self):
        graph = insertion_order_graph()
        self.assertNotEqual(list(graph.nodes()), list(range(len(graph))))

        adjacency = adjacency_from_graph(graph)
        hop_distance, delay_distance = all_pair_distances_from_graph(graph)
        validate_adjacency_hop_alignment(adjacency, hop_distance)
        validate_delay_distance(
            delay_distance, hop_distance, sd=0.01, mu=1.0
        )

        self.assertEqual(adjacency[0, 4], 1.0)
        self.assertEqual(hop_distance[0, 2], 4.0)

        canonical_graph = nx.from_numpy_array(adjacency)
        np.testing.assert_allclose(
            utils.calc_features(graph),
            utils.calc_features(canonical_graph),
        )

    def test_default_networkx_insertion_order_is_rejected(self):
        graph = insertion_order_graph()
        bad_adjacency = nx.to_numpy_array(graph, dtype=float)
        hop_distance, _ = all_pair_distances_from_graph(graph)
        with self.assertRaisesRegex(ValueError, 'node-index mismatch'):
            validate_adjacency_hop_alignment(bad_adjacency, hop_distance)

    def test_real_graph_writer_persists_aligned_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            _save_real_graph(directory, 'test', insertion_order_graph(), sd=0.1)
            adjacency = np.loadtxt(Path(directory) / '0.npy')
            hop_distance = np.loadtxt(Path(directory) / '0_GDist.npy')
            delay_distance = np.loadtxt(Path(directory) / '0_GDistDelay.npy')
            validate_adjacency_hop_alignment(adjacency, hop_distance)
            validate_delay_distance(
                delay_distance, hop_distance, sd=0.1, mu=1.0
            )

    def test_delay_generation_uses_adjacency_indices(self):
        graph = insertion_order_graph()
        adjacency = adjacency_from_graph(graph)
        hop_distance, _ = all_pair_distances_from_graph(graph)
        delay_distance = generate_delay_distance(
            adjacency,
            mu=1.0,
            sd=0.05,
            rng=np.random.default_rng(42),
        )
        validate_delay_distance(
            delay_distance, hop_distance, sd=0.05, mu=1.0
        )

        permutation = [0, 2, 1, 4, 3]
        permuted = delay_distance[np.ix_(permutation, permutation)]
        with self.assertRaisesRegex(ValueError, 'index order'):
            validate_delay_distance(
                permuted, hop_distance, sd=0.05, mu=1.0
            )

    def test_delay_generator_topology_check_rejects_bad_saved_hop(self):
        graph = insertion_order_graph()
        adjacency = adjacency_from_graph(graph)
        hop_distance, _ = all_pair_distances_from_graph(graph)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adjacency_path = root / '0.npy'
            hop_path = root / '0_GDist.npy'
            np.savetxt(adjacency_path, adjacency)
            np.savetxt(hop_path, hop_distance[[1, 0, 2, 3, 4]][:, [1, 0, 2, 3, 4]])
            with self.assertRaisesRegex(ValueError, 'node-index mismatch'):
                validate_topology(
                    adjacency,
                    len(graph),
                    adjacency_path,
                    hop_path,
                )

    def test_data_loader_accepts_aligned_data_and_rejects_mismatch(self):
        graph = insertion_order_graph()
        adjacency = adjacency_from_graph(graph)
        hop_distance, delay_distance = all_pair_distances_from_graph(graph)
        with tempfile.TemporaryDirectory() as directory:
            node_dir = Path(directory) / 'TEST' / '5_nodes'
            node_dir.mkdir(parents=True)
            np.savetxt(node_dir / '0.npy', adjacency)
            np.savetxt(node_dir / '0_GDist.npy', hop_distance)
            np.savetxt(node_dir / '0_GDistDelay_0.npy', delay_distance)
            np.savetxt(node_dir / '0_sd_0.npy', [0.01, 0.0, 5.0])

            loaded = utils.data_loader(
                directory,
                'TEST',
                5,
                1,
                False,
                load_features=False,
            )
            self.assertEqual(loaded['G_distance'][0].shape, (5, 5))

            permutation = [1, 0, 2, 3, 4]
            np.savetxt(
                node_dir / '0_GDist.npy',
                hop_distance[np.ix_(permutation, permutation)],
            )
            with self.assertRaisesRegex(ValueError, 'node-index mismatch'):
                utils.data_loader(
                    directory,
                    'TEST',
                    5,
                    1,
                    True,
                    load_features=False,
                )


class PlacementIndexContractTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.barabasi_albert_graph(20, 2, seed=42)
        nx.set_edge_attributes(self.graph, 1.0, 'delay')
        self.adjacency = adjacency_from_graph(self.graph)
        self.hop_distance, _ = all_pair_distances_from_graph(self.graph)
        self.diagonal_adjacency = self.adjacency + np.eye(len(self.graph))
        self.localization = SingleSourceLocalization(
            self.graph, self.diagonal_adjacency
        )
        self.sources = list(range(len(self.graph)))

    def _assert_valid_placement(self, method, params, model_params=None):
        sensors = self.localization.place_observers(
            None,
            method,
            'PPOAgent',
            params,
            model_params or {},
            self.sources,
            self.hop_distance,
            G_distance=self.hop_distance,
        )
        self.assertEqual(len(sensors), 3)
        self.assertEqual(len(set(sensors)), 3)
        self.assertTrue(all(0 <= sensor < len(self.graph) for sensor in sensors))
        ranks = self.localization.rank_all_sources(
            self.sources,
            sensors,
            self.hop_distance,
            'snowball',
            self.hop_distance,
            'TRBS',
        )
        self.assertEqual(len(ranks), len(self.graph))

    def test_all_non_neural_baselines_return_matrix_indices(self):
        common = {'need_model': False, 'fixed_k': 3, 'L': 4}
        for method in ('random', 'BC', 'High_Coverage_Rate', 'CB', 'HV_OBS'):
            with self.subTest(method=method):
                self._assert_valid_placement(method, common)

    def test_one_shot_and_sgbs_keep_one_index_space(self):
        class FakeNetwork:
            sensor_budget_override = None

            def load_state_dict(self, state):
                return None

        class FakePPOAgent:
            def __init__(self, **kwargs):
                self.ac_net = FakeNetwork()

            @staticmethod
            def _available(state):
                return [index for index, chosen in enumerate(state) if not chosen]

            def predict_topk(self, state, features, adjacency, k=1):
                return self._available(state)[:k]

            def predict_topk_with_log_probs(
                self, state, features, adjacency, k=1
            ):
                return [
                    (index, -float(position))
                    for position, index in enumerate(self._available(state)[:k])
                ]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'fake.pt'
            torch.save({}, checkpoint)
            model_params = {'model_path': str(checkpoint)}
            with patch.object(OP_Model, 'PPOAgent', FakePPOAgent):
                self._assert_valid_placement(
                    'one_shot',
                    {
                        'need_model': True,
                        'fixed_k': 3,
                        'feature_set': 'f3',
                    },
                    model_params,
                )
                sensors = self.localization.place_observers(
                    None,
                    'SGBS',
                    'PPOAgent',
                    {
                        'need_model': True,
                        'fixed_k': 3,
                        'feature_set': 'f3',
                        'beta': 2,
                        'gamma': 2,
                    },
                    model_params,
                    self.sources,
                    self.hop_distance,
                    G_distance=self.hop_distance,
                )
                self.assertEqual(len(sensors), 3)
                self.assertEqual(len(set(sensors)), 3)


if __name__ == '__main__':
    unittest.main()
