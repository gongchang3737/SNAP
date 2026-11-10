# -*- coding: utf-8 -*-
"""Low-cost unit tests for independent SD scenario sampling."""

import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from graph_indexing import adjacency_from_graph, all_pair_distances_from_graph
from sd_scenarios import (
    CONDITIONAL_DELAY_BANDS,
    DEFAULT_DELAY_PRIOR,
    FINAL_TEST_STREAM,
    SEARCH_STREAM,
    TRAIN_STREAM,
    VALIDATION_STREAM,
    conditional_delay_prior,
    sample_delay_scenarios,
)


class DelayScenarioSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        graph = nx.cycle_graph(7)
        graph.add_edge(0, 3)
        cls.adjacency = adjacency_from_graph(graph)
        cls.hop_distance, _ = all_pair_distances_from_graph(graph)
        cls.kwargs = {
            'topology_id': 'unit/CYCLE/7/0',
            'count': 6,
            'seed': 42,
            'stream': SEARCH_STREAM,
        }

    def test_fixed_seed_is_reproducible_and_pool_is_read_only(self):
        first = sample_delay_scenarios(
            self.adjacency, self.hop_distance, **self.kwargs
        )
        second = sample_delay_scenarios(
            self.adjacency, self.hop_distance, **self.kwargs
        )
        self.assertEqual(first.pool_fingerprint, second.pool_fingerprint)
        self.assertEqual(first.raw_normal_scales, second.raw_normal_scales)
        self.assertEqual(first.band_names, second.band_names)
        for left, right in zip(first.delay_distances, second.delay_distances):
            np.testing.assert_array_equal(left, right)
            self.assertFalse(left.flags.writeable)
        for left, right in zip(
            first.edge_weight_matrices, second.edge_weight_matrices
        ):
            np.testing.assert_array_equal(left, right)
            self.assertFalse(left.flags.writeable)

    def test_stream_and_seed_create_independent_reproducible_flows(self):
        search = sample_delay_scenarios(
            self.adjacency, self.hop_distance, **self.kwargs
        )
        train = sample_delay_scenarios(
            self.adjacency,
            self.hop_distance,
            **{**self.kwargs, 'stream': TRAIN_STREAM},
        )
        other_seed = sample_delay_scenarios(
            self.adjacency,
            self.hop_distance,
            **{**self.kwargs, 'seed': 43},
        )
        self.assertNotEqual(search.pool_fingerprint, train.pool_fingerprint)
        self.assertNotEqual(search.pool_fingerprint, other_seed.pool_fingerprint)
        self.assertFalse(np.array_equal(
            search.delay_distances[0], train.delay_distances[0]
        ))

    def test_named_workflow_streams_and_topology_ids_are_separate(self):
        fingerprints = set()
        for stream in (
            TRAIN_STREAM, SEARCH_STREAM, VALIDATION_STREAM, FINAL_TEST_STREAM
        ):
            pool = sample_delay_scenarios(
                self.adjacency,
                self.hop_distance,
                **{**self.kwargs, 'stream': stream},
            )
            fingerprints.add(pool.pool_fingerprint)
        self.assertEqual(len(fingerprints), 4)

        renamed = sample_delay_scenarios(
            self.adjacency,
            self.hop_distance,
            **{**self.kwargs, 'topology_id': 'unit/CYCLE/7/renamed'},
        )
        self.assertNotIn(renamed.pool_fingerprint, fingerprints)

    def test_sampler_does_not_change_global_numpy_random_state(self):
        np.random.seed(2026)
        expected = np.random.random(8)
        np.random.seed(2026)
        sample_delay_scenarios(
            self.adjacency, self.hop_distance, **self.kwargs
        )
        observed = np.random.random(8)
        np.testing.assert_array_equal(observed, expected)

    def test_equal_strata_and_declared_right_closed_intervals(self):
        pool = sample_delay_scenarios(
            self.adjacency, self.hop_distance, **self.kwargs
        )
        self.assertEqual(pool.count, 6)
        self.assertEqual(pool.band_names, (
            'low', 'low', 'medium', 'medium', 'high', 'high'
        ))
        bands = {band.name: band for band in DEFAULT_DELAY_PRIOR.bands}
        for name, scale in zip(pool.band_names, pool.raw_normal_scales):
            self.assertGreater(scale, bands[name].lower)
            self.assertLessEqual(scale, bands[name].upper)

    def test_edge_weights_and_distances_are_valid(self):
        pool = sample_delay_scenarios(
            self.adjacency, self.hop_distance, **self.kwargs
        )
        topology_mask = self.adjacency > 0
        for edge_weights, distances in zip(
            pool.edge_weight_matrices, pool.delay_distances
        ):
            np.testing.assert_array_equal(edge_weights, edge_weights.T)
            self.assertTrue(np.all(edge_weights[topology_mask] > 0.0))
            self.assertTrue(np.all(edge_weights[~topology_mask] == 0.0))
            np.testing.assert_allclose(distances, distances.T)
            np.testing.assert_array_equal(np.diag(distances), np.zeros(7))
            self.assertTrue(np.isfinite(distances).all())

    def test_bad_saved_hop_matrix_is_rejected_before_sampling(self):
        bad_hop = self.hop_distance.copy()
        bad_hop[0, 1] = bad_hop[1, 0] = 99.0
        with patch('sd_scenarios.truncnorm.rvs') as mocked_sampler:
            with self.assertRaisesRegex(ValueError, 'node-index mismatch'):
                sample_delay_scenarios(
                    self.adjacency, bad_hop, **self.kwargs
                )
            mocked_sampler.assert_not_called()

    def test_final_test_artifacts_cannot_influence_sampler(self):
        fake_final_delay = np.full_like(self.hop_distance, 123.0)
        fake_final_metadata = {'sd': 0.499999, 'source': 5}
        before = sample_delay_scenarios(
            self.adjacency, self.hop_distance, **self.kwargs
        )
        fake_final_delay[:] = -999.0
        fake_final_metadata.update(sd=0.01, source=1)
        after = sample_delay_scenarios(
            self.adjacency, self.hop_distance, **self.kwargs
        )
        self.assertEqual(before.pool_fingerprint, after.pool_fingerprint)
        for left, right in zip(before.delay_distances, after.delay_distances):
            np.testing.assert_array_equal(left, right)

    def test_no_disturbance_is_exact_hop_without_truncated_normal(self):
        with patch(
            'sd_scenarios.truncnorm.rvs',
            side_effect=AssertionError('truncnorm must not be called'),
        ) as mocked_sampler:
            pool = sample_delay_scenarios(
                self.adjacency,
                self.hop_distance,
                **{**self.kwargs, 'count': 2},
                no_disturbance=True,
            )
        mocked_sampler.assert_not_called()
        self.assertTrue(pool.no_disturbance)
        self.assertEqual(pool.raw_normal_scales, (0.0, 0.0))
        for distances in pool.delay_distances:
            np.testing.assert_array_equal(distances, self.hop_distance)

    def test_invalid_scenario_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'multiple of 3'):
            sample_delay_scenarios(
                self.adjacency,
                self.hop_distance,
                **{**self.kwargs, 'count': 5},
            )

    def test_conditional_priors_are_mutually_exclusive_and_reproducible(self):
        expected = {
            'low': (0.0, 0.1),
            'medium': (0.1, 0.3),
            'high': (0.3, 0.5),
        }
        fingerprints = set()
        for level, (lower, upper) in expected.items():
            prior = conditional_delay_prior(level)
            self.assertEqual(prior.bands, (CONDITIONAL_DELAY_BANDS[level],))
            first = sample_delay_scenarios(
                self.adjacency,
                self.hop_distance,
                **{**self.kwargs, 'count': 5},
                prior=prior,
            )
            second = sample_delay_scenarios(
                self.adjacency,
                self.hop_distance,
                **{**self.kwargs, 'count': 5},
                prior=prior,
            )
            self.assertEqual(first.pool_fingerprint, second.pool_fingerprint)
            self.assertEqual(first.band_names, (level,) * 5)
            self.assertTrue(all(
                lower < scale <= upper
                for scale in first.raw_normal_scales
            ))
            fingerprints.add(first.pool_fingerprint)
        self.assertEqual(len(fingerprints), 3)


if __name__ == '__main__':
    unittest.main()
