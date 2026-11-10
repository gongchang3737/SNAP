# -*- coding: utf-8 -*-
"""Shared node-index and graph-matrix consistency helpers.

SNAP treats every matrix row/column and every PPO/baseline action as the same
zero-based node index.  This module centralizes that contract so graph
generation, delay generation, loading, search, and evaluation cannot silently
use different NetworkX node orders.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def canonical_nodelist(G: nx.Graph) -> list[int]:
    """Return the only node order accepted by matrix-indexed SNAP code."""
    n_nodes = G.number_of_nodes()
    expected = list(range(n_nodes))
    if set(G.nodes()) != set(expected):
        raise ValueError(
            'SNAP graphs must use contiguous integer labels 0..N-1 before '
            f'matrix conversion; got nodes={list(G.nodes())[:10]!r}'
        )
    return expected


def relabel_to_canonical_indices(G: nx.Graph) -> nx.Graph:
    """Return a copy whose labels are deterministic contiguous matrix indices.

    Existing 0..N-1 labels are preserved even when their insertion order is
    non-canonical.  Non-contiguous or non-integer labels are sorted by their
    string representation before relabeling.
    """
    n_nodes = G.number_of_nodes()
    if set(G.nodes()) == set(range(n_nodes)):
        return G.copy()
    ordered_nodes = sorted(G.nodes(), key=lambda node: str(node))
    mapping = {node: index for index, node in enumerate(ordered_nodes)}
    return nx.relabel_nodes(G, mapping, copy=True)


def adjacency_from_graph(G: nx.Graph) -> np.ndarray:
    """Build an adjacency matrix in canonical matrix-index order."""
    nodes = canonical_nodelist(G)
    return nx.to_numpy_array(G, nodelist=nodes, dtype=float)


def all_pair_distances_from_graph(
    G: nx.Graph,
    *,
    delay_weight: str = 'delay',
) -> tuple[np.ndarray, np.ndarray]:
    """Build hop and weighted APSP matrices in canonical matrix-index order."""
    nodes = canonical_nodelist(G)
    n_nodes = len(nodes)
    hop_lengths = dict(nx.all_pairs_shortest_path_length(G))
    delay_lengths = dict(
        nx.all_pairs_dijkstra_path_length(G, weight=delay_weight)
    )
    hop_distance = np.empty((n_nodes, n_nodes), dtype=float)
    delay_distance = np.empty((n_nodes, n_nodes), dtype=float)
    for row, source in enumerate(nodes):
        if len(hop_lengths[source]) != n_nodes:
            raise ValueError('Cannot build APSP matrices for a disconnected graph')
        for column, target in enumerate(nodes):
            hop_distance[row, column] = hop_lengths[source][target]
            delay_distance[row, column] = delay_lengths[source][target]
    return hop_distance, delay_distance


def _as_square_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'{name} must be square; got shape={matrix.shape}')
    if not np.isfinite(matrix).all():
        raise ValueError(f'{name} contains NaN or Inf')
    return matrix


def hop_distance_from_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Recompute unweighted APSP using adjacency matrix indices as nodes."""
    matrix = _as_square_matrix(adjacency, 'adjacency')
    topology = csr_matrix((matrix > 0).astype(np.int8))
    return np.asarray(
        dijkstra(
            topology,
            directed=False,
            unweighted=True,
            return_predecessors=False,
        ),
        dtype=float,
    )


def validate_adjacency_hop_alignment(
    adjacency: np.ndarray,
    hop_distance: np.ndarray,
    *,
    context: str = 'graph',
    atol: float = 1e-9,
) -> np.ndarray:
    """Assert that ``A[i,j]`` and ``D_hop[i,j]`` use the same node indices."""
    adjacency = _as_square_matrix(adjacency, f'{context} adjacency')
    hop_distance = _as_square_matrix(hop_distance, f'{context} hop distance')
    if adjacency.shape != hop_distance.shape:
        raise ValueError(
            f'{context}: adjacency shape {adjacency.shape} does not match '
            f'hop-distance shape {hop_distance.shape}'
        )
    if not np.allclose(adjacency, adjacency.T, rtol=0.0, atol=atol):
        raise ValueError(f'{context}: adjacency matrix is not symmetric')
    if np.any(adjacency < 0):
        raise ValueError(f'{context}: adjacency matrix contains negative values')
    if not np.allclose(np.diag(adjacency), 0.0, rtol=0.0, atol=atol):
        raise ValueError(f'{context}: adjacency diagonal must be zero')

    recomputed = hop_distance_from_adjacency(adjacency)
    if not np.isfinite(recomputed).all():
        raise ValueError(f'{context}: adjacency matrix represents a disconnected graph')
    if not np.allclose(hop_distance, recomputed, rtol=0.0, atol=atol):
        difference = np.abs(hop_distance - recomputed)
        mismatch_count = int(np.count_nonzero(difference > atol))
        raise ValueError(
            f'{context}: node-index mismatch between adjacency and hop '
            f'distance ({mismatch_count}/{difference.size} entries differ; '
            f'max_abs_diff={float(np.max(difference)):.10g})'
        )
    return recomputed


def validate_delay_distance(
    delay_distance: np.ndarray,
    hop_distance: np.ndarray,
    *,
    sd: float | None = None,
    mu: float = 1.0,
    context: str = 'graph',
    atol: float = 1e-8,
) -> None:
    """Validate delay APSP shape/index order and optional generation bounds.

    If edge delays are sampled from ``[mu-2*sd, mu+2*sd]``, every weighted
    shortest-path entry must lie within the corresponding hop-distance bounds.
    This catches a row/column permutation between D_delay and D_hop without
    needing the individual sampled edge weights.
    """
    delay_distance = _as_square_matrix(
        delay_distance, f'{context} delay distance'
    )
    hop_distance = _as_square_matrix(hop_distance, f'{context} hop distance')
    if delay_distance.shape != hop_distance.shape:
        raise ValueError(
            f'{context}: delay-distance shape {delay_distance.shape} does not '
            f'match hop-distance shape {hop_distance.shape}'
        )
    if np.any(delay_distance < -atol):
        raise ValueError(f'{context}: delay distance contains negative values')
    if not np.allclose(delay_distance, delay_distance.T, rtol=1e-8, atol=atol):
        raise ValueError(f'{context}: delay-distance matrix is not symmetric')
    if not np.allclose(
        np.diag(delay_distance), 0.0, rtol=0.0, atol=atol
    ):
        raise ValueError(f'{context}: delay-distance diagonal must be zero')

    if sd is None:
        return
    sd = float(sd)
    if sd <= 0:
        raise ValueError(f'{context}: delay SD must be positive; got {sd}')
    lower_edge = max(mu - 2.0 * sd, 0.0)
    upper_edge = mu + 2.0 * sd
    lower_bound = lower_edge * hop_distance
    upper_bound = upper_edge * hop_distance
    below = delay_distance < lower_bound - atol
    above = delay_distance > upper_bound + atol
    if np.any(below) or np.any(above):
        violation_count = int(np.count_nonzero(below | above))
        raise ValueError(
            f'{context}: delay and hop matrices do not share a valid node '
            f'index order ({violation_count}/{delay_distance.size} entries '
            'violate truncated-delay path bounds)'
        )


def validate_index_sequence(
    indices: Iterable[int],
    n_nodes: int,
    *,
    name: str,
    unique: bool = False,
) -> list[int]:
    """Normalize and validate PPO, baseline, SGBS, or source node indices."""
    normalized = [int(index) for index in indices]
    if any(index < 0 or index >= n_nodes for index in normalized):
        raise IndexError(
            f'{name} must be matrix indices in [0, {n_nodes}); got {normalized}'
        )
    if unique and len(set(normalized)) != len(normalized):
        raise ValueError(f'{name} contains duplicate node indices: {normalized}')
    return normalized


def validate_graph_index_space(
    G: nx.Graph,
    matrix_shapes: Sequence[tuple[int, ...]] = (),
    *,
    context: str = 'graph',
) -> None:
    """Check the inexpensive runtime part of the shared-index contract."""
    nodes = canonical_nodelist(G)
    expected_shape = (len(nodes), len(nodes))
    for shape in matrix_shapes:
        if tuple(shape) != expected_shape:
            raise ValueError(
                f'{context}: matrix shape {tuple(shape)} does not match '
                f'node index space {expected_shape}'
            )
