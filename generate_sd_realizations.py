# -*- coding: utf-8 -*-
"""Generate multiple delay realizations for existing SNAP graph topologies.

Only delay-related files are created or removed.  The existing adjacency
matrix ``{graph_num}.npy`` and hop-distance matrix
``{graph_num}_GDist.npy`` are opened read-only and are never modified.

New files use zero-based realization indices::

    {graph_num}_GDistDelay_{delay_num}.npy
    {graph_num}_sd_{delay_num}.npy

The realization indices are split into three equal blocks with SD sampled
from (0, 0.1], (0.1, 0.3], and (0.3, 0.5], respectively.  Synthetic graphs
under the training and test roots receive 3 and 12 realizations.  Real graph
topologies (WDN_* and IEEE_*) receive 30 realizations in either root.

The script is dry-run by default.  Add ``--execute`` to write the new files
and, after every new realization for a graph has passed validation, delete
the two legacy unnumbered files for that graph.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import truncnorm

from graph_indexing import (
    validate_adjacency_hop_alignment,
    validate_delay_distance,
)


ADJACENCY_PATTERN = re.compile(r'^(\d+)\.npy$')
NODE_DIRECTORY_PATTERN = re.compile(r'^(\d+)_nodes$')
REAL_GRAPH_PREFIXES = ('WDN_', 'IEEE_')
SD_INTERVALS = (
    (0.0, 0.1),
    (0.1, 0.3),
    (0.3, 0.5),
)


@dataclass(frozen=True)
class DatasetPlan:
    root: Path
    synthetic_realizations: int


@dataclass
class GenerationSummary:
    graphs_seen: int = 0
    graphs_generated: int = 0
    realizations_written: int = 0
    realizations_reused: int = 0
    legacy_files_deleted: int = 0


def parse_csv(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parsed = {item.strip() for item in value.split(',') if item.strip()}
    return parsed or None


def parse_int_csv(value: str | None) -> set[int] | None:
    values = parse_csv(value)
    return None if values is None else {int(value) for value in values}


def is_real_graph(graph_type: str) -> bool:
    return graph_type.startswith(REAL_GRAPH_PREFIXES)


def realization_count(graph_type: str, plan: DatasetPlan,
                      real_realizations: int) -> int:
    count = real_realizations if is_real_graph(graph_type) else plan.synthetic_realizations
    if count <= 0 or count % 3 != 0:
        raise ValueError(
            f'Realization count must be a positive multiple of three, got {count}'
        )
    return count


def sd_interval(realization_index: int, count: int) -> tuple[float, float]:
    block_size = count // 3
    block = min(realization_index // block_size, 2)
    return SD_INTERVALS[block]


def sample_right_closed_uniform(rng: np.random.Generator,
                                lower: float, upper: float) -> float:
    """Sample from (lower, upper], retaining the interval after 6 decimals."""
    unit_right_closed = 1.0 - rng.random()  # (0, 1]
    value = round(lower + (upper - lower) * unit_right_closed, 6)
    return min(max(value, lower + 1e-6), upper)


def derived_rng(base_seed: int, dataset_root: Path, graph_type: str,
                n_nodes: int, graph_num: int,
                realization_index: int) -> np.random.Generator:
    identity = '|'.join((
        str(base_seed),
        # ``absolute`` is deterministic for this CLI and does not require the
        # Windows final-path metadata access that ``resolve`` may be denied in
        # sandboxed or network-mounted workspaces.
        str(dataset_root.absolute()).lower(),
        graph_type,
        str(n_nodes),
        str(graph_num),
        str(realization_index),
    ))
    digest = hashlib.sha256(identity.encode('utf-8')).digest()
    seed = int.from_bytes(digest[:8], byteorder='little', signed=False)
    return np.random.default_rng(seed)


def load_square_matrix(path: Path) -> np.ndarray:
    matrix = np.asarray(np.loadtxt(path), dtype=float)
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    elif matrix.ndim == 1:
        matrix = np.atleast_2d(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'Expected a square matrix in {path}, got {matrix.shape}')
    if not np.isfinite(matrix).all():
        raise ValueError(f'Non-finite value found in {path}')
    return matrix


def validate_topology(adjacency: np.ndarray, expected_nodes: int,
                      adjacency_path: Path,
                      distance_path: Path) -> np.ndarray:
    if adjacency.shape != (expected_nodes, expected_nodes):
        raise ValueError(
            f'{adjacency_path} has shape {adjacency.shape}, expected '
            f'{(expected_nodes, expected_nodes)}'
        )
    if not np.allclose(adjacency, adjacency.T, rtol=0.0, atol=1e-10):
        raise ValueError(f'Only undirected symmetric graphs are supported: {adjacency_path}')
    if np.any(adjacency < 0):
        raise ValueError(f'Negative adjacency entry found in {adjacency_path}')

    topology = csr_matrix((adjacency > 0).astype(np.int8))
    components, _ = connected_components(topology, directed=False)
    if components != 1:
        raise ValueError(f'Graph is disconnected: {adjacency_path}')

    hop_distance = load_square_matrix(distance_path)
    if hop_distance.shape != adjacency.shape:
        raise ValueError(
            f'{distance_path} has shape {hop_distance.shape}, expected {adjacency.shape}'
        )
    validate_adjacency_hop_alignment(
        adjacency,
        hop_distance,
        context=str(adjacency_path),
    )
    return hop_distance


def load_graph_metadata(node_dir: Path, graph_num: int,
                        expected_nodes: int) -> tuple[float, float]:
    """Return (graph_param, n_nodes), using legacy or numbered SD metadata."""
    candidates = [node_dir / f'{graph_num}_sd.npy']
    candidates.extend(sorted(
        node_dir.glob(f'{graph_num}_sd_*.npy'),
        key=lambda path: path.name,
    ))
    for path in candidates:
        if not path.exists():
            continue
        values = np.asarray(np.loadtxt(path), dtype=float).reshape(-1)
        if values.size != 3 or not np.isfinite(values).all():
            raise ValueError(
                f'{path} must contain [delay_sd, graph_param, n_nodes]'
            )
        graph_param, stored_nodes = float(values[1]), float(values[2])
        if not np.isclose(stored_nodes, expected_nodes):
            raise ValueError(
                f'{path} reports {stored_nodes} nodes, expected {expected_nodes}'
            )
        return graph_param, stored_nodes
    raise FileNotFoundError(
        f'No SD metadata found for graph {graph_num} in {node_dir}. '
        'The legacy file is required on the first run.'
    )


def generate_delay_distance(adjacency: np.ndarray, mu: float, sd: float,
                            rng: np.random.Generator) -> np.ndarray:
    upper_rows, upper_cols = np.nonzero(np.triu(adjacency > 0, k=1))
    if len(upper_rows) == 0:
        return np.zeros_like(adjacency, dtype=float)

    lower, upper = mu - 2.0 * sd, mu + 2.0 * sd
    delays = truncnorm.rvs(
        (lower - mu) / sd,
        (upper - mu) / sd,
        loc=mu,
        scale=sd,
        size=len(upper_rows),
        random_state=rng,
    )
    weighted = np.zeros_like(adjacency, dtype=float)
    weighted[upper_rows, upper_cols] = delays
    weighted[upper_cols, upper_rows] = delays
    distances = np.asarray(
        dijkstra(csr_matrix(weighted), directed=False, return_predecessors=False),
        dtype=float,
    )
    return distances


def atomic_savetxt(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + '.tmp')
    np.savetxt(temporary, values, fmt='%.10g')
    os.replace(temporary, path)


def validate_realization(distance_path: Path, metadata_path: Path,
                         n_nodes: int, lower: float, upper: float,
                         hop_distance: np.ndarray | None = None,
                         mu: float = 1.0) -> None:
    distances = load_square_matrix(distance_path)
    if distances.shape != (n_nodes, n_nodes):
        raise ValueError(
            f'{distance_path} has shape {distances.shape}, expected {(n_nodes, n_nodes)}'
        )
    if np.any(distances < 0) or not np.allclose(
        distances, distances.T, rtol=1e-8, atol=1e-9
    ):
        raise ValueError(f'Invalid delay-distance matrix: {distance_path}')
    if not np.allclose(np.diag(distances), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError(f'Nonzero diagonal in {distance_path}')

    metadata = np.asarray(np.loadtxt(metadata_path), dtype=float).reshape(-1)
    if metadata.size != 3 or not np.isfinite(metadata).all():
        raise ValueError(f'Invalid metadata: {metadata_path}')
    sd = float(metadata[0])
    if not (lower < sd <= upper):
        raise ValueError(
            f'{metadata_path} has sd={sd}, expected ({lower}, {upper}]'
        )
    if not np.isclose(metadata[2], n_nodes):
        raise ValueError(f'Wrong node count in {metadata_path}')
    if hop_distance is not None:
        validate_delay_distance(
            distances,
            hop_distance,
            sd=sd,
            mu=mu,
            context=str(distance_path),
        )


def discover_node_directories(root: Path, graph_types: set[str] | None,
                              sizes: set[int] | None) -> Iterable[tuple[str, int, Path]]:
    if not root.exists():
        raise FileNotFoundError(f'Dataset root does not exist: {root}')
    for graph_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        graph_type = graph_dir.name
        if graph_types is not None and graph_type not in graph_types:
            continue
        for node_dir in sorted(path for path in graph_dir.iterdir() if path.is_dir()):
            match = NODE_DIRECTORY_PATTERN.fullmatch(node_dir.name)
            if match is None:
                continue
            n_nodes = int(match.group(1))
            if sizes is not None and n_nodes not in sizes:
                continue
            yield graph_type, n_nodes, node_dir


def graph_numbers(node_dir: Path) -> Sequence[int]:
    numbers = []
    for path in node_dir.iterdir():
        match = ADJACENCY_PATTERN.fullmatch(path.name)
        if match is not None:
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def process_graph(plan: DatasetPlan, graph_type: str, n_nodes: int,
                  node_dir: Path, graph_num: int, count: int, mu: float,
                  base_seed: int, execute: bool, overwrite: bool,
                  summary: GenerationSummary) -> None:
    adjacency_path = node_dir / f'{graph_num}.npy'
    distance_path = node_dir / f'{graph_num}_GDist.npy'
    legacy_delay_path = node_dir / f'{graph_num}_GDistDelay.npy'
    legacy_sd_path = node_dir / f'{graph_num}_sd.npy'

    if not distance_path.exists():
        raise FileNotFoundError(f'Missing hop-distance matrix: {distance_path}')
    adjacency = load_square_matrix(adjacency_path)
    hop_distance = validate_topology(
        adjacency, n_nodes, adjacency_path, distance_path
    )
    graph_param, stored_nodes = load_graph_metadata(node_dir, graph_num, n_nodes)

    summary.graphs_seen += 1
    if not execute:
        return

    graph_changed = False
    for realization_index in range(count):
        lower, upper = sd_interval(realization_index, count)
        delay_path = node_dir / f'{graph_num}_GDistDelay_{realization_index}.npy'
        sd_path = node_dir / f'{graph_num}_sd_{realization_index}.npy'

        if delay_path.exists() and sd_path.exists() and not overwrite:
            validate_realization(
                delay_path,
                sd_path,
                n_nodes,
                lower,
                upper,
                hop_distance=hop_distance,
                mu=mu,
            )
            summary.realizations_reused += 1
            continue
        if (delay_path.exists() != sd_path.exists()) and not overwrite:
            raise RuntimeError(
                f'Partial realization found for graph {graph_num}, realization '
                f'{realization_index} in {node_dir}. Rerun with --overwrite.'
            )

        rng = derived_rng(
            base_seed,
            plan.root,
            graph_type,
            n_nodes,
            graph_num,
            realization_index,
        )
        sd = sample_right_closed_uniform(rng, lower, upper)
        delay_distance = generate_delay_distance(adjacency, mu, sd, rng)
        atomic_savetxt(delay_path, delay_distance)
        atomic_savetxt(
            sd_path,
            np.array([sd, graph_param, stored_nodes], dtype=float),
        )
        validate_realization(
            delay_path,
            sd_path,
            n_nodes,
            lower,
            upper,
            hop_distance=hop_distance,
            mu=mu,
        )
        summary.realizations_written += 1
        graph_changed = True

    # Delete legacy delay files only after all numbered realizations validate.
    for realization_index in range(count):
        lower, upper = sd_interval(realization_index, count)
        validate_realization(
            node_dir / f'{graph_num}_GDistDelay_{realization_index}.npy',
            node_dir / f'{graph_num}_sd_{realization_index}.npy',
            n_nodes,
            lower,
            upper,
            hop_distance=hop_distance,
            mu=mu,
        )
    for legacy_path in (legacy_delay_path, legacy_sd_path):
        if legacy_path.exists():
            legacy_path.unlink()
            summary.legacy_files_deleted += 1
    if graph_changed:
        summary.graphs_generated += 1


def process_plan(plan: DatasetPlan, real_realizations: int, mu: float,
                 base_seed: int, execute: bool, overwrite: bool,
                 graph_types: set[str] | None,
                 sizes: set[int] | None) -> GenerationSummary:
    summary = GenerationSummary()
    print(f'[{"EXECUTE" if execute else "DRY-RUN"}] scanning {plan.root}')
    for graph_type, n_nodes, node_dir in discover_node_directories(
        plan.root, graph_types, sizes
    ):
        count = realization_count(graph_type, plan, real_realizations)
        numbers = graph_numbers(node_dir)
        if not numbers:
            continue
        print(
            f'  {graph_type}/{node_dir.name}: {len(numbers)} graphs, '
            f'{count} delay realizations each',
            flush=True,
        )
        for graph_num in numbers:
            process_graph(
                plan=plan,
                graph_type=graph_type,
                n_nodes=n_nodes,
                node_dir=node_dir,
                graph_num=graph_num,
                count=count,
                mu=mu,
                base_seed=base_seed,
                execute=execute,
                overwrite=overwrite,
                summary=summary,
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', default='data/train_data')
    parser.add_argument('--test-root', default='data/test_data')
    parser.add_argument('--train-realizations', type=int, default=3)
    parser.add_argument('--test-realizations', type=int, default=12)
    parser.add_argument('--real-realizations', type=int, default=30)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--graph-types', default=None,
        help='Optional comma-separated graph directory names',
    )
    parser.add_argument(
        '--sizes', default=None,
        help='Optional comma-separated node sizes',
    )
    parser.add_argument(
        '--execute', action='store_true',
        help='Write numbered realizations and delete validated legacy files',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Regenerate existing numbered realizations deterministically',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mu <= 0:
        raise ValueError('--mu must be positive')
    plans = (
        DatasetPlan(Path(args.train_root), args.train_realizations),
        DatasetPlan(Path(args.test_root), args.test_realizations),
    )
    graph_types = parse_csv(args.graph_types)
    sizes = parse_int_csv(args.sizes)

    total = GenerationSummary()
    for plan in plans:
        summary = process_plan(
            plan=plan,
            real_realizations=args.real_realizations,
            mu=args.mu,
            base_seed=args.seed,
            execute=args.execute,
            overwrite=args.overwrite,
            graph_types=graph_types,
            sizes=sizes,
        )
        for field in total.__dataclass_fields__:
            setattr(total, field, getattr(total, field) + getattr(summary, field))

    mode = 'completed' if args.execute else 'preview completed; no files changed'
    print('-' * 72)
    print(f'{mode}')
    print(f'graphs seen:              {total.graphs_seen}')
    print(f'graphs generated/updated: {total.graphs_generated}')
    print(f'realizations written:     {total.realizations_written}')
    print(f'realizations reused:      {total.realizations_reused}')
    print(f'legacy files deleted:     {total.legacy_files_deleted}')
    if not args.execute:
        print('Run again with --execute after reviewing this preview.')


if __name__ == '__main__':
    main()
