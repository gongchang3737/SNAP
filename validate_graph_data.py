# -*- coding: utf-8 -*-
"""Audit SNAP graph files against the shared node-index contract.

This command is read-only.  It verifies that every adjacency matrix and hop
distance matrix use the same row/column order, and optionally verifies every
delay realization and its metadata.

Examples
--------
python validate_graph_data.py
python validate_graph_data.py --graph-types IEEE_14Bus,IEEE_57Bus
python validate_graph_data.py --roots data/test_data --skip-delays
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from generate_sd_realizations import (
    discover_node_directories,
    graph_numbers,
    load_square_matrix,
    parse_csv,
    parse_int_csv,
)
from graph_indexing import (
    validate_adjacency_hop_alignment,
    validate_delay_distance,
)


def _numbered_indices(node_dir: Path, graph_num: int, stem: str) -> set[int]:
    pattern = re.compile(
        rf'^{re.escape(str(graph_num))}_{re.escape(stem)}_(\d+)\.npy$'
    )
    indices = set()
    for path in node_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None:
            indices.add(int(match.group(1)))
    return indices


def _metadata_sd(path: Path, expected_nodes: int) -> float:
    values = np.asarray(np.loadtxt(path), dtype=float).reshape(-1)
    if values.size != 3 or not np.isfinite(values).all():
        raise ValueError(
            f'{path} must contain [delay_sd, graph_param, n_nodes]'
        )
    if not np.isclose(values[2], expected_nodes):
        raise ValueError(
            f'{path} reports {values[2]} nodes; expected {expected_nodes}'
        )
    return float(values[0])


def audit_graph(node_dir: Path, graph_num: int, n_nodes: int,
                check_delays: bool, mu: float) -> int:
    adjacency_path = node_dir / f'{graph_num}.npy'
    hop_path = node_dir / f'{graph_num}_GDist.npy'
    adjacency = load_square_matrix(adjacency_path)
    hop_distance = load_square_matrix(hop_path)
    if adjacency.shape != (n_nodes, n_nodes):
        raise ValueError(
            f'{adjacency_path} has shape {adjacency.shape}; '
            f'expected {(n_nodes, n_nodes)}'
        )
    validate_adjacency_hop_alignment(
        adjacency,
        hop_distance,
        context=str(adjacency_path),
    )
    if not check_delays:
        return 0

    delay_indices = _numbered_indices(
        node_dir, graph_num, 'GDistDelay'
    )
    sd_indices = _numbered_indices(node_dir, graph_num, 'sd')
    if delay_indices != sd_indices:
        raise ValueError(
            f'{node_dir}/graph_{graph_num}: numbered delay indices '
            f'{sorted(delay_indices)} do not match SD indices '
            f'{sorted(sd_indices)}'
        )

    pairs = [
        (
            node_dir / f'{graph_num}_GDistDelay_{index}.npy',
            node_dir / f'{graph_num}_sd_{index}.npy',
        )
        for index in sorted(delay_indices)
    ]
    legacy_delay = node_dir / f'{graph_num}_GDistDelay.npy'
    legacy_sd = node_dir / f'{graph_num}_sd.npy'
    if legacy_delay.exists() != legacy_sd.exists():
        raise ValueError(
            f'{node_dir}/graph_{graph_num}: legacy delay/SD files are partial'
        )
    if legacy_delay.exists():
        pairs.append((legacy_delay, legacy_sd))
    if not pairs:
        raise FileNotFoundError(
            f'{node_dir}/graph_{graph_num}: no delay realization found'
        )

    for delay_path, sd_path in pairs:
        sd = _metadata_sd(sd_path, n_nodes)
        validate_delay_distance(
            load_square_matrix(delay_path),
            hop_distance,
            sd=sd,
            mu=mu,
            context=str(delay_path),
        )
    return len(pairs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--roots',
        default='data/train_data,data/test_data',
        help='Comma-separated dataset roots',
    )
    parser.add_argument('--graph-types', default=None)
    parser.add_argument('--sizes', default=None)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--skip-delays', action='store_true')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    roots = [Path(value.strip()) for value in args.roots.split(',') if value.strip()]
    graph_types = parse_csv(args.graph_types)
    sizes = parse_int_csv(args.sizes)
    graphs_checked = 0
    delays_checked = 0
    errors = []

    for root in roots:
        try:
            directories = list(discover_node_directories(
                root, graph_types, sizes
            ))
        except Exception as exc:
            errors.append(str(exc))
            continue
        for graph_type, n_nodes, node_dir in directories:
            for graph_num in graph_numbers(node_dir):
                try:
                    delays_checked += audit_graph(
                        node_dir,
                        graph_num,
                        n_nodes,
                        check_delays=not args.skip_delays,
                        mu=args.mu,
                    )
                    graphs_checked += 1
                except Exception as exc:
                    errors.append(
                        f'{root}/{graph_type}/{node_dir.name}/'
                        f'graph_{graph_num}: {exc}'
                    )

    print(
        f'Checked {graphs_checked} graph topologies and '
        f'{delays_checked} delay realizations.'
    )
    if errors:
        print(f'FAILED: {len(errors)} consistency error(s):')
        for error in errors:
            print(f'  - {error}')
        return 1
    print('PASS: all checked files share one canonical node-index order.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
