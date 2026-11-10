# -*- coding: utf-8 -*-
"""
data_generator.py — Synthetic & Real-World Graph Dataset Generator for SNAP

Generates graph instances compatible with the existing data_loader (utils.py).
Each graph instance produces 4 files:
    {index}.npy          — adjacency matrix       (np.savetxt)
    {index}_GDist.npy     — unweighted shortest-path distances
    {index}_GDistDelay.npy— delay-weighted shortest-path distances
    {index}_sd.npy        — [delay_sd, graph_param, n_nodes]

Usage:
    python data_generator.py                    # default: all synthetic types
    python data_generator.py --type BA --nodes 50 --num 2000
    python data_generator.py --regular-test     # BA/ER/GEO/WS, 50..200 nodes, under data/test_data
    python data_generator.py --snap-test-data   # BA/ER/GEO/WS, framework test sizes, under data/test_data
    python data_generator.py --snap-train-data  # BA/ER/GEO/WS, 20/50/100 nodes, under data/train_data
    python data_generator.py --real             # attempt real-world graphs
"""

import numpy as np
import networkx as nx
import os
import re
import sys
import argparse
from scipy.stats import truncnorm

from graph_indexing import (
    adjacency_from_graph,
    all_pair_distances_from_graph,
    relabel_to_canonical_indices,
    validate_adjacency_hop_alignment,
    validate_delay_distance,
)


# ──────────────────────────────────────────────
#  Core I/O helpers
# ──────────────────────────────────────────────

def add_delay(G, mu=1.0, sd=0.3):
    """Assign truncated-Gaussian edge delays (matching utils.addDelay)."""
    if G.number_of_edges() == 0:
        return G
    lower, upper = mu - 2 * sd, mu + 2 * sd
    delays = truncnorm.rvs(
        (lower - mu) / sd, (upper - mu) / sd,
        loc=mu, scale=sd, size=G.number_of_edges()
    )
    for (u, v), d in zip(G.edges(), delays):
        G[u][v]['delay'] = d
    return G


def compute_all_pair_distances(G):
    """Return APSP matrices using the canonical 0..N-1 matrix order."""
    return all_pair_distances_from_graph(G)


def save_one_graph(output_dir, idx, G, adj, dist, dist_delay, sd, graph_param,
                   n_nodes, mu=1.0):
    """Persist one graph instance as 4 .npy text files."""
    context = f'{output_dir}/graph_{idx}'
    validate_adjacency_hop_alignment(adj, dist, context=context)
    validate_delay_distance(
        dist_delay,
        dist,
        sd=sd,
        mu=mu,
        context=context,
    )
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, f'{idx}.npy'), adj, fmt='%.10g')
    np.savetxt(os.path.join(output_dir, f'{idx}_GDist.npy'), dist, fmt='%.10g')
    np.savetxt(os.path.join(output_dir, f'{idx}_GDistDelay.npy'), dist_delay, fmt='%.10g')
    # _sd.npy stores [delay_sd, graph_param, n_nodes]
    np.savetxt(os.path.join(output_dir, f'{idx}_sd.npy'),
               np.array([sd, graph_param, n_nodes]), fmt='%.10g')


# ──────────────────────────────────────────────
#  Synthetic graph builders
# ──────────────────────────────────────────────

def build_ba(n, m, mu=1.0, sd=0.3, seed=None):
    """Barabási-Albert scale-free graph."""
    if seed is not None:
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    else:
        G = nx.barabasi_albert_graph(n, m)
    adj = adjacency_from_graph(G)
    G = add_delay(G, mu, sd)
    return G, adj


def build_er(n, p, mu=1.0, sd=0.3, seed=None, max_retry=200):
    """Erdős-Rényi random graph (connected). Retries until connected."""
    rng = np.random.RandomState(seed)
    for _ in range(max_retry):
        G = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 2**31 - 1))
        if nx.is_connected(G):
            break
    else:
        # fallback: add edges to largest component
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0]).copy()
        # add new nodes and edges to restore size n
        while G.number_of_nodes() < n:
            new_node = G.number_of_nodes()
            G.add_node(new_node)
            target = np.random.choice(list(G.nodes()))
            G.add_edge(new_node, target)

    adj = adjacency_from_graph(G)
    G = add_delay(G, mu, sd)
    return G, adj


def build_ws(n, k, p_rewire, mu=1.0, sd=0.3, seed=None, max_retry=100):
    """Watts-Strogatz small-world graph (may disconnect at high p, retry)."""
    rng = np.random.RandomState(seed)
    for _ in range(max_retry):
        G = nx.watts_strogatz_graph(n, k, p_rewire, seed=rng.randint(0, 2**31 - 1))
        if nx.is_connected(G):
            break
    adj = adjacency_from_graph(G)
    G = add_delay(G, mu, sd)
    return G, adj


def build_geometric(n, radius, dim=2, mu=1.0, sd=0.3, seed=None, max_retry=500):
    """Random Geometric Graph (connected)."""
    rng = np.random.RandomState(seed)
    for _ in range(max_retry):
        G = nx.random_geometric_graph(n, radius, dim=dim, seed=rng.randint(0, 2**31 - 1))
        if nx.is_connected(G):
            break
    else:
        # fallback: increase radius
        G = nx.random_geometric_graph(n, radius * 1.5, dim=dim)
    adj = adjacency_from_graph(G)
    G = add_delay(G, mu, sd)
    return G, adj


# ──────────────────────────────────────────────
#  Parameter samplers (uniform over sensible ranges)
# ──────────────────────────────────────────────

def sample_er_params(n):
    """Return (p) ensuring average degree roughly [4, 0.3*n]."""
    p_min = max(0.05, 4.0 / n)
    p_max = min(0.4, 0.3)
    p = np.random.uniform(p_min, p_max)
    return p


def sample_ba_params(n):
    """Return (m) with 2 <= m < n."""
    m_max = min(10, n - 1)
    m = np.random.randint(2, m_max + 1)
    return m


def sample_ws_params(n):
    """Return (k, p_rewire)."""
    k_max = min(8, n - 1)
    k = np.random.choice(range(2, k_max + 1, 2))  # even
    if k < 2:
        k = 2
    p_rewire = np.random.uniform(0.1, 0.4)
    return k, p_rewire


def sample_geometric_params(n, dim=2):
    """Return (radius) guided by the percolation threshold."""
    # approximate critical radius for connectivity in RGG
    r_crit = np.sqrt(np.log(n) / n)
    radius = np.random.uniform(1.2 * r_crit, 3.5 * r_crit)
    return float(radius)


# ──────────────────────────────────────────────
#  Batch generation
# ──────────────────────────────────────────────

def generate_synthetic_dataset(graph_type, n_nodes, num_graphs,
                               base_dir='data/train_data',
                               mu=1.0, sd_range=(0.1, 0.5), seed=42):
    """
    Generate a folder of graph instances.

    Parameters
    ----------
    graph_type : str  — one of {'BA', 'ER', 'WS', 'GEO'}
    n_nodes    : int  — number of nodes per graph
    num_graphs : int  — how many instances to generate
    base_dir   : str  — root output directory
    mu         : float— mean edge delay
    sd_range   : tuple— (min_sd, max_sd) delay std. dev. uniformly sampled per graph
    seed       : int  — base random seed
    """
    output_dir = os.path.join(base_dir, graph_type, f'{n_nodes}_nodes')
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    for idx in range(num_graphs):
        sd = np.round(rng.uniform(*sd_range), 6)
        g_seed = rng.randint(0, 2**31 - 1)

        if graph_type == 'BA':
            m = sample_ba_params(n_nodes)
            G, adj = build_ba(n_nodes, m, mu, sd, seed=g_seed)
            graph_param = m

        elif graph_type == 'ER':
            p = sample_er_params(n_nodes)
            G, adj = build_er(n_nodes, p, mu, sd, seed=g_seed)
            graph_param = round(p, 6)

        elif graph_type == 'WS':
            k, p_rewire = sample_ws_params(n_nodes)
            G, adj = build_ws(n_nodes, k, p_rewire, mu, sd, seed=g_seed)
            graph_param = round(p_rewire, 6)  # save rewire prob

        elif graph_type == 'GEO':
            radius = sample_geometric_params(n_nodes)
            G, adj = build_geometric(n_nodes, radius, mu=mu, sd=sd, seed=g_seed)
            graph_param = round(radius, 6)

        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")

        dist, dist_delay = compute_all_pair_distances(G)

        save_one_graph(output_dir, idx, G, adj, dist, dist_delay,
                       sd, graph_param, n_nodes, mu=mu)

        if (idx + 1) % max(1, num_graphs // 10) == 0:
            print(f'  [{graph_type} n={n_nodes}] {idx+1}/{num_graphs} done')

    print(f'  [{graph_type} n={n_nodes}] complete → {output_dir}')


def generate_all_synthetic(base_dir='data/train_data',
                           num_graphs=1000,
                           n_nodes_list=(20, 50, 100),
                           graph_types=('BA', 'ER', 'WS', 'GEO')):
    """Convenience wrapper: generate all synthetic datasets."""
    print(f'Generating synthetic datasets (num_graphs={num_graphs})')
    print(f'  Graph types: {graph_types}')
    print(f'  Node sizes:  {n_nodes_list}')
    print('-' * 50)
    for gt in graph_types:
        for n in n_nodes_list:
            generate_synthetic_dataset(gt, n, num_graphs, base_dir=base_dir)


def generate_regular_test_data(base_dir='data/test_data',
                               num_graphs=100,
                               n_nodes_list=tuple(range(50, 201, 10)),
                               graph_types=('BA', 'ER', 'GEO', 'WS'),
                               seed=2026):
    """Generate regular test graphs under data/test_data for BA/ER/GEO/WS."""
    print('Generating regular SNAP test datasets')
    print(f'  Output root: {base_dir}')
    print(f'  Graph types: {graph_types}')
    print(f'  Node sizes:  {n_nodes_list}')
    print(f'  Graphs per size: {num_graphs}')
    print('-' * 50)

    for type_idx, graph_type in enumerate(graph_types):
        for node_idx, n_nodes in enumerate(n_nodes_list):
            split_seed = seed + type_idx * 100000 + node_idx * 1000
            generate_synthetic_dataset(
                graph_type,
                n_nodes,
                num_graphs,
                base_dir=base_dir,
                seed=split_seed,
            )


def generate_snap_test_data(base_dir='data/test_data',
                                 num_graphs=100,
                                 seed=2026):
    """Generate SNAP test_data: BA/ER/GEO/WS at framework test sizes."""
    generate_regular_test_data(
        base_dir=base_dir,
        num_graphs=num_graphs,
        n_nodes_list=(20, 50, 100, 150, 200, 300, 400, 500, 600),
        graph_types=('BA', 'ER', 'GEO', 'WS'),
        seed=seed,
    )


def generate_snap_train_data(base_dir='data/train_data',
                                  num_graphs=100,
                                  seed=2026):
    """Generate SNAP train_data: BA/ER/GEO/WS at 20, 50, and 100 nodes."""
    generate_regular_test_data(
        base_dir=base_dir,
        num_graphs=num_graphs,
        n_nodes_list=(20, 50, 100),
        graph_types=('BA', 'ER', 'GEO', 'WS'),
        seed=seed,
    )


# ──────────────────────────────────────────────
#  Real-world graph importers
# ──────────────────────────────────────────────

def _ensure_connected(G):
    """Return the largest connected component of G."""
    if nx.is_connected(G):
        return G
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def _save_real_graph(output_dir, name, G, mu=1.0, sd=0.3, graph_param=0):
    """Save a single real-world graph as instance 0."""
    G = _ensure_connected(G)
    # Matrix row i, distance row i, PPO action i, and reported sensor i must
    # all identify the same node.  Insertion order is never used as an index.
    G = relabel_to_canonical_indices(G)
    G = add_delay(G, mu, sd)
    n = G.number_of_nodes()
    adj = adjacency_from_graph(G)
    dist, dist_delay = compute_all_pair_distances(G)
    os.makedirs(output_dir, exist_ok=True)
    save_one_graph(
        output_dir, 0, G, adj, dist, dist_delay,
        sd, graph_param, n, mu=mu,
    )
    print(f'  [{name}] {n} nodes → {output_dir}')


# ──────────────────────────────────────────────
#  Lightweight EPANET .inp parser (no wntr dependency)
# ──────────────────────────────────────────────

def _parse_inp_section(content, section_tag):
    """Extract lines of an EPANET .inp section [TAG] ...
    Returns list of lines between [TAG] and the next [...] or [END]."""
    pattern = re.escape(section_tag) + r'\s*\n(.*?)(?=\n\s*\[|$)'
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    lines = match.group(1).strip().split('\n')
    # filter out comments and empty lines
    result = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        result.append(line)
    return result


def _parse_inp_node_ids(content, section_tag):
    """Extract node IDs from a [JUNCTIONS], [RESERVOIRS], or [TANKS] section."""
    ids = set()
    for line in _parse_inp_section(content, section_tag):
        parts = line.split()
        if parts:
            ids.add(parts[0])
    return ids


def _parse_inp_edges(content, section_tag):
    """Extract (node1, node2) edges from [PIPES], [PUMPS], or [VALVES]."""
    edges = []
    for line in _parse_inp_section(content, section_tag):
        parts = line.split()
        if len(parts) >= 3:
            edges.append((parts[1], parts[2]))
    return edges


def parse_epanet_inp(filepath):
    """Parse an EPANET .inp file and return a networkx Graph of the topology."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    nodes = set()
    for tag in ['[JUNCTIONS]', '[RESERVOIRS]', '[TANKS]']:
        nodes.update(_parse_inp_node_ids(content, tag))

    edges = []
    for tag in ['[PIPES]', '[PUMPS]', '[VALVES]']:
        edges.extend(_parse_inp_edges(content, tag))

    G = nx.Graph()
    for u, v in edges:
        if u in nodes and v in nodes:
            G.add_edge(u, v)

    for n in nodes:
        if n not in G:
            G.add_node(n)

    # Relabel to 0..n-1 integer IDs for compatibility with compute_all_pair_distances
    mapping = {old: new for new, old in enumerate(sorted(nodes, key=str))}
    G = nx.relabel_nodes(G, mapping, copy=True)
    return G


def load_epanet_water_network(net_name, base_dir='data/train_data', mu=1.0, sd=0.3):
    """Load an EPANET water distribution network from local .inp file.
    Supports Net1, Net2, Net3 (files expected in data/epanet/)."""
    inp_path = os.path.join(os.path.dirname(__file__) or '.',
                            'data', 'epanet', f'{net_name}.inp')
    if not os.path.exists(inp_path):
        print(f'[SKIP] {inp_path} not found. Download from EPANET examples.')
        return

    G = parse_epanet_inp(inp_path)
    n = G.number_of_nodes()
    output_dir = os.path.join(base_dir, f'WDN_{net_name}', f'{n}_nodes')
    _save_real_graph(output_dir, f'WDN_{net_name}', G, mu, sd, graph_param=0)


def build_ieee_bus_graph(case=30):
    """
    Build IEEE standard bus system as a networkx graph from known adjacency.
    Supports case ∈ {14, 30, 57, 118}.

    Node indices are 0-based internally.
    """
    # ── IEEE 30-Bus adjacency (branch list: from, to; 1-indexed) ──
    IEEE30_BRANCHES = [
        (1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (2, 6), (4, 6),
        (5, 7), (6, 7), (6, 8), (6, 9), (6, 10), (9, 11), (9, 10),
        (4, 12), (12, 13), (12, 14), (12, 15), (12, 16), (14, 15),
        (16, 17), (15, 18), (18, 19), (19, 20), (10, 20), (10, 17),
        (10, 21), (10, 22), (21, 22), (15, 23), (22, 24), (23, 24),
        (24, 25), (25, 26), (25, 27), (28, 27), (27, 29), (27, 30),
        (29, 30), (8, 28), (6, 28),
    ]

    IEEE118_BRANCHES = [
        (1, 2), (1, 3), (2, 12), (3, 5), (3, 12), (4, 5), (4, 11),
        (5, 6), (5, 8), (5, 11), (6, 7), (7, 12), (8, 9), (8, 30),
        (9, 10), (11, 12), (11, 13), (12, 14), (12, 16), (12, 117),
        (13, 15), (14, 15), (15, 17), (15, 19), (15, 33), (16, 17),
        (17, 18), (17, 30), (17, 31), (17, 113), (18, 19), (19, 20),
        (19, 34), (20, 21), (21, 22), (22, 23), (23, 24), (23, 25),
        (23, 32), (24, 70), (24, 72), (25, 26), (25, 27), (26, 30),
        (27, 28), (27, 32), (27, 115), (28, 29), (29, 31), (30, 38),
        (31, 32), (32, 113), (32, 114), (33, 37), (34, 36), (34, 37),
        (34, 43), (35, 36), (35, 37), (37, 38), (37, 39), (37, 40),
        (38, 65), (39, 40), (40, 41), (40, 42), (41, 42), (42, 49),
        (43, 44), (44, 45), (45, 46), (45, 49), (46, 47), (46, 48),
        (47, 49), (47, 69), (48, 49), (49, 50), (49, 51), (49, 54),
        (49, 66), (49, 69), (50, 57), (51, 52), (51, 58), (52, 53),
        (53, 54), (54, 55), (54, 56), (54, 59), (55, 56), (55, 59),
        (56, 57), (56, 58), (56, 59), (59, 60), (59, 61), (59, 63),
        (60, 61), (60, 62), (61, 62), (61, 64), (62, 66), (62, 67),
        (63, 64), (64, 65), (65, 66), (65, 68), (66, 67), (68, 69),
        (68, 81), (68, 116), (69, 70), (69, 75), (69, 77), (70, 71),
        (70, 74), (70, 75), (71, 72), (71, 73), (74, 75), (75, 77),
        (75, 118), (76, 77), (76, 118), (77, 78), (77, 80), (77, 82),
        (78, 79), (79, 80), (80, 81), (80, 96), (80, 97), (80, 98),
        (80, 99), (82, 83), (82, 96), (83, 84), (83, 85), (84, 85),
        (85, 86), (85, 88), (85, 89), (86, 87), (88, 89), (89, 90),
        (89, 92), (90, 91), (91, 92), (92, 93), (92, 94), (92, 100),
        (92, 102), (93, 94), (94, 95), (94, 96), (94, 100), (95, 96),
        (96, 97), (98, 100), (99, 100), (100, 101), (100, 103),
        (100, 104), (100, 106), (101, 102), (103, 104), (103, 105),
        (103, 110), (104, 105), (105, 106), (105, 107), (105, 108),
        (106, 107), (108, 109), (109, 110), (110, 111), (110, 112),
        (114, 115),
    ]

    IEEE14_BRANCHES = [
        (1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5),
        (4, 7), (4, 9), (5, 6), (6, 11), (6, 12), (6, 13), (7, 8),
        (7, 9), (9, 10), (9, 14), (10, 11), (12, 13), (13, 14),
    ]

    IEEE57_BRANCHES = [
        (1, 2), (1, 15), (1, 16), (1, 17), (2, 3), (3, 4), (3, 15),
        (4, 5), (4, 6), (4, 18), (5, 6), (6, 7), (6, 8), (7, 8),
        (7, 29), (8, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 55),
        (10, 12), (10, 51), (11, 13), (11, 41), (11, 43), (12, 13),
        (12, 16), (12, 17), (13, 14), (13, 15), (13, 49), (14, 15),
        (14, 46), (15, 45), (18, 19), (19, 20), (20, 21), (21, 22),
        (22, 23), (22, 38), (23, 24), (24, 25), (24, 26), (25, 30),
        (26, 27), (27, 28), (28, 29), (29, 52), (30, 31), (31, 32),
        (32, 33), (32, 34), (34, 35), (35, 36), (36, 37), (36, 40),
        (37, 38), (37, 39), (38, 44), (38, 48), (38, 49), (39, 57),
        (40, 56), (41, 42), (41, 43), (41, 56), (42, 56), (44, 45),
        (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (52, 53),
        (53, 54), (54, 55), (56, 57),
    ]

    branches_map = {
        14: IEEE14_BRANCHES,
        30: IEEE30_BRANCHES,
        57: IEEE57_BRANCHES,
        118: IEEE118_BRANCHES,
    }

    if case not in branches_map:
        raise ValueError(f"IEEE {case}-bus not supported. Choose from {list(branches_map.keys())}")

    branches = branches_map[case]
    G = nx.Graph()
    G.add_edges_from([(u - 1, v - 1) for (u, v) in branches])  # 0-indexed
    return G


def load_ieee_bus(case=30, base_dir='data/train_data', mu=1.0, sd=0.3):
    """Generate and save an IEEE bus system graph."""
    G = build_ieee_bus_graph(case)
    n = G.number_of_nodes()
    output_dir = os.path.join(base_dir, f'IEEE_{case}Bus', f'{n}_nodes')
    _save_real_graph(output_dir, f'IEEE_{case}Bus', G, mu, sd, graph_param=case)


def load_all_real(base_dir='data/train_data', mu=1.0, sd=0.3):
    """Generate all real-world graphs (water distribution + power)."""
    print('Generating real-world graph datasets...')
    print('-' * 50)

    # Water Distribution Networks (local .inp parser, no wntr needed)
    for net_name in ['Net1', 'Net2', 'Net3']:
        load_epanet_water_network(net_name, base_dir, mu, sd)

    # IEEE Power Bus Systems
    for case in [14, 30, 57, 118]:
        load_ieee_bus(case, base_dir, mu, sd)

    print('Done.')


# ──────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SNAP Graph Dataset Generator'
    )
    parser.add_argument('--type', type=str, default=None,
                        help='Graph type: BA, ER, WS, GEO (default: all)')
    parser.add_argument('--nodes', type=int, default=None,
                        help='Number of nodes (for single-type mode)')
    parser.add_argument('--num', type=int, default=1000,
                        help='Number of graph instances per config (default: 1000)')
    parser.add_argument('--real', action='store_true',
                        help='Generate real-world graphs (EPANET, IEEE bus)')
    parser.add_argument('--regular-test', action='store_true',
                        help='Generate BA/ER/GEO/WS regular test data at 50..200 nodes under data/test_data')
    parser.add_argument('--snap-test-data', action='store_true',
                        help='Generate BA/ER/GEO/WS at framework test sizes under data/test_data')
    parser.add_argument('--snap-train-data', action='store_true',
                        help='Generate BA/ER/GEO/WS at 20,50,100 nodes under data/train_data')
    parser.add_argument('--base-dir', type=str, default='data/train_data',
                        help='Output base directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.regular_test:
        base_dir = args.base_dir
        if base_dir == 'data/train_data':
            base_dir = 'data/test_data'
        generate_regular_test_data(base_dir=base_dir, num_graphs=args.num, seed=args.seed)
        return

    if args.snap_test_data:
        base_dir = args.base_dir
        if base_dir == 'data/train_data':
            base_dir = 'data/test_data'
        generate_snap_test_data(base_dir=base_dir, num_graphs=args.num, seed=args.seed)
        return

    if args.snap_train_data:
        base_dir = args.base_dir
        generate_snap_train_data(base_dir=base_dir, num_graphs=args.num, seed=args.seed)
        return

    if args.real:
        load_all_real(base_dir=args.base_dir)
        return

    if args.type is not None:
        if args.nodes is None:
            print('Error: --nodes required when --type is specified.')
            sys.exit(1)
        gt = args.type.upper()
        generate_synthetic_dataset(gt, args.nodes, args.num, base_dir=args.base_dir)
    else:
        generate_all_synthetic(
            base_dir=args.base_dir,
            num_graphs=args.num,
            n_nodes_list=(20, 50, 100),
            graph_types=('BA', 'ER', 'WS', 'GEO'),
        )


if __name__ == '__main__':
    main()
