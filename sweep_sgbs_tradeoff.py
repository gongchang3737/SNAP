# -*- coding: utf-8 -*-
"""Sweep SGBS beta/gamma and build quality--time Pareto summaries.

This script never overwrites the normal ``run_experiment.py test`` outputs.
Use a held-out validation data directory for parameter selection, then rerun
only the selected operating points on the untouched test set.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev

import experiment_config as cfg
import utils
from OP_Tester import OP_Tester
from run_experiment import parse_bool, parse_csv, parse_int_csv, set_seed


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('expected a positive integer')
    return parsed


def parse_pairs(value):
    pairs = []
    for item in parse_csv(value):
        try:
            beta_text, gamma_text = item.split(':', maxsplit=1)
            pair = (positive_int(beta_text), positive_int(gamma_text))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f'invalid beta:gamma pair {item!r}'
            ) from exc
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def parameter_grid(args):
    if args.pairs:
        return parse_pairs(args.pairs)
    betas = parse_int_csv(args.beta_values)
    gammas = parse_int_csv(args.gamma_values)
    if any(value < 1 for value in betas + gammas):
        raise ValueError('all beta and gamma values must be positive')
    return [(beta, gamma) for beta in betas for gamma in gammas]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_method(args, graph_type, n_nodes, method, output_path,
               beta=1, gamma=1, num_graphs=None):
    sensor_budget = cfg.get_sensor_budget(n_nodes)
    method_params = cfg.build_method_params(
        model_id=args.model_id,
        model_method=args.model_method,
        include_baselines=False,
        include_neural=True,
        beta=beta,
        gamma=gamma,
        fixed_k=sensor_budget,
        feature_set=args.feature_set,
    )
    method_params['test_methods'] = [method]
    tester_params = cfg.build_tester_params(
        model_id=args.model_id,
        graph_type=graph_type,
        n_nodes=n_nodes,
        num_graphs=num_graphs or args.num_graphs,
        no_sd=args.no_sd,
        fixed_k=sensor_budget,
        feature_set=args.feature_set,
        data_path=args.data_path,
        results_save_path=str(output_path),
    )
    set_seed(args.seed)
    OP_Tester(tester_params, method_params, cfg.build_env_params()).run()
    with output_path.open(encoding='utf-8') as file:
        return json.load(file)[args.model_method][method]


def summarize_repeats(results):
    first_ranks = [float(value) for value in results[0]['rank']]
    ave_ranks = [float(result['ave_rank']) for result in results]
    total_times = [float(result['ave_time']) for result in results]
    placement_times = [float(result['ave_placement_time']) for result in results]
    feature_times = [float(result['ave_feature_time']) for result in results]
    search_times = [float(result['ave_search_time']) for result in results]
    model_load_times = [float(result['ave_model_load_time']) for result in results]
    evaluation_times = [float(result['ave_evaluation_time']) for result in results]
    return {
        'ave_rank': mean(ave_ranks),
        'rank_std_across_graphs': pstdev(first_ranks) if len(first_ranks) > 1 else 0.0,
        'ave_placement_time_seconds': mean(placement_times),
        'ave_feature_time_seconds': mean(feature_times),
        'ave_search_time_seconds': mean(search_times),
        'placement_time_std_across_repeats': (
            pstdev(placement_times) if len(placement_times) > 1 else 0.0
        ),
        'ave_model_load_time_seconds': mean(model_load_times),
        'ave_evaluation_time_seconds': mean(evaluation_times),
        'ave_total_time_seconds': mean(total_times),
    }


def mark_pareto(rows):
    for row in rows:
        dominated = any(
            other is not row
            and other['ave_rank'] <= row['ave_rank']
            and other['ave_placement_time_seconds'] <= row['ave_placement_time_seconds']
            and (
                other['ave_rank'] < row['ave_rank']
                or other['ave_placement_time_seconds'] < row['ave_placement_time_seconds']
            )
            for other in rows
        )
        row['pareto'] = not dominated
    return rows


def select_operating_points(rows):
    pareto = [row for row in rows if row['pareto']]
    if not pareto:
        return []
    fast = min(pareto, key=lambda row: row['ave_placement_time_seconds'])
    quality = min(pareto, key=lambda row: row['ave_rank'])

    min_rank = min(row['ave_rank'] for row in pareto)
    max_rank = max(row['ave_rank'] for row in pareto)
    min_time = min(row['ave_placement_time_seconds'] for row in pareto)
    max_time = max(row['ave_placement_time_seconds'] for row in pareto)

    def ideal_distance(row):
        rank_span = max(max_rank - min_rank, 1e-12)
        time_span = max(max_time - min_time, 1e-12)
        rank_norm = (row['ave_rank'] - min_rank) / rank_span
        time_norm = (row['ave_placement_time_seconds'] - min_time) / time_span
        return math.hypot(rank_norm, time_norm)

    balanced = min(pareto, key=ideal_distance)
    selected = []
    for label, row in (('fast', fast), ('balanced', balanced), ('quality', quality)):
        selected.append({
            'operating_point': label,
            'beta': row['beta'],
            'gamma': row['gamma'],
            'ave_rank': row['ave_rank'],
            'ave_placement_time_seconds': row['ave_placement_time_seconds'],
        })
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', required=True)
    parser.add_argument(
        '--model-method', default='PPOAgent',
        choices=['PPOAgent', 'GCN_DoubleDQN'],
    )
    parser.add_argument('--feature-set', default=utils.DEFAULT_GRAPH_FEATURE_SET,
                        choices=list(utils.GRAPH_FEATURE_SETS))
    parser.add_argument('--graph-types', default='BA,WS,ER')
    parser.add_argument('--test-sizes', default='20,50,100,200,300,500')
    parser.add_argument('--num-graphs', type=positive_int, default=30)
    parser.add_argument('--beta-values', default='1,2,4,8')
    parser.add_argument('--gamma-values', default='1,2,4,8,16')
    parser.add_argument(
        '--pairs', default=None,
        help='Optional comma-separated beta:gamma pairs; overrides the grid',
    )
    parser.add_argument('--data-path', default=cfg.TEST_DATA_PATH)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--timing-repeats', type=positive_int, default=1,
                        help='Use 3 for final timing tables')
    parser.add_argument('--warmup', action='store_true',
                        help='Run and discard one graph before each scenario/configuration')
    parser.add_argument('--skip-one-shot', action='store_true')
    parser.add_argument('--no-sd', '--no_sd', nargs='?', const=True,
                        type=parse_bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.feature_set = utils.normalize_feature_set(args.feature_set)
    args.data_path = os.path.join(args.data_path, '')
    graph_types = parse_csv(args.graph_types)
    test_sizes = parse_int_csv(args.test_sizes)
    pairs = parameter_grid(args)
    output_dir = Path(args.output_dir or (
        f'result/{args.model_id}/sgbs_tradeoff_{args.feature_set}'
    ))
    raw_dir = output_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    scenario_rows = []
    one_shot_rows = []
    for graph_type in graph_types:
        for n_nodes in test_sizes:
            scenario = f'{graph_type}_{n_nodes}'
            if not args.skip_one_shot:
                one_shot_results = []
                for repeat in range(args.timing_repeats):
                    path = raw_dir / f'{scenario}_one_shot_repeat{repeat + 1}.json'
                    one_shot_results.append(run_method(
                        args, graph_type, n_nodes, 'one_shot', path,
                    ))
                one_shot_rows.append({
                    'graph_type': graph_type,
                    'n_nodes': n_nodes,
                    **summarize_repeats(one_shot_results),
                })

            for beta, gamma in pairs:
                print(
                    f'Sweeping {scenario}: beta={beta}, gamma={gamma}',
                    flush=True,
                )
                if args.warmup:
                    warmup_path = raw_dir / (
                        f'{scenario}_beta{beta}_gamma{gamma}_warmup.json'
                    )
                    run_method(
                        args, graph_type, n_nodes, 'SGBS', warmup_path,
                        beta=beta, gamma=gamma, num_graphs=1,
                    )

                repeat_results = []
                for repeat in range(args.timing_repeats):
                    path = raw_dir / (
                        f'{scenario}_beta{beta}_gamma{gamma}_repeat{repeat + 1}.json'
                    )
                    repeat_results.append(run_method(
                        args, graph_type, n_nodes, 'SGBS', path,
                        beta=beta, gamma=gamma,
                    ))
                scenario_rows.append({
                    'graph_type': graph_type,
                    'n_nodes': n_nodes,
                    'beta': beta,
                    'gamma': gamma,
                    **summarize_repeats(repeat_results),
                })

    scenario_fields = [
        'graph_type', 'n_nodes', 'beta', 'gamma', 'ave_rank',
        'rank_std_across_graphs', 'ave_placement_time_seconds',
        'ave_feature_time_seconds', 'ave_search_time_seconds',
        'placement_time_std_across_repeats', 'ave_model_load_time_seconds',
        'ave_evaluation_time_seconds', 'ave_total_time_seconds',
    ]
    write_csv(output_dir / 'scenario_results.csv', scenario_fields, scenario_rows)

    if one_shot_rows:
        write_csv(
            output_dir / 'one_shot_results.csv',
            [field for field in scenario_fields if field not in ('beta', 'gamma')],
            one_shot_rows,
        )

    aggregate_rows = []
    for beta, gamma in pairs:
        selected = [
            row for row in scenario_rows
            if row['beta'] == beta and row['gamma'] == gamma
        ]
        aggregate_rows.append({
            'beta': beta,
            'gamma': gamma,
            'num_scenarios': len(selected),
            'ave_rank': mean(row['ave_rank'] for row in selected),
            'ave_placement_time_seconds': mean(
                row['ave_placement_time_seconds'] for row in selected
            ),
            'ave_feature_time_seconds': mean(
                row['ave_feature_time_seconds'] for row in selected
            ),
            'ave_search_time_seconds': mean(
                row['ave_search_time_seconds'] for row in selected
            ),
            'ave_total_time_seconds': mean(
                row['ave_total_time_seconds'] for row in selected
            ),
        })
    mark_pareto(aggregate_rows)
    aggregate_rows.sort(key=lambda row: (row['ave_placement_time_seconds'], row['ave_rank']))
    aggregate_fields = [
        'beta', 'gamma', 'num_scenarios', 'ave_rank',
        'ave_placement_time_seconds', 'ave_feature_time_seconds',
        'ave_search_time_seconds', 'ave_total_time_seconds', 'pareto',
    ]
    write_csv(output_dir / 'aggregate_results.csv', aggregate_fields, aggregate_rows)

    operating_points = select_operating_points(aggregate_rows)
    write_csv(
        output_dir / 'selected_operating_points.csv',
        ['operating_point', 'beta', 'gamma', 'ave_rank',
         'ave_placement_time_seconds'],
        operating_points,
    )
    metadata = {
        'model_id': args.model_id,
        'model_method': args.model_method,
        'feature_set': args.feature_set,
        'feature_names': utils.get_graph_feature_names(args.feature_set),
        'graph_types': graph_types,
        'test_sizes': test_sizes,
        'num_graphs': args.num_graphs,
        'pairs': pairs,
        'data_path': args.data_path,
        'no_sd': args.no_sd,
        'seed': args.seed,
        'timing_repeats': args.timing_repeats,
        'quality_metric': 'mean rank (lower is better)',
        'time_metric': 'placement/search time excluding model load and final rank evaluation',
    }
    with (output_dir / 'metadata.json').open('w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f'Wrote trade-off results to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
