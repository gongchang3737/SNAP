# -*- coding: utf-8 -*-
"""Command-line runner for SNAP experiment framework."""

import argparse
import os
import random
import re

import numpy as np
import torch

import experiment_config as cfg
import utils
from OP_Trainer import OP_Trainer
from OP_Tester import OP_Tester


SGBS_RESULTS_DIRS = {
    'hop': os.path.join('result', 'hop_sgbs'),
    'mc': os.path.join('result', 'mc_sgbs'),
    'pmc': os.path.join('result', 'pmc_sgbs'),
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_csv(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_int_csv(value):
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ('true', '1', 'yes', 'y', 'on'):
        return True
    if normalized in ('false', '0', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError('expected true or false')


def build_explicit_results_path(args, graph_type, n_nodes, sensor_budget):
    """Resolve an explicit or method-specific SGBS result path.

    An explicitly supplied ``--results-dir`` always wins.  Otherwise SD tests
    that include SGBS are routed by their complete-placement scoring mode.
    one-shot-only, baseline-only, and no-SD runs keep their historical paths.
    """
    neural_methods = parse_csv(args.neural_methods)
    use_sgbs_default = (
        args.results_dir is None
        and not args.no_sd
        and not args.only_baselines
        and 'SGBS' in neural_methods
    )
    if (
        args.results_dir is None
        and args.result_tag is None
        and not use_sgbs_default
    ):
        return None
    if args.result_tag is None:
        methods = '-'.join(neural_methods)
        if args.sd_score_mode in ('mc', 'pmc'):
            resolved_search_seed = (
                args.seed if args.search_seed is None else args.search_seed
            )
            conditional = (
                f'_{args.known_sd_level}'
                if args.sd_score_mode == 'pmc' else ''
            )
            tag = (
                f'{methods}_{args.sd_score_mode}{conditional}_'
                f'M{args.search_scenarios}_seed{resolved_search_seed}'
            )
        else:
            tag = f'{methods}_hop_seed{args.seed}'
    else:
        tag = args.result_tag
    if re.fullmatch(r'[A-Za-z0-9_.-]+', tag) is None:
        raise ValueError(
            '--result-tag may contain only letters, digits, dot, underscore, '
            'and hyphen'
        )
    if args.results_dir is not None:
        directory = args.results_dir
    elif use_sgbs_default:
        directory = SGBS_RESULTS_DIRS[args.sd_score_mode]
    else:
        directory = os.path.join('result', args.model_id)
    filename = (
        f'{args.model_id}_{graph_type}_{n_nodes}_k{sensor_budget}_{tag}.json'
    )
    return os.path.join(directory, filename)


def train(args):
    set_seed(args.seed)
    trainer_params = cfg.build_trainer_params(
        model_id=args.model_id,
        model_method=args.model_method,
        graph_types=parse_csv(args.graph_types),
        n_nodes_list=parse_int_csv(args.train_sizes),
        num_graphs=args.num_graphs,
        no_sd=args.no_sd,
        num_episodes=args.num_episodes,
        prompt_record=args.prompt_record,
        feature_set=args.feature_set,
        seed=args.seed,
    )
    model_params = cfg.build_model_params(
        args.model_method,
        feature_set=args.feature_set,
    )
    env_params = cfg.build_env_params()
    OP_Trainer(trainer_params, model_params, env_params).run()


def test(args):
    if args.no_baselines and args.only_baselines:
        raise ValueError('--no-baselines and --only-baselines cannot be used together')
    if args.beta < 1 or args.gamma < 1:
        raise ValueError('--beta and --gamma must both be positive integers')
    if args.sd_score_mode in ('mc', 'pmc') and args.no_sd:
        raise ValueError(
            f'--sd-score-mode {args.sd_score_mode} requires --no-sd False'
        )
    if args.sd_score_mode == 'mc' and args.search_scenarios % 3 != 0:
        raise ValueError('--search-scenarios must be a positive multiple of 3')
    if args.search_scenarios <= 0:
        raise ValueError('--search-scenarios must be positive')
    if args.delay_mu <= 0:
        raise ValueError('--delay-mu must be positive')
    if args.sd_score_mode != 'pmc' and args.known_sd_level != 'all':
        raise ValueError(
            '--known-sd-level is only meaningful with --sd-score-mode pmc'
        )

    set_seed(args.seed)
    graph_types = parse_csv(args.graph_types)
    test_sizes = parse_int_csv(args.test_sizes)
    neural_methods = parse_csv(args.neural_methods)
    unknown_neural_methods = set(neural_methods) - {'one_shot', 'SGBS'}
    if unknown_neural_methods:
        raise ValueError(
            '--neural-methods only supports one_shot and SGBS; got '
            f'{sorted(unknown_neural_methods)}'
        )
    env_params = cfg.build_env_params()
    search_seed = args.seed if args.search_seed is None else args.search_seed
    for graph_type in graph_types:
        for n_nodes in test_sizes:
            sensor_budget = cfg.get_sensor_budget(n_nodes, override=args.fixed_k)
            method_params = cfg.build_method_params(
                model_id=args.model_id,
                model_method=args.model_method,
                include_baselines=not args.no_baselines,
                include_neural=not args.only_baselines,
                beta=args.beta,
                gamma=args.gamma,
                fixed_k=sensor_budget,
                feature_set=args.feature_set,
                neural_methods=neural_methods,
                sd_score_mode=args.sd_score_mode,
                search_scenarios=args.search_scenarios,
                search_seed=search_seed,
                sd_prior=args.sd_prior,
                delay_mu=args.delay_mu,
                known_sd_level=args.known_sd_level,
                topology_prefix=f'{graph_type}/{n_nodes}_nodes',
                run_seed=args.seed,
            )
            print(
                f'Testing model={args.model_id}, graph_type={graph_type}, '
                f'n_nodes={n_nodes}, sensors={sensor_budget}, no_sd={args.no_sd}, '
                f'feature_set={args.feature_set}, '
                f'sd_score_mode={args.sd_score_mode}, '
                f'search_scenarios={args.search_scenarios}, '
                f'search_seed={search_seed}'
                f', known_sd_level={args.known_sd_level}'
            )
            tester_params = cfg.build_tester_params(
                model_id=args.model_id,
                graph_type=graph_type,
                n_nodes=n_nodes,
                num_graphs=args.num_graphs,
                no_sd=args.no_sd,
                fixed_k=sensor_budget,
                feature_set=args.feature_set,
                sd_score_mode=args.sd_score_mode,
                search_scenarios=args.search_scenarios,
                search_seed=search_seed,
                known_sd_level=args.known_sd_level,
                results_save_path=build_explicit_results_path(
                    args, graph_type, n_nodes, sensor_budget
                ),
            )
            OP_Tester(tester_params, method_params, env_params).run()


def build_parser():
    parser = argparse.ArgumentParser(description='Run SNAP training and testing experiments.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train', help='Train a SNAP neural solver')
    train_parser.add_argument('--model-id', required=True)
    train_parser.add_argument('--model-method', default='PPOAgent', choices=['PPOAgent', 'GCN_DoubleDQN'])
    train_parser.add_argument('--graph-types', default='BA,ER,GEO,WS')
    train_parser.add_argument('--train-sizes', default='20,50,100')
    train_parser.add_argument('--num-graphs', type=int, default=cfg.TRAIN_GRAPHS_PER_GROUP)
    train_parser.add_argument('--num-episodes', type=int, default=5000)
    train_parser.add_argument('--no-sd', '--no_sd', nargs='?', const=True, type=parse_bool, default=False,
                              help='Use hop-count distances instead of delay-weighted distances. '
                                   'Accepts true/false; default is false.')
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument(
        '--feature-set',
        default=utils.DEFAULT_GRAPH_FEATURE_SET,
        choices=list(utils.GRAPH_FEATURE_SETS),
        help='Static graph feature ablation: f0, f1, f2, or f3 (default: f3)',
    )
    train_parser.add_argument('--prompt-record', action='store_true')
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser('test', help='Evaluate a SNAP neural solver')
    test_parser.add_argument('--model-id', required=True)
    test_parser.add_argument('--model-method', default='PPOAgent', choices=['PPOAgent', 'GCN_DoubleDQN'])
    test_parser.add_argument('--graph-types', default='BA,ER,GEO,WS')
    test_parser.add_argument('--test-sizes', default=','.join(map(str, cfg.TEST_SIZES)))
    test_parser.add_argument('--num-graphs', type=int, default=cfg.TEST_GRAPHS_PER_GROUP)
    test_parser.add_argument('--beta', type=int, default=4)
    test_parser.add_argument('--gamma', type=int, default=4)
    test_parser.add_argument(
        '--sd-score-mode',
        choices=('hop', 'mc', 'pmc'),
        default='hop',
        help='SGBS complete-placement score: hop, unconditional MC, or conditional-prior MC',
    )
    test_parser.add_argument(
        '--search-scenarios',
        type=int,
        default=12,
        help='Independent stratified scenarios per topology for --sd-score-mode mc',
    )
    test_parser.add_argument(
        '--known-sd-level',
        choices=('all', 'low', 'medium', 'high'),
        default='all',
        help='Known mutually exclusive SD level for pmc; all evaluates all three conditional deployments',
    )
    test_parser.add_argument(
        '--search-seed',
        type=int,
        default=None,
        help='Independent search-scenario seed (default: --seed)',
    )
    test_parser.add_argument(
        '--sd-prior',
        choices=('equal_lmh',),
        default='equal_lmh',
        help='Predeclared delay prior; never inferred from test metadata',
    )
    test_parser.add_argument(
        '--delay-mu',
        type=float,
        default=1.0,
        help='Mean edge delay for independent search scenarios',
    )
    test_parser.add_argument(
        '--results-dir',
        default=None,
        help=(
            'Optional output directory override. By default SD SGBS results '
            'are routed to result/hop_sgbs, result/mc_sgbs, or '
            'result/pmc_sgbs according to --sd-score-mode.'
        ),
    )
    test_parser.add_argument(
        '--result-tag',
        default=None,
        help='Safe filename tag used with --results-dir',
    )
    test_parser.add_argument('--fixed-k', type=int, default=None,
                             help='Override the configured sensor count for every requested test size')
    test_parser.add_argument('--no-sd', '--no_sd', nargs='?', const=True, type=parse_bool, default=False,
                             help='Use hop-count distances instead of delay-weighted distances. '
                                  'Accepts true/false; default is false.')
    test_parser.add_argument('--no-baselines', action='store_true')
    test_parser.add_argument(
        '--neural-methods',
        default='one_shot,SGBS',
        help='Comma-separated neural inference methods: one_shot,SGBS',
    )
    test_parser.add_argument('--only-baselines', action='store_true',
                             help='Evaluate only non-neural baseline methods')
    test_parser.add_argument('--seed', type=int, default=42)
    test_parser.add_argument(
        '--feature-set',
        default=utils.DEFAULT_GRAPH_FEATURE_SET,
        choices=list(utils.GRAPH_FEATURE_SETS),
        help='Feature set used to train the checkpoint (default: f3)',
    )
    test_parser.set_defaults(func=test)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
