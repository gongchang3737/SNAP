"""Compare SGBS beam settings on the existing no-SD IEEE_118Bus test graph."""

import argparse
import csv
import json
from pathlib import Path

import experiment_config as config
from OP_Tester import OP_Tester
from run_experiment import set_seed


GRAPH_TYPE = 'IEEE_118Bus'
N_NODES = 118
MODEL_ID = 'ppo_mix_v3_no_sd'
BASELINE_PATH = Path('result/baselines/baselines_IEEE_118Bus_118_k12.json')
OUTPUT_DIR = Path('result/ppo_mix_v3_no_sd/sgbs_tuning_IEEE_118Bus')


def parse_pairs(value):
    pairs = []
    for item in value.split(','):
        beta, gamma = map(int, item.split(':'))
        if beta < 1 or gamma < 1:
            raise ValueError('beta and gamma must both be positive integers')
        pairs.append((beta, gamma))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pairs', default='4:8,8:4,8:8,12:8,8:12,12:12',
        help='Comma-separated beta:gamma pairs',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with BASELINE_PATH.open(encoding='utf-8') as file:
        baselines = json.load(file)['PPOAgent']
    best_name, best_result = min(
        baselines.items(), key=lambda item: item[1]['ave_rank']
    )
    best_rank = float(best_result['ave_rank'])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / 'summary.csv'
    rows = []
    if summary_path.exists():
        with summary_path.open(newline='', encoding='utf-8-sig') as file:
            for row in csv.DictReader(file):
                rows.append((
                    int(row['beta']), int(row['gamma']), float(row['ave_rank']),
                    float(row['ave_time_seconds']), row['best_baseline'],
                    float(row['baseline_ave_rank']), float(row['improvement_percent']),
                ))

    for beta, gamma in parse_pairs(args.pairs):
        set_seed(args.seed)
        method_params = config.build_method_params(
            MODEL_ID, 'PPOAgent', include_baselines=False,
            beta=beta, gamma=gamma, fixed_k=12,
        )
        method_params['test_methods'] = ['SGBS']
        tester_params = config.build_tester_params(
            MODEL_ID, GRAPH_TYPE, N_NODES, num_graphs=1,
            no_sd=True, fixed_k=12,
        )
        tester_params['results_save_path'] = str(
            OUTPUT_DIR / f'sgbs_beta{beta}_gamma{gamma}.json'
        )
        OP_Tester(tester_params, method_params, config.build_env_params()).run()

        with Path(tester_params['results_save_path']).open(encoding='utf-8') as file:
            result = json.load(file)['PPOAgent']['SGBS']
        rank = float(result['ave_rank'])
        elapsed = float(result['ave_time'])
        rows = [row for row in rows if row[:2] != (beta, gamma)]
        rows.append((beta, gamma, rank, elapsed, best_name, best_rank,
                     100.0 * (best_rank - rank) / best_rank))
        print(
            f'beta={beta}, gamma={gamma}: rank={rank:.6f}, '
            f'time={elapsed:.2f}s, versus {best_name}={best_rank:.6f}',
            flush=True,
        )
        with summary_path.open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow((
                'beta', 'gamma', 'ave_rank', 'ave_time_seconds',
                'best_baseline', 'baseline_ave_rank', 'improvement_percent',
            ))
            writer.writerows(rows)


if __name__ == '__main__':
    main()
