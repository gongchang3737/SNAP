"""SNAP Fig.5: Source localization performance across topology and graph size.

Usage:
    python plot_snap_fig5.py --results result.zip --outdir fig5_output

Requires: matplotlib, numpy (both pip-installable).
Data: actual JSON/CSV files inside the SNAP result archive. Fixed HCR baseline in all 18 settings.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

GRAPHS = ("BA", "ER", "WS")
SIZES = (20, 50, 100, 200, 300, 500)
BUDGETS = {20: 3, 50: 5, 100: 10, 200: 15, 300: 20, 500: 30}
REFERENCE_BASELINE = "High_Coverage_Rate"  # HCR in baseline CSV
METHODS = ("HCR (fixed baseline)", "One-shot", "Hop-SGBS", "MC-SGBS", "pMC-SGBS")
COLORS = {
    "HCR (fixed baseline)": "#6B7785", "One-shot": "#D5A044",
    "Hop-SGBS": "#3978AC", "MC-SGBS": "#34A68C", "pMC-SGBS": "#8155AC",
}
MARKERS = {"HCR (fixed baseline)": "s", "One-shot": "D", "Hop-SGBS": "o", "MC-SGBS": "^", "pMC-SGBS": "P"}


def find_json(names, folder, graph, n, k):
    # Folder prefixes prevent accidentally mixing legacy SGBS with new audited modes.
    expr = re.compile(rf"(?:^|/){re.escape(folder)}/[^/]*_{graph}_{n}_k{k}_(?:.+)\.json$")
    matched = [x for x in names if expr.search(x)]
    if len(matched) != 1:
        raise ValueError(f"Expected one {folder}/{graph}-{n}-k{k} JSON, found {len(matched)}: {matched[:5]}")
    return matched[0]


def read_rank(zf, names, folder, graph, n, k, key):
    fn = find_json(names, folder, graph, n, k)
    obj = json.loads(zf.read(fn))
    try:
        value = float(obj["PPOAgent"][key]["ave_rank"])
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Missing PPOAgent/{key}/ave_rank in {fn}") from e
    if not np.isfinite(value):
        raise ValueError(f"Invalid rank in {fn}: {value}")
    return value


def load_data(zip_path):
    records = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        summary = [x for x in names if x.endswith('baselines/baselines_summary.csv')]
        if len(summary) != 1:
            raise ValueError("Cannot uniquely locate baselines/baselines_summary.csv")
        rows = list(csv.DictReader(io.StringIO(zf.read(summary[0]).decode('utf-8-sig'))))
        baselines = {}
        for row in rows:
            if row['graph_type'] not in GRAPHS or int(row['n_nodes']) not in SIZES:
                continue
            # Important: only compare against baseline results WITH stochastic delays.
            if row['has_delay_distribution'].strip().lower() != 'true':
                continue
            key = (row['graph_type'], int(row['n_nodes']), int(row['sensor_budget']))
            baselines.setdefault(key, []).append((row['method'], float(row['ave_rank'])))

        for graph in GRAPHS:
            for n in SIZES:
                k = BUDGETS[n]
                key = (graph, n, k)
                if key not in baselines:
                    raise ValueError(f"No SD baseline for {key}")
                baseline_dict = dict(baselines[key])
                if len(baseline_dict) != len(baselines[key]):
                    raise ValueError(f"Duplicate baseline entries for {key}")
                if REFERENCE_BASELINE not in baseline_dict:
                    raise ValueError(f"Missing fixed HCR baseline for {key}")
                baseline_method = REFERENCE_BASELINE
                baseline_value = baseline_dict[REFERENCE_BASELINE]
                if not np.isfinite(baseline_value) or baseline_value <= 0:
                    raise ValueError(f"Invalid fixed HCR rank for {key}: {baseline_value}")
                ranks = {
                    "HCR (fixed baseline)": baseline_value,
                    "One-shot": read_rank(zf, names, 'ppo_mix_v3_sd', graph, n, k, 'one_shot'),
                    "Hop-SGBS": read_rank(zf, names, 'hop_sgbs', graph, n, k, 'SGBS'),
                    "MC-SGBS": read_rank(zf, names, 'mc_sgbs', graph, n, k, 'SGBS'),
                    "pMC-SGBS": read_rank(zf, names, 'pmc_sgbs', graph, n, k, 'SGBS'),
                }
                for method in METHODS:
                    v = ranks[method]
                    records.append({
                        'graph_type':graph, 'n_nodes':n, 'sensor_budget':k,
                        'method':method, 'average_rank':v,
                        'reference_baseline_method':'HCR',
                        'reference_baseline_rank':baseline_value,
                        'improvement_pct': 100 * (baseline_value-v)/baseline_value,
                        'dataset':'with_delay_distribution',
                    })
    if len(records) != len(GRAPHS)*len(SIZES)*len(METHODS):
        raise ValueError(f"Expected 90 plot records, got {len(records)}")
    return records


def validate_reference(records):
    baselines = [r for r in records if r['method'] == 'HCR (fixed baseline)']
    assert len(baselines) == len(GRAPHS)*len(SIZES)
    assert all(r['reference_baseline_method'] == "HCR" for r in records)
    assert all(abs(r['improvement_pct']) < 1e-10 for r in baselines)


def draw_figure(records, output):
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':9.5,
        'axes.labelsize':10.5, 'axes.titlesize':12, 'xtick.labelsize':9,
        'ytick.labelsize':9, 'pdf.fonttype':42, 'ps.fonttype':42,
        'svg.fonttype':'none', 'axes.spines.top':False,
        'axes.spines.right':False, 'savefig.facecolor':'white'
    })
    # Unlike a three-panel figure, each column also shows relative improvement.
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.2), sharex='col', constrained_layout=False)
    plt.subplots_adjust(left=.085, right=.985, bottom=.16, top=.95, wspace=.26, hspace=.22)
    xs = np.arange(len(SIZES))
    for col, graph in enumerate(GRAPHS):
        ax_top, ax_bottom = axes[0,col], axes[1,col]
        for method in METHODS:
            subset = [r for r in records if r['graph_type']==graph and r['method']==method]
            subset.sort(key=lambda r:r['n_nodes'])
            assert [r['n_nodes'] for r in subset] == list(SIZES)
            ranks = [r['average_rank'] for r in subset]
            gains = [r['improvement_pct'] for r in subset]
            plot_kwargs = dict(color=COLORS[method], marker=MARKERS[method], markersize=5,
                               linewidth=1.9 if method!='HCR (fixed baseline)' else 1.6,
                               label=method, zorder=3)
            ax_top.plot(xs, ranks, **plot_kwargs)
            ax_bottom.plot(xs, gains, **plot_kwargs)
        for ax in (ax_top, ax_bottom):
            ax.axvspan(2.5, 5.35, facecolor='#F3F1FA', alpha=.55, zorder=0)
            ax.axvline(2.5, color='#897BA9', linestyle=(0,(4,3)), linewidth=1.05)
            ax.grid(axis='y', color='#dce4ea', linewidth=.7, alpha=.8)
            ax.set_xlim(-.2, 5.2)
            ax.tick_params(axis='both', length=0, pad=6)
        ax_top.set_title(f'({chr(97+col)}) {graph} network', fontsize=12, weight='semibold', pad=10)
        ax_top.set_ylim(bottom=0)
        ax_bottom.axhline(0, color='#6B7785', linewidth=1.0, alpha=.9)
        ax_bottom.set_xticks(xs, [str(n) for n in SIZES])
        ax_bottom.set_xlabel('Number of nodes (N)')
        if col == 0:
            ax_top.set_ylabel('Average source rank (lower is better)')
            ax_bottom.set_ylabel('Improvement over HCR (%)')
    handles=[Line2D([], [], color=COLORS[m], marker=MARKERS[m], linewidth=2, markersize=5, label=m) for m in METHODS]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(.5,.055),
               frameon=False, fontsize=10, handlelength=2.4, columnspacing=1.8)
    for ext in ('png','pdf','svg'):
        fig.savefig(output/f'SNAP_Fig5_topology_scale.{ext}', dpi=400 if ext=='png' else None)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results', default='result.zip', help='ZIP archive of the SNAP result directory')
    ap.add_argument('--outdir', default='fig5_output')
    args=ap.parse_args()
    output=Path(args.outdir);output.mkdir(parents=True,exist_ok=True)
    records=load_data(args.results)
    validate_reference(records)
    with (output/'SNAP_Fig5_plot_data.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    draw_figure(records, output)
    print(f'Generated Fig.5 in {output.resolve()} using {len(records)} measured results.')
    for graph in GRAPHS:
        row=next(r for r in records if r['graph_type']==graph and r['n_nodes']==500 and r['method']=='pMC-SGBS')
        print(f'{graph}-500: pMC AR={row["average_rank"]:.4f}, improvement vs {row["reference_baseline_method"]}={row["improvement_pct"]:.2f}%')

if __name__=='__main__':
    main()
