"""SNAP Fig. 6 (final style): grouped bar chart with log y-axis on top,
and six heatmaps below, using a muted brown-to-teal diverging colormap.

Usage:
    python plot_snap_fig6_barlog_refined.py --results result.zip --outdir fig6_barlog_refined
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

from plot_snap_fig6 import GRAPHS, SIZES, REGIMES, MODES, COLORS, read_records, make_rows

RENAME = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
INTERVALS = {'low': '(0, 0.1]', 'medium': '(0.1, 0.3]', 'high': '(0.3, 0.5]'}


def macro_values(data):
    return {
        method: np.array([
            np.mean([data[(g, n, method, r)] for g in GRAPHS for n in SIZES])
            for r in REGIMES
        ])
        for method in MODES
    }


def gain_matrices(data):
    gains = {}
    for method in ('MC-SGBS', 'pMC-SGBS'):
        for graph in GRAPHS:
            mat = np.zeros((3, 6))
            for i, regime in enumerate(REGIMES):
                for j, n in enumerate(SIZES):
                    hop = data[(graph, n, 'Hop-SGBS', regime)]
                    gains_val = 100.0 * (hop - data[(graph, n, method, regime)]) / hop
                    mat[i, j] = gains_val
            gains[(method, graph)] = mat
    return gains


def build_reference_cmap():
    """Muted diverging map matching the reference image color depth."""
    colors = [
        '#874e0a',  # strong negative dark brown
        '#cea053',  # negative mid tan
        '#f6e8c3',  # near-zero warm light
        '#ddeeed',  # near-zero cool light
        '#79c8bc',  # positive mid teal
        '#13776f',  # positive dark teal
        '#003f34',  # strong positive dark green-teal
    ]
    return LinearSegmentedColormap.from_list('snap_ref_div', colors, N=256)


def draw_top_bar(ax, macro):
    x = np.arange(len(REGIMES))
    width = 0.22
    order = ['Hop-SGBS', 'MC-SGBS', 'pMC-SGBS']
    offsets = [-width, 0.0, width]
    for method, dx in zip(order, offsets):
        y = macro[method]
        bars = ax.bar(x + dx, y, width=width * 0.92, color=COLORS[method],
                      edgecolor='white', linewidth=0.9, label=method, zorder=3)
        for bar, yi in zip(bars, y):
            ax.annotate(f'{yi:.3f}',
                        (bar.get_x() + bar.get_width() / 2.0, yi),
                        textcoords='offset points', xytext=(0, 4),
                        ha='center', va='bottom', fontsize=7.7, color=COLORS[method])
    ax.set_yscale('log')
    ax.set_ylim(0.46, 39)
    ax.set_xticks(x, [f'{RENAME[r]}\n{INTERVALS[r]}' for r in REGIMES])
    ax.set_ylabel('Average source rank\n(log scale; lower is better)')
    ax.set_title('(a) Mean localization performance across all 18 synthetic settings',
                 loc='left', fontweight='bold', pad=8)
    ax.grid(axis='y', linestyle=':', alpha=0.24, zorder=0)
    ax.legend(loc='upper left', ncol=3, frameon=False, bbox_to_anchor=(0.0, 1.02),
              handlelength=1.5, columnspacing=1.2)


def draw_heatmap(ax, mat, title, cmap, norm):
    im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_yticks(range(len(REGIMES)), [RENAME[r] for r in REGIMES])
    ax.set_xlabel('Graph size N')
    ax.set_ylabel('Delay variability')
    ax.set_title(title, loc='left', fontweight='bold', pad=4)
    ax.set_xticks(np.arange(-.5, len(SIZES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(REGIMES), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = 'white' if abs(val) > 12 else '#4d5962'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center', fontsize=7.6, color=color)
    return im


def write_csv(rows, out_csv: Path):
    fields = ['record_type', 'graph_type', 'n_nodes', 'sensor_budget', 'sd_regime',
              'method', 'average_rank', 'reference_method', 'reference_rank',
              'improvement_pct', 'aggregation']
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_fig6(data, outdir: Path, dpi: int):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9.4,
        'axes.titlesize': 10.1, 'axes.labelsize': 9.1,
        'xtick.labelsize': 8.3, 'ytick.labelsize': 8.3,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })

    fig = plt.figure(figsize=(13.3, 9.6), facecolor='white')
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 1.0, 1.0],
                          left=0.075, right=0.88, top=0.97, bottom=0.05,
                          hspace=0.35, wspace=0.32)

    ax_top = fig.add_subplot(gs[0, :])
    draw_top_bar(ax_top, macro_values(data))

    gains = gain_matrices(data)
    cmap = build_reference_cmap()
    all_gains = np.concatenate([m.ravel() for m in gains.values()])
    vmax = max(abs(float(np.nanmin(all_gains))), abs(float(np.nanmax(all_gains))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    titles = {
        ('MC-SGBS', 'BA'): '(b) BA | MC-SGBS vs Hop',
        ('MC-SGBS', 'ER'): '(c) ER | MC-SGBS vs Hop',
        ('MC-SGBS', 'WS'): '(d) WS | MC-SGBS vs Hop',
        ('pMC-SGBS', 'BA'): '(e) BA | pMC-SGBS vs Hop',
        ('pMC-SGBS', 'ER'): '(f) ER | pMC-SGBS vs Hop',
        ('pMC-SGBS', 'WS'): '(g) WS | pMC-SGBS vs Hop',
    }

    im = None
    for row_idx, method in enumerate(['MC-SGBS', 'pMC-SGBS'], start=1):
        for col_idx, graph in enumerate(GRAPHS):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = draw_heatmap(ax, gains[(method, graph)], titles[(method, graph)], cmap, norm)

    # cax = fig.add_axes([0.90, 0.205, 0.018, 0.515])
    # cax = fig.add_axes([0.90, 0.14, 0.018, 0.44])
    # cax = fig.add_axes([0.90, 0.155, 0.018, 0.455])
    cax = fig.add_axes([0.90, 0.05, 0.018, 0.53])
    # [left, bottom, width, height]
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Relative improvement over Hop-SGBS (%)\npositive = lower source rank', fontsize=8.8)
    cb.ax.tick_params(labelsize=8)

    stem = 'SNAP_Fig6_delay_uncertainty_barlog_refined'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(outdir / f'{stem}.{ext}', dpi=dpi if ext == 'png' else None,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, default=Path('fig6_barlog_refined'))
    ap.add_argument('--dpi', type=int, default=400)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    data = read_records(args.results)
    rows = make_rows(data)
    write_csv(rows, args.outdir / 'SNAP_Fig6_plot_data.csv')
    plot_fig6(data, args.outdir, dpi=args.dpi)


if __name__ == '__main__':
    main()
