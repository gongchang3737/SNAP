"""SNAP Fig. 6 — delay variability effects from audited result.zip.

Run: python plot_snap_fig6.py --results result.zip --outdir fig6_output
Uses original low/medium/high summaries, not invented data. Averages across the
18 BA/ER/WS x graph-size settings are unweighted macro averages.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GRAPHS = ('BA', 'ER', 'WS')
SIZES = (20, 50, 100, 200, 300, 500)
BUDGETS = {20: 3, 50: 5, 100: 10, 200: 15, 300: 20, 500: 30}
MODES = {'Hop-SGBS':'hop_sgbs', 'MC-SGBS':'mc_sgbs', 'pMC-SGBS':'pmc_sgbs'}
REGIMES = ('low', 'medium', 'high')
COLORS = {'Hop-SGBS':'#3978AC', 'MC-SGBS':'#34A68C', 'pMC-SGBS':'#8155AC'}


def find_result(names, folder, graph, n, k):
    pattern = re.compile(rf'^{folder}/[^/]*_{graph}_{n}_k{k}_.+\.json$')
    matches = [name for name in names if pattern.fullmatch(name)]
    if len(matches) != 1:
        raise ValueError(f'Expected one file for {folder}/{graph}-{n}-k{k}; found {len(matches)}: {matches}')
    return matches[0]


def read_records(archive):
    data = {}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for graph in GRAPHS:
            for n in SIZES:
                k = BUDGETS[n]
                for method, folder in MODES.items():
                    filename = find_result(names, folder, graph, n, k)
                    obj = json.loads(z.read(filename))
                    item = obj['PPOAgent']['SGBS']
                    groups = item['sd_groups']
                    for regime in REGIMES:
                        rank = float(groups[regime]['ave_rank'])
                        if not np.isfinite(rank) or rank <= 0:
                            raise ValueError(f'Invalid rank: {filename}: {regime}')
                        data[(graph, n, method, regime)] = rank
                    # Equal-size strata: mean of three regime means = pooled average.
                    avg = np.mean([data[(graph,n,method,r)] for r in REGIMES])
                    if not np.isclose(avg, float(item['ave_rank']), rtol=1e-6, atol=1e-6):
                        raise ValueError(f'SD groups inconsistent with overall rank: {filename}')
    return data


def make_rows(data):
    rows = []
    for graph in GRAPHS:
        for n in SIZES:
            for regime in REGIMES:
                href = data[(graph,n,'Hop-SGBS',regime)]
                for method in MODES:
                    ar = data[(graph,n,method,regime)]
                    rows.append(dict(record_type='setting', graph_type=graph, n_nodes=n,
                                     sensor_budget=BUDGETS[n], sd_regime=regime,
                                     method=method, average_rank=ar, reference_method='Hop-SGBS',
                                     reference_rank=href, improvement_pct=100*(href-ar)/href,
                                     aggregation='per graph-type and node size'))
    for regime in REGIMES:
        hop_macro = np.mean([data[(g,n,'Hop-SGBS',regime)] for g in GRAPHS for n in SIZES])
        for method in MODES:
            rank_macro = np.mean([data[(g,n,method,regime)] for g in GRAPHS for n in SIZES])
            rows.append(dict(record_type='macro_18', graph_type='ALL', n_nodes='ALL', sensor_budget='varies',
                             sd_regime=regime, method=method, average_rank=rank_macro,
                             reference_method='Hop-SGBS', reference_rank=hop_macro,
                             improvement_pct=100*(hop_macro-rank_macro)/hop_macro,
                             aggregation='ratio of 18-setting macro-average ranks'))
    return rows


def plot(rows, outdir, dpi):
    lookup = {(r['sd_regime'],r['method']):r for r in rows if r['record_type']=='macro_18'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(9.2,7.5),gridspec_kw={'height_ratios':[1.05,1]},layout='constrained')
    x=np.arange(3)
    offsets = {'Hop-SGBS':-.23,'MC-SGBS':0,'pMC-SGBS':.23}
    width=.21
    for method in MODES:
        vals=[float(lookup[(r,method)]['average_rank']) for r in REGIMES]
        bars=ax1.bar(x+offsets[method],vals,width,color=COLORS[method],alpha=.88,label=method, zorder=3)
        for bar,v in zip(bars,vals):
            ax1.annotate(f'{v:.2f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,4),
                         textcoords='offset points',ha='center',va='bottom',fontsize=8)
    ax1.set_yscale('log')
    ax1.set_ylim(.43,max(float(lookup[('high',m)]['average_rank']) for m in MODES)*1.55)
    ax1.set_xticks(x,('Low  (0, 0.1]','Medium  (0.1, 0.3]','High  (0.3, 0.5]'))
    ax1.set_ylabel('Average source rank (log scale; lower is better)')
    ax1.set_title('(a) Source localization under propagation-delay variability',loc='left',weight='bold',pad=13)
    ax1.grid(axis='y',linestyle=':',alpha=.28,zorder=0)
    ax1.legend(frameon=False,ncol=3,loc='upper left')

    for method, offset in [('MC-SGBS',-.13),('pMC-SGBS',.13)]:
        vals=[float(lookup[(r,method)]['improvement_pct']) for r in REGIMES]
        bars=ax2.bar(x+offset,vals,.25,color=COLORS[method],alpha=.9,label=method,zorder=3)
        for bar,v in zip(bars,vals):
            ax2.annotate(f'{v:+.1f}%',(bar.get_x()+bar.get_width()/2,v),
                         xytext=(0,4 if v>=0 else -4),textcoords='offset points',
                         va='bottom' if v>=0 else 'top',ha='center',fontsize=9)
    ax2.axhline(0,color='#687889',linewidth=1)
    ax2.set_xticks(x,('Low','Medium','High'))
    ax2.set_ylabel('Improvement over Hop-SGBS (%)')
    ax2.set_xlabel('Propagation-delay variability (pre-truncation SD parameter)')
    ax2.set_title('(b) Benefit of uncertainty-aware completion scoring',loc='left',weight='bold',pad=13)
    ax2.grid(axis='y',linestyle=':',alpha=.28,zorder=0)
    ax2.legend(frameon=False,ncol=2,loc='upper left')
    allgains=[float(lookup[(r,m)]['improvement_pct']) for r in REGIMES for m in ('MC-SGBS','pMC-SGBS')]
    ax2.set_ylim(min(-4,min(allgains)-3),max(allgains)+7)
    for ext in ('png','svg','pdf'):
        fig.savefig(outdir/f'SNAP_Fig6_delay_uncertainty.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('result.zip'))
    parser.add_argument('--outdir',type=Path,default=Path('fig6_output'))
    parser.add_argument('--dpi',type=int,default=400)
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    data=read_records(args.results)
    rows=make_rows(data)
    with (args.outdir/'SNAP_Fig6_plot_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    plot(rows,args.outdir,args.dpi)
    print('Saved Fig. 6 (PNG/SVG/PDF) and 171 CSV rows:', args.outdir)
    for row in rows:
        if row['record_type']=='macro_18':
            print(row['sd_regime'],row['method'],f"rank={row['average_rank']:.4f}",f"gain={row['improvement_pct']:+.2f}%")

if __name__=='__main__':
    main()
