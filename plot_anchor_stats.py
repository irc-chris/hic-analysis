#!/gpfs0/apps/x86/anaconda3/bin/python3
#SBATCH --job-name=plot_anchor_stats
#SBATCH --output=plot_anchor_stats_%j.out
#SBATCH --error=plot_anchor_stats_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --partition=int

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# ── helpers ───────────────────────────────────────────────────────────────────

def _regression(x, y):
    '''Returns (slope, intercept, pearson_r) for finite x/y pairs.'''
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None, None, None
    coeffs   = np.polyfit(x, y, 1)
    r        = np.corrcoef(x, y)[0, 1]
    return coeffs[0], coeffs[1], r


def _plot_series(ax, x_vals, y_vals, color, label_base):
    '''Scatter + regression line for one series. Returns number of valid points.'''
    mask  = np.isfinite(x_vals) & np.isfinite(y_vals)
    xv, yv = x_vals[mask], y_vals[mask]
    n     = mask.sum()

    slope, intercept, r = _regression(x_vals, y_vals)

    scatter_label = f'{label_base}  (n={n})'
    if r is not None:
        scatter_label += f'  r={r:.2f}'

    ax.scatter(xv, yv, color=color, alpha=0.65, s=30, linewidths=0,
               label=scatter_label, zorder=3)

    if slope is not None and n >= 2:
        x_line = np.array([xv.min(), xv.max()])
        ax.plot(x_line, slope * x_line + intercept,
                color=color, linewidth=1.2, linestyle='--', alpha=0.8, zorder=2)

    return n


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Plot anchor stats TSV produced by anchor_loop_hic.py.\n\n'
            'Draws two scatter series on one plot:\n'
            '  • ChIP-seq log2  vs  AlphaGenome log2  (blue)\n'
            '  • Hi-C log2FC    vs  AlphaGenome log2  (green)\n\n'
            'Each series includes a dashed regression line and Pearson r.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('stats',
                        help='Anchor stats TSV from anchor_loop_hic.py')
    parser.add_argument('--output', default=None,
                        help='Output path (default: <stats_stem>_plot.pdf)')
    parser.add_argument('--title', default=None,
                        help='Plot title (default: auto-generated from filename)')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.stats)[0] + '_plot.pdf'

    df = pd.read_csv(args.stats, sep='\t')

    # replace ±inf with nan so isfinite filtering works cleanly
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    ag       = df['ag_log2'].to_numpy(dtype=float)
    chip     = df['chipseq_log2'].to_numpy(dtype=float)
    hic      = df['anc_lfc'].to_numpy(dtype=float)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    _plot_series(ax, chip, ag, color='steelblue',    label_base='ChIP-seq log2')
    _plot_series(ax, hic,  ag, color='mediumseagreen', label_base='Hi-C log2FC')

    # reference lines
    ax.axhline(0, color='lightgray', linewidth=0.8, linestyle='-', zorder=1)
    ax.axvline(0, color='lightgray', linewidth=0.8, linestyle='-', zorder=1)

    ax.set_xlabel('log2 signal / fold-change', fontsize=11)
    ax.set_ylabel('AlphaGenome prediction (log2)', fontsize=11)

    title = args.title or (
        f'Anchor stats — {os.path.basename(args.stats)}'
    )
    ax.set_title(title, fontsize=12, pad=10)

    ax.legend(framealpha=0.85, fontsize=9)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Plot saved → {args.output}')


if __name__ == '__main__':
    main()
