#!/gpfs0/apps/x86/anaconda3/bin/python3
#SBATCH --job-name=plot_anchor_stats
#SBATCH --output=plot_anchor_stats_%j.out
#SBATCH --error=plot_anchor_stats_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --partition=int

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_anchor_bed(bed_file):
    '''
    Reads an anchor BED file with columns: CHR START END AG CHIPSEQ.
    Returns a dict keyed by (chr, start, end) → {'ag': float, 'chipseq': float}.
    '''
    anchors = {}
    with open(bed_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 5:
                continue
            key = (parts[0], int(parts[1]), int(parts[2]))
            try:
                ag = float(parts[3])
            except (ValueError, IndexError):
                ag = float('nan')
            try:
                chipseq = float(parts[4])
            except (ValueError, IndexError):
                chipseq = float('nan')
            anchors[key] = {'ag': ag, 'chipseq': chipseq}
    return anchors


def _regression(x, y):
    '''Returns (slope, intercept, pearson_r) for finite x/y pairs.'''
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None, None, None
    coeffs = np.polyfit(x, y, 1)
    r      = np.corrcoef(x, y)[0, 1]
    return coeffs[0], coeffs[1], r


def _plot_series(ax, x_vals, y_vals, color, label_base):
    '''Scatter + regression line for one series.'''
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    xv, yv = x_vals[mask], y_vals[mask]
    n = mask.sum()

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Plot anchor stats TSV produced by anchor_loop_hic.py.\n\n'
            'Draws two scatter series on one plot:\n'
            '  • ChIP-seq  vs  AlphaGenome  (blue)\n'
            '  • Hi-C log2FC  vs  AlphaGenome  (green)\n\n'
            'AG and ChIP-seq values are read directly from --anchor_bed.\n'
            'Each series includes a dashed regression line and Pearson r.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config',     default=None, metavar='CONFIG.JSON',
                        help='JSON config file; CLI args override values set here.')
    parser.add_argument('--stats',      default=None,
                        help='Anchor stats TSV from anchor_loop_hic.py')
    parser.add_argument('--anchor_bed', default=None,
                        help='Anchor BED file: CHR START END AG CHIPSEQ')
    parser.add_argument('--output',     default=None,
                        help='Output path (default: <stats_stem>_plot.pdf)')
    parser.add_argument('--title',      default=None,
                        help='Plot title (default: auto-generated from filename)')

    # two-pass parse: load config first, CLI overrides second
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as fh:
            parser.set_defaults(**json.load(fh))
    args = parser.parse_args()

    if args.stats is None:
        parser.error('--stats is required.')
    if args.output is None:
        args.output = os.path.splitext(args.stats)[0] + '_plot.pdf'

    # ── load TSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.stats, sep='\t')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ── pull ag / chipseq from BED if provided ────────────────────────────────
    if args.anchor_bed:
        bed = parse_anchor_bed(args.anchor_bed)
        df['ag']      = df.apply(
            lambda r: bed.get((r['chr'], r['start'], r['end']), {}).get('ag',      float('nan')), axis=1)
        df['chipseq'] = df.apply(
            lambda r: bed.get((r['chr'], r['start'], r['end']), {}).get('chipseq', float('nan')), axis=1)
    else:
        # fall back to pre-computed log2 columns already in the TSV
        df['ag']      = df['ag_log2']
        df['chipseq'] = df['chipseq_log2']

    ag   = df['ag'].to_numpy(dtype=float)
    chip = df['chipseq'].to_numpy(dtype=float)
    hic  = df['anc_lfc'].to_numpy(dtype=float)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    _plot_series(ax, chip, ag, color='steelblue',     label_base='ChIP-seq')
    _plot_series(ax, hic,  ag, color='mediumseagreen', label_base='Hi-C log2FC')

    ax.axhline(0, color='lightgray', linewidth=0.8, zorder=1)
    ax.axvline(0, color='lightgray', linewidth=0.8, zorder=1)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    ax.set_xlabel('signal / fold-change', fontsize=11)
    ax.set_ylabel('AlphaGenome score', fontsize=11)
    ax.set_title(args.title or f'Anchor stats — {os.path.basename(args.stats)}',
                 fontsize=12, pad=10)
    ax.legend(framealpha=0.85, fontsize=9)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Plot saved → {args.output}')


if __name__ == '__main__':
    main()
