#!/gpfs0/apps/x86/anaconda3/bin/python3
#SBATCH --job-name=anchor_hic_plots
#SBATCH --output=anchor_hic_plots_%j.out
#SBATCH --error=anchor_hic_plots_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --partition=weka2

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

import hicstraw

# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

def ts():
    '''Returns a [HH:MM:SS] timestamp string for print statements.'''
    return datetime.now().strftime('[%H:%M:%S]')

# ── constants ─────────────────────────────────────────────────────────────────
REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])

# ── reused verbatim from hic_plots.py ─────────────────────────────────────────

def preprocess(matrix, scale=0.1, epsilon=1e-9):
    '''
    Preprocesses a Hi-C matrix for better viewing using tanh/MAD scaling.
    Verbatim from hic_plots.py.
    '''
    if np.abs(np.sum(matrix)) < epsilon:
        return matrix
    flattenedData = matrix.flatten()
    flattenedData = flattenedData[flattenedData > 0]
    medianVal = np.median(flattenedData)
    madVal = np.median(np.abs(flattenedData - medianVal))
    if madVal == 0:
        madVal = epsilon
    m = np.tanh(scale * (matrix - medianVal) / madVal)
    return m

# ── parsing ───────────────────────────────────────────────────────────────────

def parse_anchor_bed(bed_file):
    '''
    Parses a BED file of belt anchors (CHR START END).
    Returns a list of (chr, start, end) tuples in file order.
    Lines starting with '#' are skipped.
    '''
    anchors = []
    with open(bed_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                parts = line.split()
            anchors.append((parts[0], int(parts[1]), int(parts[2])))
    return anchors


def parse_loop_bedpe(bedpe_file):
    '''
    Parses a BEDPE loop file (CHR START0 END0 CHR START1 END1 ...).
    Only the first 6 columns are used; any extra columns are ignored.
    Returns a list of dicts with keys: chr0, start0, end0, chr1, start1, end1.
    '''
    loops = []
    with open(bedpe_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            # skip header lines (first field must look like a chromosome)
            if not (parts[0][-1].isdigit() or parts[0][-1] in 'XYM' or parts[0].startswith('chr')):
                continue
            if len(parts) < 6:
                logger.warning(f"Skipping line with fewer than 6 fields: {line}")
                continue
            loops.append({
                'chr0':   parts[0],
                'start0': int(parts[1]),
                'end0':   int(parts[2]),
                'chr1':   parts[3],
                'start1': int(parts[4]),
                'end1':   int(parts[5]),
            })
    return loops

# ── grouping ──────────────────────────────────────────────────────────────────

def group_loops_by_anchor(loops, anchors):
    '''
    Groups loops by which belt anchor they share.

    For each BED anchor (chr, start, end), finds loops where anchor0 OR anchor1
    overlaps the BED anchor via standard genomic interval overlap:
        bedpe_start < bed_end  AND  bedpe_end > bed_start  (same chromosome)

    Returns:
        {(chr, start, end): [{'loop': loop_dict, 'matching_anchor': 0 or 1}, ...]}

    All BED anchors appear as keys (empty list if no loops match).
    If both BEDPE anchors overlap the same BED anchor, the loop is added once
    (via anchor0) and a warning is logged.
    '''
    anchor_groups = {anchor: [] for anchor in anchors}

    for loop in loops:
        for anchor in anchors:
            bed_chr, bed_start, bed_end = anchor

            a0_overlaps = (
                loop['chr0'] == bed_chr and
                loop['start0'] < bed_end and loop['end0'] > bed_start
            )
            a1_overlaps = (
                loop['chr1'] == bed_chr and
                loop['start1'] < bed_end and loop['end1'] > bed_start
            )

            if a0_overlaps and a1_overlaps:
                logger.warning(
                    f"Both anchors of loop "
                    f"{loop['chr0']}:{loop['start0']}-{loop['end0']} × "
                    f"{loop['chr1']}:{loop['start1']}-{loop['end1']} "
                    f"overlap BED anchor {bed_chr}:{bed_start}-{bed_end}. "
                    f"Adding once (via anchor0)."
                )
                anchor_groups[anchor].append({'loop': loop, 'matching_anchor': 0})
            elif a0_overlaps:
                anchor_groups[anchor].append({'loop': loop, 'matching_anchor': 0})
            elif a1_overlaps:
                anchor_groups[anchor].append({'loop': loop, 'matching_anchor': 1})

    return anchor_groups

# ── HiC matrix fetch ──────────────────────────────────────────────────────────

def get_hic_matrices(hic_map_configs, chr_num, start0, end0, start1, end1, resolution, cache):
    '''
    Fetches HiC contact matrices for a given region from all 3 HiC maps.
    Returns a list of 3 numpy arrays: [unphased, ref, alt].

    cache: dict keyed on (hic_path, chr_num, norm) -> (HiCFile, MatrixZoomData)
    Stores both the HiCFile and MatrixZoomData to keep the file handle alive.

    On any error, returns a zero-filled matrix and logs the error.
    '''
    n_bins_x = max(1, (end0 - start0) // resolution)
    n_bins_y = max(1, (end1 - start1) // resolution)

    matrices = []
    for config in hic_map_configs:
        hic_path = config.get('path') or config.get('url')
        hic_norm = config['norm']
        cache_key = (hic_path, chr_num, hic_norm)

        if cache_key not in cache:
            try:
                hic_file = hicstraw.HiCFile(hic_path)
                matrix_obj = hic_file.getMatrixZoomData(
                    chr_num, chr_num, "observed", hic_norm, "BP", resolution
                )
                cache[cache_key] = (hic_file, matrix_obj)  # keep hic_file alive
            except Exception as e:
                logger.error(f"Failed to open HiC '{hic_path}' for chr '{chr_num}': {e}")
                cache[cache_key] = None

        cached = cache[cache_key]
        if cached is None:
            matrices.append(np.zeros((n_bins_x, n_bins_y)))
        else:
            _, matrix_obj = cached
            try:
                numpy_matrix = matrix_obj.getRecordsAsMatrix(start0, end0, start1, end1)
                matrices.append(numpy_matrix)
            except Exception as e:
                logger.error(
                    f"Failed to fetch matrix {chr_num}:{start0}-{end0} × {start1}-{end1}: {e}"
                )
                matrices.append(np.zeros((n_bins_x, n_bins_y)))

    return matrices

# ── plotting ──────────────────────────────────────────────────────────────────

def plot_mini_hic(ax, matrix, vmin, vmax):
    '''
    Renders a single mini HiC contact matrix into a matplotlib Axes.
    All tick marks and labels are hidden.
    '''
    ax.matshow(matrix, cmap=REDMAP, vmin=vmin, vmax=vmax, aspect='auto')
    ax.tick_params(
        axis='both', which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False
    )


def emit_no_loops_page(pdf, anchor):
    '''Emits a placeholder page noting no loops were found for this anchor.'''
    bed_chr, bed_start, bed_end = anchor
    fig = plt.figure(figsize=(8, 3))
    fig.text(
        0.5, 0.5,
        f"No loops found for anchor {bed_chr}:{bed_start:,}-{bed_end:,}",
        ha='center', va='center', fontsize=12
    )
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def plot_anchor_page(pdf, anchor, loop_entries, hic_map_configs, width,
                     resolution, cache, page_num, total_pages):
    '''
    Creates one PDF page for a given belt anchor.

    Layout: n_rows × 3 grid of mini HiC maps, one row per loop.
    Columns (left to right): Unphased | Ref (hap1) | Alt (hap2)
    Row labels show the loop coordinates on the left margin.

    Color scale:
      - Unphased (col 0): tanh/MAD via preprocess(), own vmin/vmax
      - Ref (col 1) and Alt (col 2): raw counts, share per-row vmax
    '''
    plt.rcParams['font.family'] = 'monospace'

    bed_chr, bed_start, bed_end = anchor
    n_rows = len(loop_entries)

    cell_h    = 2.5   # inches per loop row
    fig_width = 10.0
    fig_height = 1.0 + n_rows * cell_h

    fig = plt.figure(figsize=(fig_width, fig_height))

    # page title
    page_label = f" (page {page_num} of {total_pages})" if total_pages > 1 else ""
    fig.suptitle(
        f"Anchor: {bed_chr}:{bed_start:,}-{bed_end:,}{page_label}",
        fontsize=11, y=0.98
    )

    # GridSpec — leave generous left margin for row labels
    gs = gridspec.GridSpec(
        n_rows, 3,
        hspace=0.55,
        wspace=0.20,
        left=0.22,
        right=0.97,
        top=0.88,
        bottom=0.04,
    )

    col_names = ['Unphased', 'Ref (hap1)', 'Alt (hap2)']

    for row_idx, entry in enumerate(loop_entries):
        loop = entry['loop']

        # view window centred on each anchor midpoint
        mid0 = (loop['start0'] + loop['end0']) // 2
        mid1 = (loop['start1'] + loop['end1']) // 2

        view_start0 = max(0, mid0 - width)
        view_end0   = mid0 + width
        view_start1 = max(0, mid1 - width)
        view_end1   = mid1 + width

        chr_num = loop['chr0']   # anchor0 chromosome (same as anchor1 for cis loops)

        # fetch raw matrices
        raw = get_hic_matrices(
            hic_map_configs, chr_num,
            view_start0, view_end0,
            view_start1, view_end1,
            resolution, cache
        )

        # process matrices and determine per-column color scales
        # col 0 (unphased): tanh/MAD preprocessing
        proc_unphased = preprocess(raw[0], scale=hic_map_configs[0]['scale'])
        unphased_vmin = float(proc_unphased.min())
        unphased_vmax = float(proc_unphased.max())

        # col 1 & 2 (phased): raw, shared per-row vmax
        phased_vmax = max(float(raw[1].max()), float(raw[2].max()))

        display_matrices = [proc_unphased, raw[1], raw[2]]
        vmins = [unphased_vmin, 0.0,        0.0]
        vmaxs = [unphased_vmax, phased_vmax, phased_vmax]

        # row label (left margin of leftmost cell)
        row_label = (
            f"Loop {row_idx + 1}\n"
            f"{loop['chr0']}:{loop['start0']:,}-{loop['end0']:,}\n"
            f"× {loop['chr1']}:{loop['start1']:,}-{loop['end1']:,}"
        )

        for col_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            plot_mini_hic(ax, display_matrices[col_idx], vmins[col_idx], vmaxs[col_idx])

            # column header on first row only
            if row_idx == 0:
                ax.set_title(col_names[col_idx], fontsize=9, pad=4)

            # row label on leftmost column only
            if col_idx == 0:
                ax.set_ylabel(
                    row_label,
                    fontsize=5.5,
                    rotation=0,
                    labelpad=105,
                    va='center',
                    ha='right',
                )

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# ── helpers ───────────────────────────────────────────────────────────────────

def chunk(lst, n):
    '''Yields successive n-sized chunks from lst.'''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate anchor-centric HiC certificate plots. '
                    'Produces one PDF page per belt anchor showing all '
                    'associated loops as mini HiC maps (unphased | ref | alt).\n\n'
                    'All parameters can be supplied via a JSON config file '
                    '(--config config.json). CLI args override config file values.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # config file (loaded first; CLI args override)
    parser.add_argument('--config', metavar='CONFIG.JSON',
                        help='JSON config file. Any key matching a CLI arg name '
                             'sets that arg\'s default. CLI args override config values.')

    # inputs (all optional at parse time; validated below)
    parser.add_argument('--outp_dir',   default=None, help='Output directory')
    parser.add_argument('--anchor_bed', default=None, help='BED file of belt anchors (CHR START END)')
    parser.add_argument('--loop_bedpe', default=None,
                        help='BEDPE loop file (CHR START0 END0 CHR START1 END1 ...)')

    # HiC maps
    parser.add_argument('--unphased_hic', default=None,
                        help='Path or URL for the unphased HiC map')
    parser.add_argument('--ref_hic', default=None,
                        help='Path or URL for the ref/hap1 phased HiC map')
    parser.add_argument('--alt_hic', default=None,
                        help='Path or URL for the alt/hap2 phased HiC map')

    # HiC parameters
    parser.add_argument('--norm',       default=None, help='HiC normalization (default: KR)')
    parser.add_argument('--resolution', default=None, type=int,
                        help='Resolution in bp (default: 1000)')
    parser.add_argument('--width',      default=None, type=int,
                        help='Half-width of the view window around each anchor midpoint in bp '
                             '(default: 250000 → 500 kb total view)')

    # output
    parser.add_argument('--output_name',        default=None,
                        help='Base name for the output PDF (default: anchor_hic_plots)')
    parser.add_argument('--max_loops_per_page', default=None, type=int,
                        help='Max loops shown per anchor page before splitting (default: 12)')

    # ── two-pass parse: config file first, then CLI overrides ─────────────────
    # First pass: just extract --config (ignore everything else)
    pre_args, _ = parser.parse_known_args()

    config_defaults = {}
    if pre_args.config:
        with open(pre_args.config) as fh:
            config_defaults = json.load(fh)
        parser.set_defaults(**config_defaults)

    # Second pass: full parse (CLI values override config defaults)
    args = parser.parse_args()

    # apply hard-coded defaults for anything still None
    if args.norm               is None: args.norm               = 'NONE'
    if args.resolution         is None: args.resolution         = 1000
    if args.width              is None: args.width              = 250000
    if args.output_name        is None: args.output_name        = 'anchor_hic_plots'
    if args.max_loops_per_page is None: args.max_loops_per_page = 12

    # validate required fields
    missing = [name for name, val in [
        ('--outp_dir',    args.outp_dir),
        ('--anchor_bed',  args.anchor_bed),
        ('--loop_bedpe',  args.loop_bedpe),
        ('--unphased_hic', args.unphased_hic),
        ('--ref_hic',     args.ref_hic),
        ('--alt_hic',     args.alt_hic),
    ] if val is None]
    if missing:
        parser.error(
            f"The following required arguments are missing (supply via CLI or --config):\n"
            + '\n'.join(f'  {m}' for m in missing)
        )

    os.makedirs(args.outp_dir, exist_ok=True)

    # assemble hic_map_configs from CLI args
    def make_config(name, hic_path, norm, scale):
        key = 'url' if hic_path.startswith('http') else 'path'
        return {'name': name, 'norm': norm, 'scale': scale, key: hic_path}

    hic_map_configs = [
        make_config('Unphased',   args.unphased_hic, args.norm, 0.1),
        make_config('Ref (hap1)', args.ref_hic,      args.norm, 1.0),
        make_config('Alt (hap2)', args.alt_hic,      args.norm, 1.0),
    ]

    print(f"{ts()} anchor_hic_plots.py — arguments:")
    print(f"  Output dir:         {args.outp_dir}")
    print(f"  Anchor BED:         {args.anchor_bed}")
    print(f"  Loop BEDPE:         {args.loop_bedpe}")
    print(f"  Unphased HiC:       {args.unphased_hic}")
    print(f"  Ref HiC:            {args.ref_hic}")
    print(f"  Alt HiC:            {args.alt_hic}")
    print(f"  Norm / Resolution:  {args.norm} / {args.resolution} bp  "
          f"(alternatives: KR, VC_SQRT, VC — must be pre-computed in the .hic file)")
    print(f"  View half-width:    {args.width:,} bp  ({args.width*2/1000:.0f} kb total)")
    print(f"  Max loops/page:     {args.max_loops_per_page}")

    # ── parse inputs ──────────────────────────────────────────────────────────
    print(f"\n{ts()} Parsing anchor BED...", flush=True)
    anchors = parse_anchor_bed(args.anchor_bed)
    if not anchors:
        print(f"{ts()} ERROR: No anchors found in BED file. Exiting.")
        sys.exit(1)
    print(f"{ts()}   {len(anchors)} anchor(s) loaded.")

    print(f"{ts()} Parsing loop BEDPE...", flush=True)
    loops = parse_loop_bedpe(args.loop_bedpe)
    print(f"{ts()}   {len(loops)} loop(s) loaded.")

    # ── group loops by anchor ─────────────────────────────────────────────────
    print(f"{ts()} Grouping loops by anchor...", flush=True)
    anchor_groups = group_loops_by_anchor(loops, anchors)
    for anchor, entries in anchor_groups.items():
        logger.debug(f"  {anchor[0]}:{anchor[1]}-{anchor[2]} → {len(entries)} loop(s)")

    # ── generate PDF ──────────────────────────────────────────────────────────
    pdf_path = os.path.join(args.outp_dir, args.output_name + '.pdf')
    print(f"\n{ts()} Writing PDF: {pdf_path}", flush=True)

    hic_cache = {}
    total_pages = 0

    with PdfPages(pdf_path) as pdf:
        for anchor in anchors:
            entries = anchor_groups[anchor]
            bed_chr, bed_start, bed_end = anchor

            if not entries:
                print(f"{ts()}   {bed_chr}:{bed_start:,}-{bed_end:,}  — no loops, emitting placeholder page.")
                emit_no_loops_page(pdf, anchor)
                total_pages += 1
                continue

            pages = list(chunk(entries, args.max_loops_per_page))
            n_anchor_pages = len(pages)
            print(
                f"{ts()}   {bed_chr}:{bed_start:,}-{bed_end:,}  — "
                f"{len(entries)} loop(s), {n_anchor_pages} page(s).",
                flush=True
            )

            for page_num, page_entries in enumerate(pages, start=1):
                plot_anchor_page(
                    pdf, anchor, page_entries,
                    hic_map_configs, args.width,
                    args.resolution, hic_cache,
                    page_num, n_anchor_pages
                )
                total_pages += 1

    print(f"\n{ts()} Done. {total_pages} page(s) written to {pdf_path}")


if __name__ == '__main__':
    main()
