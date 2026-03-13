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
from matplotlib.patches import Rectangle

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


def compute_anchor_sum(matrix, bin_r0, bin_r1, bin_c0, bin_c1):
    '''
    Returns the sum of raw contacts within the rectangle defined by the actual
    loop anchor extents in bin-index space.

    bin_r0/r1 — row (y-axis) bin range corresponding to anchor-0 start/end
    bin_c0/c1 — col (x-axis) bin range corresponding to anchor-1 start/end

    Values are clamped to the matrix bounds. Returns 0.0 if the clamped region
    is empty.
    '''
    ri0 = max(0,               int(bin_r0))
    ri1 = min(matrix.shape[0], int(bin_r1))
    ci0 = max(0,               int(bin_c0))
    ci1 = min(matrix.shape[1], int(bin_c1))
    if ri0 >= ri1 or ci0 >= ci1:
        return 0.0
    return float(matrix[ri0:ri1, ci0:ci1].sum())

# ── parsing ───────────────────────────────────────────────────────────────────

def parse_anchor_bed(bed_file):
    '''
    Parses a BED file of belt anchors.

    Expected columns:
        CHR  START  END  [CHIPSEQ  [AG]]

    Returns a list of dicts with keys:
        chr     — chromosome string
        start   — int
        end     — int
        ag      — float or None  (column 4, AlphaGenome predicted signal)
        chipseq — float or None  (column 5, empiric ChIP-seq signal)

    Columns 4 and 5 are optional; the function is backward-compatible with
    3-column BED files. Lines starting with '#' are skipped.
    '''
    def _try_float(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    anchors = []
    with open(bed_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                parts = line.split()
            anchors.append({
                'chr':     parts[0],
                'start':   int(parts[1]),
                'end':     int(parts[2]),
                'ag': _try_float(parts[3]) if len(parts) > 3 and parts[3] not in ('', '.') else None,
                'chipseq':      _try_float(parts[4]) if len(parts) > 4 and parts[4] not in ('', '.') else None,
            })
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

    anchors: list of dicts (keys: chr, start, end, chipseq, ag).
    anchor_groups: keyed by (chr, start, end) tuples for hashability.

    For each BED anchor, finds loops where anchor0 OR anchor1 overlaps the
    BED anchor via standard genomic interval overlap:
        bedpe_start < bed_end  AND  bedpe_end > bed_start  (same chromosome)

    Returns:
        {(chr, start, end): [{'loop': loop_dict, 'matching_anchor': 0 or 1}, ...]}

    All BED anchors appear as keys (empty list if no loops match).
    If both BEDPE anchors overlap the same BED anchor, the loop is added once
    (via anchor0) and a warning is logged.
    '''
    anchor_groups = {(a['chr'], a['start'], a['end']): [] for a in anchors}

    for loop in loops:
        for a in anchors:
            bed_chr   = a['chr']
            bed_start = a['start']
            bed_end   = a['end']
            key = (bed_chr, bed_start, bed_end)

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
                anchor_groups[key].append({'loop': loop, 'matching_anchor': 0})
            elif a0_overlaps:
                anchor_groups[key].append({'loop': loop, 'matching_anchor': 0})
            elif a1_overlaps:
                anchor_groups[key].append({'loop': loop, 'matching_anchor': 1})

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


def _build_header(anchor, page_num=None, total_pages=None):
    '''
    Builds the page suptitle string from anchor dict fields.
    Includes ChIP-seq and AlphaGenome values if present.
    Appends pagination if total_pages > 1.
    '''
    bed_chr   = anchor['chr']
    bed_start = anchor['start']
    bed_end   = anchor['end']
    chipseq   = anchor.get('chipseq')
    ag        = anchor.get('ag')

    parts = [f"Anchor: {bed_chr}:{bed_start:,}–{bed_end:,}"]
    if chipseq is not None:
        parts.append(f"empiric ChIP-seq: {chipseq:.3g}")
    if ag is not None:
        parts.append(f"AlphaGenome pred: {ag:.3g}")
    if page_num is not None and total_pages is not None and total_pages > 1:
        parts.append(f"page {page_num} of {total_pages}")

    return "   |   ".join(parts)


def emit_no_loops_page(pdf, anchor):
    '''Emits a placeholder page noting no loops were found for this anchor.'''
    bed_chr   = anchor['chr']
    bed_start = anchor['start']
    bed_end   = anchor['end']

    fig = plt.figure(figsize=(8, 3))
    fig.suptitle(_build_header(anchor), fontsize=9, y=0.97)
    fig.text(
        0.5, 0.4,
        f"No loops found for anchor {bed_chr}:{bed_start:,}-{bed_end:,}",
        ha='center', va='center', fontsize=11
    )
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def plot_anchor_page(pdf, anchor, loop_entries, hic_map_configs, width,
                     resolution, cache, page_num, total_pages):
    '''
    Creates one PDF page for a given belt anchor.

    Layout: n_rows × 3 grid of mini HiC maps, one row per loop.
    Columns (left to right): Unphased | Ref (hap1) | Alt (hap2)

    Header: Anchor coords  |  empiric ChIP-seq: X  |  AlphaGenome pred: Y  |  page N of M

    Per-cell title: column name (first row only) + contact sum Σ=N (every row).
    Crosshairs (dashed blue) mark the anchor midpoints within each HiC panel.

    Color scale: preprocess() applied to all 3 columns; vmin/vmax shared per row
    so all three maps are visually comparable.

    Row labels include loop coordinates and inter-anchor distance.
    '''
    plt.rcParams['font.family'] = 'monospace'

    n_rows = len(loop_entries)

    cell_h    = 2.5    # inches per loop row
    fig_width = 10.0
    fig_height = 1.2 + n_rows * cell_h

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle(
        _build_header(anchor, page_num, total_pages),
        fontsize=9, y=0.995
    )

    # GridSpec — leave generous left margin for row labels
    gs = gridspec.GridSpec(
        n_rows, 3,
        hspace=0.65,
        wspace=0.20,
        left=0.22,
        right=0.97,
        top=0.84,
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

        # anchor extents in bin-index space
        # row axis (y) = anchor 0 region;  col axis (x) = anchor 1 region
        bin_a0_start = (loop['start0'] - view_start0) / resolution
        bin_a0_end   = (loop['end0']   - view_start0) / resolution
        bin_a1_start = (loop['start1'] - view_start1) / resolution
        bin_a1_end   = (loop['end1']   - view_start1) / resolution

        # contact sums: raw counts within the actual anchor overlap rectangle
        contact_sums = [
            compute_anchor_sum(raw[i], bin_a0_start, bin_a0_end, bin_a1_start, bin_a1_end)
            for i in range(3)
        ]

        # preprocess all 3 columns; each gets its own independent color scale
        # so the unphased map (far more reads) doesn't wash out ref/alt
        proc  = [preprocess(raw[i], scale=hic_map_configs[i]['scale']) for i in range(3)]
        vmins = [m.min() for m in proc]
        vmaxs = [m.max() for m in proc]

        # crosshair midpoints in bin-index space (centre of each anchor)
        bin_mid0 = (mid0 - view_start0) / resolution   # horizontal line (row axis)
        bin_mid1 = (mid1 - view_start1) / resolution   # vertical line   (col axis)

        # inter-anchor distance for row label
        loop_dist_kb = abs(mid1 - mid0) / 1000

        # ref/alt ratio and log2 fold-change (comparable because matched coverage)
        s_ref = contact_sums[1]
        s_alt = contact_sums[2]
        if s_ref > 0 and s_alt > 0:
            ratio  = s_ref / s_alt
            log2fc = np.log2(ratio)
            allelic_str = f"R/A: {ratio:.2f}×   log2FC: {log2fc:+.2f}"
        elif s_ref == 0 and s_alt == 0:
            allelic_str = "R/A: N/A   log2FC: N/A"
        elif s_alt == 0:
            allelic_str = "R/A: \u221e   log2FC: +\u221e"
        else:  # s_ref == 0
            allelic_str = "R/A: 0   log2FC: -\u221e"

        row_label = (
            f"Loop {row_idx + 1}\n"
            f"{loop['chr0']}:{loop['start0']:,}-{loop['end0']:,}\n"
            f"× {loop['chr1']}:{loop['start1']:,}-{loop['end1']:,}\n"
            f"dist: {loop_dist_kb:.0f} kb\n"
            f"{allelic_str}"
        )

        for col_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            plot_mini_hic(ax, proc[col_idx], vmins[col_idx], vmaxs[col_idx])

            # crosshairs at anchor midpoints
            ax.axhline(bin_mid0, color='dodgerblue', linewidth=0.5, linestyle='--', alpha=0.6)
            ax.axvline(bin_mid1, color='dodgerblue', linewidth=0.5, linestyle='--', alpha=0.6)

            # box indicating the loop contact point — centred on the two anchor
            # midpoints, sized to the actual anchor footprints but clamped to a
            # minimum of 3 bins so it's always visible regardless of resolution
            MIN_BOX_BINS = 3
            box_w = max(MIN_BOX_BINS, bin_a1_end - bin_a1_start)
            box_h = max(MIN_BOX_BINS, bin_a0_end - bin_a0_start)
            ax.add_patch(Rectangle(
                (bin_mid1 - box_w / 2 - 0.5, bin_mid0 - box_h / 2 - 0.5),
                box_w, box_h,
                linewidth=1.0, edgecolor='dodgerblue',
                facecolor='none', alpha=0.9, zorder=5,
            ))

            # cell title: column name + contact sum, shown on every cell and every row
            ax.set_title(
                f"{col_names[col_idx]}  \u03a3={contact_sums[col_idx]:,.0f}",
                fontsize=8, color='black'
            )

            # xlabel at top: view/loop coordinates — first column only
            # (same info for all 3 cells in a row, so no need to repeat)
            if col_idx == 0:
                ax.set_xlabel(
                    f"View: {view_start1:,} \u2013 {view_end1:,}"
                    f"\nLoop: {loop['start1']:,} \u2013 {loop['end1']:,}",
                    fontsize=6, color='black'
                )
                ax.xaxis.set_label_position('top')

            # row label — positioned to the left of the leftmost cell using
            # transAxes coordinates (robust to figure size changes)
            if col_idx == 0:
                ax.text(
                    -0.02, 0.5, row_label,
                    transform=ax.transAxes,
                    fontsize=6.5, ha='right', va='center',
                    fontfamily='monospace',
                    clip_on=False,
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
    parser.add_argument('--anchor_bed', default=None,
                        help='BED file of belt anchors (CHR START END [CHIPSEQ [AG]])')
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
    parser.add_argument('--norm',       default=None,
                        help='HiC normalization (default: NONE; alternatives: KR, VC_SQRT, VC)')
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
        ('--outp_dir',     args.outp_dir),
        ('--anchor_bed',   args.anchor_bed),
        ('--loop_bedpe',   args.loop_bedpe),
        ('--unphased_hic', args.unphased_hic),
        ('--ref_hic',      args.ref_hic),
        ('--alt_hic',      args.alt_hic),
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

    print(f"{ts()} anchor_hic_plots_v2.py — arguments:")
    print(f"  Output dir:         {args.outp_dir}")
    print(f"  Anchor BED:         {args.anchor_bed}")
    print(f"  Loop BEDPE:         {args.loop_bedpe}")
    print(f"  Unphased HiC:       {args.unphased_hic}")
    print(f"  Ref HiC:            {args.ref_hic}")
    print(f"  Alt HiC:            {args.alt_hic}")
    print(f"  Norm / Resolution:  {args.norm} / {args.resolution} bp")
    print(f"  View half-width:    {args.width:,} bp  ({args.width*2/1000:.0f} kb total)")
    print(f"  Max loops/page:     {args.max_loops_per_page}")

    # ── parse inputs ──────────────────────────────────────────────────────────
    print(f"\n{ts()} Parsing anchor BED...", flush=True)
    anchors = parse_anchor_bed(args.anchor_bed)
    if not anchors:
        print(f"{ts()} ERROR: No anchors found in BED file. Exiting.")
        sys.exit(1)
    print(f"{ts()}   {len(anchors)} anchor(s) loaded.", flush=True)

    # report how many anchors have ChIP-seq / AG data
    n_chipseq = sum(1 for a in anchors if a.get('chipseq') is not None)
    n_ag      = sum(1 for a in anchors if a.get('ag')      is not None)
    if n_chipseq or n_ag:
        print(f"{ts()}   ChIP-seq values: {n_chipseq}/{len(anchors)}  "
              f"|  AlphaGenome values: {n_ag}/{len(anchors)}", flush=True)

    print(f"{ts()} Parsing loop BEDPE...", flush=True)
    loops = parse_loop_bedpe(args.loop_bedpe)
    print(f"{ts()}   {len(loops)} loop(s) loaded.", flush=True)

    # ── group loops by anchor ─────────────────────────────────────────────────
    print(f"{ts()} Grouping loops by anchor...", flush=True)
    anchor_groups = group_loops_by_anchor(loops, anchors)
    for a in anchors:
        key = (a['chr'], a['start'], a['end'])
        logger.debug(f"  {key[0]}:{key[1]}-{key[2]} → {len(anchor_groups[key])} loop(s)")

    # ── generate PDF ──────────────────────────────────────────────────────────
    pdf_path = os.path.join(args.outp_dir, args.output_name + '.pdf')
    print(f"\n{ts()} Writing PDF: {pdf_path}", flush=True)

    hic_cache   = {}
    total_pages = 0

    with PdfPages(pdf_path) as pdf:
        for a in anchors:
            key     = (a['chr'], a['start'], a['end'])
            entries = anchor_groups[key]

            if not entries:
                print(
                    f"{ts()}   {a['chr']}:{a['start']:,}-{a['end']:,}  "
                    f"— no loops, emitting placeholder page.",
                    flush=True
                )
                emit_no_loops_page(pdf, a)
                total_pages += 1
                continue

            pages          = list(chunk(entries, args.max_loops_per_page))
            n_anchor_pages = len(pages)
            print(
                f"{ts()}   {a['chr']}:{a['start']:,}-{a['end']:,}  — "
                f"{len(entries)} loop(s), {n_anchor_pages} page(s).",
                flush=True
            )

            for page_num, page_entries in enumerate(pages, start=1):
                plot_anchor_page(
                    pdf, a, page_entries,
                    hic_map_configs, args.width,
                    args.resolution, hic_cache,
                    page_num, n_anchor_pages
                )
                total_pages += 1

    print(f"\n{ts()} Done. {total_pages} page(s) written to {pdf_path}")


if __name__ == '__main__':
    main()
