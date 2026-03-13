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
import time
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

# ── helpers ────────────────────────────────────────────────────────────────────

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
    Returns the sum of raw contacts within the loop anchor region.

    bin_r0/r1 — row (y-axis) bin range for anchor-0 start/end
    bin_c0/c1 — col (x-axis) bin range for anchor-1 start/end

    Uses floor for start and ceil for end so that anchors smaller than one
    resolution bin still contribute at least 1 bin on each axis (rather than
    collapsing to an empty slice and returning 0).

    Values are clamped to the matrix bounds.
    '''
    ri0 = max(0,               int(np.floor(bin_r0)))
    ri1 = min(matrix.shape[0], max(ri0 + 1, int(np.ceil(bin_r1))))
    ci0 = max(0,               int(np.floor(bin_c0)))
    ci1 = min(matrix.shape[1], max(ci0 + 1, int(np.ceil(bin_c1))))
    return float(matrix[ri0:ri1, ci0:ci1].sum())


def _hic_fetch_with_retry(fn, label, max_retries=4, base_delay=3.0):
    '''
    Calls fn() and retries up to max_retries times on any exception.
    Uses exponential backoff: 3 s, 6 s, 12 s, 24 s …
    Raises the last exception if all attempts fail.
    '''
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"[retry {attempt + 1}/{max_retries - 1}] {label}: {e}"
                f" — retrying in {delay:.0f}s"
            )
            time.sleep(delay)

# ── parsing ───────────────────────────────────────────────────────────────────

def parse_anchor_bed(bed_file):
    '''
    Parses a BED file of belt anchors.

    Expected columns:
        CHR  START  END  [AG  [CHIPSEQ]]

    Returns a list of dicts with keys:
        chr     — chromosome string
        start   — int
        end     — int
        ag      — float or None  (column 4, AlphaGenome predicted signal)
        chipseq — float or None  (column 5, empiric ChIP-seq signal)

    Columns 4 and 5 are optional; backward-compatible with 3-column BED files.
    Lines starting with '#' are skipped.
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
                'ag':      _try_float(parts[3]) if len(parts) > 3 and parts[3] not in ('', '.') else None,
                'chipseq': _try_float(parts[4]) if len(parts) > 4 and parts[4] not in ('', '.') else None,
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

    anchors: list of dicts (keys: chr, start, end, ag, chipseq).
    Returns: {(chr, start, end): [{'loop': dict, 'matching_anchor': 0|1}, ...]}

    Standard interval overlap:
        bedpe_start < bed_end  AND  bedpe_end > bed_start  (same chromosome)

    Both BEDPE anchors overlapping the same BED anchor → added once (anchor0),
    WARNING logged.  Anchors with no loops → empty list.
    '''
    anchor_groups = {(a['chr'], a['start'], a['end']): [] for a in anchors}

    for loop in loops:
        for a in anchors:
            bed_chr, bed_start, bed_end = a['chr'], a['start'], a['end']
            key = (bed_chr, bed_start, bed_end)

            a0 = loop['chr0'] == bed_chr and loop['start0'] < bed_end and loop['end0'] > bed_start
            a1 = loop['chr1'] == bed_chr and loop['start1'] < bed_end and loop['end1'] > bed_start

            if a0 and a1:
                logger.warning(
                    f"Both anchors of loop "
                    f"{loop['chr0']}:{loop['start0']}-{loop['end0']} × "
                    f"{loop['chr1']}:{loop['start1']}-{loop['end1']} "
                    f"overlap BED anchor {bed_chr}:{bed_start}-{bed_end}. "
                    f"Adding once (via anchor0)."
                )
                anchor_groups[key].append({'loop': loop, 'matching_anchor': 0})
            elif a0:
                anchor_groups[key].append({'loop': loop, 'matching_anchor': 0})
            elif a1:
                anchor_groups[key].append({'loop': loop, 'matching_anchor': 1})

    return anchor_groups

# ── HiC matrix fetch ──────────────────────────────────────────────────────────

def get_hic_matrices(hic_map_configs, chr_num, start0, end0, start1, end1, resolution, cache):
    '''
    Fetches HiC contact matrices for a given region from all 3 HiC maps.
    Returns a list of 3 numpy arrays: [unphased, ref, alt].

    cache: dict keyed on (hic_path, chr_num, norm) -> (HiCFile, MatrixZoomData)

    Network errors are retried with exponential backoff (_hic_fetch_with_retry).
    A failed getRecordsAsMatrix call also invalidates the cache entry so the
    file is re-opened on the next loop rather than reusing a broken handle.
    On persistent failure a zero matrix is returned and the error is logged.
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
                def _open():
                    hf = hicstraw.HiCFile(hic_path)
                    mo = hf.getMatrixZoomData(
                        chr_num, chr_num, "observed", hic_norm, "BP", resolution
                    )
                    return hf, mo
                hic_file, matrix_obj = _hic_fetch_with_retry(
                    _open, f"open {hic_path} chr={chr_num}"
                )
                cache[cache_key] = (hic_file, matrix_obj)
            except Exception as e:
                logger.error(
                    f"Failed to open HiC '{hic_path}' for chr '{chr_num}' after retries: {e}"
                )
                cache[cache_key] = None

        cached = cache[cache_key]
        if cached is None:
            matrices.append(np.zeros((n_bins_x, n_bins_y)))
        else:
            _, matrix_obj = cached
            try:
                def _fetch():
                    return matrix_obj.getRecordsAsMatrix(start0, end0, start1, end1)
                numpy_matrix = _hic_fetch_with_retry(
                    _fetch, f"{chr_num}:{start0}-{end0}×{start1}-{end1}"
                )
                matrices.append(numpy_matrix)
            except Exception as e:
                logger.error(
                    f"Failed to fetch matrix {chr_num}:{start0}-{end0} × "
                    f"{start1}-{end1} after retries: {e}"
                )
                cache[cache_key] = None   # invalidate — re-open next time
                matrices.append(np.zeros((n_bins_x, n_bins_y)))

    return matrices

# ── plotting ──────────────────────────────────────────────────────────────────

def plot_mini_hic(ax, matrix, vmin, vmax):
    '''Renders a single mini HiC matrix. Ticks and labels are hidden.'''
    ax.matshow(matrix, cmap=REDMAP, vmin=vmin, vmax=vmax, aspect='auto')
    ax.tick_params(
        axis='both', which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False
    )


def _build_header(anchor, page_num=None, total_pages=None):
    '''Page suptitle: anchor coords | AG pred | ChIP-seq | page N of M.'''
    parts = [f"Anchor: {anchor['chr']}:{anchor['start']:,}–{anchor['end']:,}"]
    if anchor.get('ag') is not None:
        parts.append(f"AlphaGenome pred: {anchor['ag']:.3g}")
    if anchor.get('chipseq') is not None:
        parts.append(f"empiric ChIP-seq: {anchor['chipseq']:.3g}")
    if page_num is not None and total_pages is not None and total_pages > 1:
        parts.append(f"page {page_num} of {total_pages}")
    return "   |   ".join(parts)


def emit_no_loops_page(pdf, anchor):
    '''Emits a placeholder page for anchors with no associated loops.'''
    fig = plt.figure(figsize=(8, 3))
    fig.suptitle(_build_header(anchor), fontsize=9, y=0.97)
    fig.text(
        0.5, 0.4,
        f"No loops found for anchor "
        f"{anchor['chr']}:{anchor['start']:,}-{anchor['end']:,}",
        ha='center', va='center', fontsize=11
    )
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def plot_anchor_page(pdf, anchor, loop_entries, hic_map_configs, width,
                     resolution, cache, page_num, total_pages):
    '''
    Creates one PDF page for a given belt anchor.

    Layout   : n_rows × 3 grid — Unphased | Ref (hap1) | Alt (hap2)
    Header   : anchor coords | AlphaGenome pred | empiric ChIP-seq | page N/M
    Cell title: column name  +  Σ=N  (contacts within the loop anchor region)
    xlabel   : view/loop coords at top of first column only
    Row label: loop coords, distance, R/A ratio, log2FC
    Box      : dodgerblue rectangle at the loop contact point (min 3 bins)
    Crosshairs: dashed lines at each anchor midpoint
    Color    : independent per-column scale (unphased not shared with phased)
    '''
    plt.rcParams['font.family'] = 'monospace'

    n_rows     = len(loop_entries)
    fig_width  = 10.0
    fig_height = 1.2 + n_rows * 2.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle(_build_header(anchor, page_num, total_pages), fontsize=9, y=0.995)

    gs = gridspec.GridSpec(
        n_rows, 3,
        hspace=0.65, wspace=0.20,
        left=0.22, right=0.97, top=0.84, bottom=0.04,
    )

    col_names = ['Unphased', 'Ref (hap1)', 'Alt (hap2)']

    for row_idx, entry in enumerate(loop_entries):
        loop = entry['loop']

        # ── view window centred on each anchor midpoint ───────────────────────
        mid0 = (loop['start0'] + loop['end0']) // 2
        mid1 = (loop['start1'] + loop['end1']) // 2

        view_start0 = max(0, mid0 - width);  view_end0 = mid0 + width
        view_start1 = max(0, mid1 - width);  view_end1 = mid1 + width

        chr_num = loop['chr0']

        # ── fetch raw matrices ────────────────────────────────────────────────
        raw = get_hic_matrices(
            hic_map_configs, chr_num,
            view_start0, view_end0,
            view_start1, view_end1,
            resolution, cache
        )

        # ── anchor extents in bin-index space ────────────────────────────────
        # row axis (y) = anchor-0;  col axis (x) = anchor-1
        bin_a0_start = (loop['start0'] - view_start0) / resolution
        bin_a0_end   = (loop['end0']   - view_start0) / resolution
        bin_a1_start = (loop['start1'] - view_start1) / resolution
        bin_a1_end   = (loop['end1']   - view_start1) / resolution

        # ── Σ: raw contacts within the anchor region (at least 1 bin each axis)
        contact_sums = [
            compute_anchor_sum(raw[i], bin_a0_start, bin_a0_end, bin_a1_start, bin_a1_end)
            for i in range(3)
        ]

        # ── preprocess; independent color scale per column ───────────────────
        proc  = [preprocess(raw[i], scale=hic_map_configs[i]['scale']) for i in range(3)]
        vmins = [m.min() for m in proc]
        vmaxs = [m.max() for m in proc]

        # ── crosshair + box positions in bin space ────────────────────────────
        bin_mid0 = (mid0 - view_start0) / resolution
        bin_mid1 = (mid1 - view_start1) / resolution

        MIN_BOX_BINS = 3
        box_w = max(MIN_BOX_BINS, bin_a1_end - bin_a1_start)
        box_h = max(MIN_BOX_BINS, bin_a0_end - bin_a0_start)

        # ── allelic metrics ───────────────────────────────────────────────────
        s_ref, s_alt = contact_sums[1], contact_sums[2]
        if s_ref > 0 and s_alt > 0:
            ratio  = s_ref / s_alt
            log2fc = np.log2(ratio)
            allelic_str = f"R/A: {ratio:.2f}\u00d7   log2FC: {log2fc:+.2f}"
        elif s_ref == 0 and s_alt == 0:
            allelic_str = "R/A: N/A   log2FC: N/A"
        elif s_alt == 0:
            allelic_str = "R/A: \u221e   log2FC: +\u221e"
        else:
            allelic_str = "R/A: 0   log2FC: \u2212\u221e"

        row_label = (
            f"Loop {row_idx + 1}\n"
            f"{loop['chr0']}:{loop['start0']:,}-{loop['end0']:,}\n"
            f"\u00d7 {loop['chr1']}:{loop['start1']:,}-{loop['end1']:,}\n"
            f"dist: {abs(mid1 - mid0) / 1000:.0f} kb\n"
            f"{allelic_str}"
        )

        # ── render each column ────────────────────────────────────────────────
        for col_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            plot_mini_hic(ax, proc[col_idx], vmins[col_idx], vmaxs[col_idx])

            # crosshairs
            ax.axhline(bin_mid0, color='dodgerblue', linewidth=0.5, linestyle='--', alpha=0.6)
            ax.axvline(bin_mid1, color='dodgerblue', linewidth=0.5, linestyle='--', alpha=0.6)

            # loop contact box (centred on anchor midpoints, min 3 bins visible)
            ax.add_patch(Rectangle(
                (bin_mid1 - box_w / 2 - 0.5, bin_mid0 - box_h / 2 - 0.5),
                box_w, box_h,
                linewidth=1.0, edgecolor='dodgerblue',
                facecolor='none', alpha=0.9, zorder=5,
            ))

            # cell title: column name + Σ (contacts within anchor)
            ax.set_title(
                f"{col_names[col_idx]}  \u03a3={contact_sums[col_idx]:,.0f}",
                fontsize=8, color='black'
            )

            # view/loop coords: first column only (same for all 3 cells in row)
            if col_idx == 0:
                ax.set_xlabel(
                    f"View: {view_start1:,} \u2013 {view_end1:,}"
                    f"\nLoop: {loop['start1']:,} \u2013 {loop['end1']:,}",
                    fontsize=6, color='black'
                )
                ax.xaxis.set_label_position('top')

            # row label (left margin, robust to figure-size changes)
            if col_idx == 0:
                ax.text(
                    -0.02, 0.5, row_label,
                    transform=ax.transAxes,
                    fontsize=6.5, ha='right', va='center',
                    fontfamily='monospace', clip_on=False,
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
        description='Generate anchor-centric HiC certificate plots (v3). '
                    'Produces one PDF page per belt anchor showing all '
                    'associated loops as mini HiC maps (unphased | ref | alt).\n\n'
                    'All parameters can be supplied via a JSON config file '
                    '(--config config.json). CLI args override config file values.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--config',    metavar='CONFIG.JSON')
    parser.add_argument('--outp_dir',  default=None)
    parser.add_argument('--anchor_bed', default=None,
                        help='BED: CHR START END [AG [CHIPSEQ]]')
    parser.add_argument('--loop_bedpe', default=None,
                        help='BEDPE: CHR START0 END0 CHR START1 END1 ...')
    parser.add_argument('--unphased_hic', default=None)
    parser.add_argument('--ref_hic',      default=None)
    parser.add_argument('--alt_hic',      default=None)
    parser.add_argument('--norm',         default=None,
                        help='HiC normalization (default: NONE)')
    parser.add_argument('--resolution',   default=None, type=int,
                        help='Resolution in bp (default: 1000)')
    parser.add_argument('--width',        default=None, type=int,
                        help='Half-width of view window in bp (default: 250000)')
    parser.add_argument('--output_name',        default=None)
    parser.add_argument('--max_loops_per_page', default=None, type=int)

    # two-pass parse: config file first, CLI overrides second
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as fh:
            parser.set_defaults(**json.load(fh))
    args = parser.parse_args()

    if args.norm               is None: args.norm               = 'NONE'
    if args.resolution         is None: args.resolution         = 1000
    if args.width              is None: args.width              = 250000
    if args.output_name        is None: args.output_name        = 'anchor_hic_plots'
    if args.max_loops_per_page is None: args.max_loops_per_page = 12

    missing = [n for n, v in [
        ('--outp_dir',     args.outp_dir),
        ('--anchor_bed',   args.anchor_bed),
        ('--loop_bedpe',   args.loop_bedpe),
        ('--unphased_hic', args.unphased_hic),
        ('--ref_hic',      args.ref_hic),
        ('--alt_hic',      args.alt_hic),
    ] if v is None]
    if missing:
        parser.error("Missing required args:\n" + '\n'.join(f'  {m}' for m in missing))

    os.makedirs(args.outp_dir, exist_ok=True)

    def make_config(name, hic_path, norm, scale):
        key = 'url' if hic_path.startswith('http') else 'path'
        return {'name': name, 'norm': norm, 'scale': scale, key: hic_path}

    hic_map_configs = [
        make_config('Unphased',   args.unphased_hic, args.norm, 0.1),
        make_config('Ref (hap1)', args.ref_hic,      args.norm, 1.0),
        make_config('Alt (hap2)', args.alt_hic,      args.norm, 1.0),
    ]

    print(f"{ts()} anchor_hic_plots_v3.py — arguments:")
    print(f"  Output dir:         {args.outp_dir}")
    print(f"  Anchor BED:         {args.anchor_bed}")
    print(f"  Loop BEDPE:         {args.loop_bedpe}")
    print(f"  Unphased HiC:       {args.unphased_hic}")
    print(f"  Ref HiC:            {args.ref_hic}")
    print(f"  Alt HiC:            {args.alt_hic}")
    print(f"  Norm / Resolution:  {args.norm} / {args.resolution} bp")
    print(f"  View half-width:    {args.width:,} bp  ({args.width * 2 / 1000:.0f} kb total)")
    print(f"  Max loops/page:     {args.max_loops_per_page}")

    print(f"\n{ts()} Parsing anchor BED...", flush=True)
    anchors = parse_anchor_bed(args.anchor_bed)
    if not anchors:
        print(f"{ts()} ERROR: No anchors found. Exiting.")
        sys.exit(1)
    print(f"{ts()}   {len(anchors)} anchor(s) loaded.", flush=True)

    n_ag      = sum(1 for a in anchors if a.get('ag')      is not None)
    n_chipseq = sum(1 for a in anchors if a.get('chipseq') is not None)
    if n_ag or n_chipseq:
        print(f"{ts()}   AlphaGenome values: {n_ag}/{len(anchors)}  "
              f"|  ChIP-seq values: {n_chipseq}/{len(anchors)}", flush=True)

    print(f"{ts()} Parsing loop BEDPE...", flush=True)
    loops = parse_loop_bedpe(args.loop_bedpe)
    print(f"{ts()}   {len(loops)} loop(s) loaded.", flush=True)

    print(f"{ts()} Grouping loops by anchor...", flush=True)
    anchor_groups = group_loops_by_anchor(loops, anchors)
    for a in anchors:
        key = (a['chr'], a['start'], a['end'])
        logger.debug(f"  {key[0]}:{key[1]}-{key[2]} → {len(anchor_groups[key])} loop(s)")

    pdf_path = os.path.join(args.outp_dir, args.output_name + '.pdf')
    print(f"\n{ts()} Writing PDF: {pdf_path}", flush=True)

    hic_cache   = {}
    total_pages = 0

    with PdfPages(pdf_path) as pdf:
        for a in anchors:
            key     = (a['chr'], a['start'], a['end'])
            entries = anchor_groups[key]

            if not entries:
                print(f"{ts()}   {a['chr']}:{a['start']:,}-{a['end']:,}"
                      f"  — no loops, emitting placeholder.", flush=True)
                emit_no_loops_page(pdf, a)
                total_pages += 1
                continue

            pages          = list(chunk(entries, args.max_loops_per_page))
            n_anchor_pages = len(pages)
            print(f"{ts()}   {a['chr']}:{a['start']:,}-{a['end']:,}"
                  f"  — {len(entries)} loop(s), {n_anchor_pages} page(s).", flush=True)

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
