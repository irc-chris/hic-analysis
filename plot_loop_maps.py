#!/usr/bin/env python3
"""
plot_loop_maps.py
-----------------
Standalone function to plot Hi-C contact maps for a single loop.

Layout: 2 rows × N columns
  Row 0 (overview): wide region around the loop, gray overlay outside zoom window
  Row 1 (zoomed):   tight region around both anchors, gray overlay outside loop contact area

Usage:
    from plot_loop_maps import plot_loop_maps
    from matplotlib.backends.backend_pdf import PdfPages

    loop = {"chr": "chr14", "start0": 106565000, "end0": 106566000,
                             "start1": 106570000, "end1": 106571000}

    with PdfPages("out.pdf") as pdf:
        plot_loop_maps(loop, [hic_unphased, hic_ref, hic_alt],
                       titles=["Unphased", "Ref", "Alt"], pdf=pdf)
"""

import math
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

from plot_hic_region import get_matrix_object

REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])


# ── matrix fetching ───────────────────────────────────────────────────────────

def _fetch(hic_path, chr_name, start, end, resolution, norm):
    """Fetch a symmetric square matrix for region [start, end) × [start, end)."""
    matrix_obj = get_matrix_object(hic_path, chr_name, resolution, norm)
    raw = matrix_obj.getRecordsAsMatrix(start, end, start, end)
    n = math.ceil((end - start) / resolution)
    return raw[:n, :n]


# ── gray overlay helpers ───────────────────────────────────────────────────────

def _gray_mask_outside(shape, row_lo, row_hi, col_lo, col_hi):
    """
    Returns a float32 array (same shape as matrix) that is 0.5 (gray) everywhere
    except in the rectangle [row_lo:row_hi, col_lo:col_hi] which is NaN
    (rendered transparent by imshow).
    """
    mask = np.full(shape, 0.5, dtype=np.float32)
    mask[row_lo:row_hi, col_lo:col_hi] = np.nan
    return mask


def _bin_slice(region_start, region_end, sub_start, sub_end, resolution):
    """Convert genomic coords to bin indices within [region_start, region_end)."""
    lo = max(0, (sub_start - region_start) // resolution)
    hi = min(math.ceil((region_end - region_start) / resolution),
             math.ceil((sub_end   - region_start) / resolution))
    return int(lo), int(hi)


# ── single panel drawing ──────────────────────────────────────────────────────

def _draw(ax, matrix, mask, region_start, region_end, chr_name,
          vmax, title, text_color="black"):
    ax.matshow(matrix, cmap=REDMAP, vmin=0, vmax=vmax)
    ax.imshow(mask, cmap='gray', alpha=0.5, vmin=0, vmax=1,
              origin='upper', interpolation='nearest')
    ax.set_title(title, fontsize=9, color=text_color, pad=4)
    ax.set_xlabel(f"{chr_name}:{region_start:,}–{region_end:,}", fontsize=7,
                  color=text_color)
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False,
                   labelleft=False, labelright=False)


# ── public function ───────────────────────────────────────────────────────────

def plot_loop_maps(
    loop,
    hic_paths,
    titles=None,
    resolution=1000,
    norm="NONE",
    vmax=None,
    overview_pad=50_000,
    zoom_pad=5_000,
    pdf=None,
    output_file="loop_maps.pdf",
):
    """
    Plot Hi-C contact maps for a single loop across multiple Hi-C files.

    Parameters
    ----------
    loop : dict
        Keys: chr, start0, end0, start1, end1
    hic_paths : list of str
        One path per Hi-C file.
    titles : list of str, optional
        Column titles; defaults to ["Map 0", "Map 1", ...].
    resolution : int
        Resolution in bp.
    norm : str
        Hi-C normalisation (e.g. "NONE", "KR").
    vmax : float or None
        Colour scale max.  If None, determined from data.
    overview_pad : int
        bp padding around the loop bounding box for the overview panel.
    zoom_pad : int
        bp padding around the loop bounding box for the zoomed panel.
    pdf : PdfPages or None
        If provided, saves into this PDF.  Otherwise saves to output_file.
    output_file : str
        Path to save if pdf is None.
    """
    chr_name = loop['chr']
    start0, end0 = loop['start0'], loop['end0']
    start1, end1 = loop['start1'], loop['end1']

    if titles is None:
        titles = [f"Map {i}" for i in range(len(hic_paths))]

    n_cols = len(hic_paths)

    # ── derive regions ────────────────────────────────────────────────────────
    lo = min(start0, start1)
    hi = max(end0,   end1)

    ov_s = ((lo - overview_pad) // resolution) * resolution
    ov_e = math.ceil((hi + overview_pad) / resolution) * resolution

    zm_s = ((lo - zoom_pad) // resolution) * resolution
    zm_e = math.ceil((hi + zoom_pad) / resolution) * resolution

    # ── fetch matrices ────────────────────────────────────────────────────────
    ov_matrices = [_fetch(p, chr_name, ov_s, ov_e, resolution, norm)
                   for p in hic_paths]
    zm_matrices = [_fetch(p, chr_name, zm_s, zm_e, resolution, norm)
                   for p in hic_paths]

    # ── auto vmax ─────────────────────────────────────────────────────────────
    if vmax is None:
        all_vals = np.concatenate([m.ravel() for m in ov_matrices + zm_matrices])
        vmax = float(np.nanpercentile(all_vals[all_vals > 0], 99)) if np.any(all_vals > 0) else 1.0

    # ── gray overlays ─────────────────────────────────────────────────────────
    # Overview: gray out everything outside the zoom window
    ov_shape = ov_matrices[0].shape
    zm_r_lo, zm_r_hi = _bin_slice(ov_s, ov_e, zm_s, zm_e, resolution)
    zm_c_lo, zm_c_hi = _bin_slice(ov_s, ov_e, zm_s, zm_e, resolution)
    ov_mask = _gray_mask_outside(ov_shape, zm_r_lo, zm_r_hi, zm_c_lo, zm_c_hi)

    # Zoomed: gray out everything outside the actual loop anchor contact area
    zm_shape = zm_matrices[0].shape
    lp_r_lo, lp_r_hi = _bin_slice(zm_s, zm_e, start0, end0, resolution)
    lp_c_lo, lp_c_hi = _bin_slice(zm_s, zm_e, start1, end1, resolution)
    zm_mask = _gray_mask_outside(zm_shape, lp_r_lo, lp_r_hi, lp_c_lo, lp_c_hi)

    # ── figure ────────────────────────────────────────────────────────────────
    plt.rcParams['font.family'] = 'monospace'
    fig = plt.figure(figsize=(4 * n_cols, 9))
    gs = gridspec.GridSpec(2, n_cols, hspace=0.15, wspace=0.05)

    loop_label = (f"{chr_name}:{start0:,}–{end0:,}  ×  "
                  f"{chr_name}:{start1:,}–{end1:,}")
    fig.suptitle(loop_label, fontsize=10, y=1.01)

    for col, (title, ov_mat, zm_mat) in enumerate(
            zip(titles, ov_matrices, zm_matrices)):

        # row 0: overview
        ax_ov = fig.add_subplot(gs[0, col])
        _draw(ax_ov, ov_mat, ov_mask, ov_s, ov_e, chr_name, vmax,
              f"{title}\nOverview")

        # row 1: zoomed
        ax_zm = fig.add_subplot(gs[1, col])
        _draw(ax_zm, zm_mat, zm_mask, zm_s, zm_e, chr_name, vmax,
              f"{title}\nZoomed")

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    else:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved: {output_file}")

    plt.close(fig)


# ── example ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hic_unphased = "/gpfs0/work/suhas/temp_processing/Juicer2_hg38_reprocess/combined/LCL/GM12878/combined_new_8.7.23/total/inter_30.hic"
    hic_ref      = "/mnt/altnas/work2/suhas/temp_processing/intact_HiC_GRCh38_juicer2/LCLs/diploid_maps/GM12878/total/r.hic"
    hic_alt      = "/mnt/altnas/work2/suhas/temp_processing/intact_HiC_GRCh38_juicer2/LCLs/diploid_maps/GM12878/total/a.hic"

    loop = {
        "chr":    "chr14",
        "start0": 106_565_460,
        "end0":   106_565_670,
        "start1": 106_570_000,
        "end1":   106_571_000,
    }

    with PdfPages("loop_maps.pdf") as pdf:
        plot_loop_maps(
            loop,
            [hic_unphased, hic_ref, hic_alt],
            titles=["Unphased", "Ref", "Alt"],
            resolution=1000,
            overview_pad=50_000,
            zoom_pad=5_000,
            pdf=pdf,
        )

    print("Done.")
