import hicstraw
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

# ─────────────────────────────────────────────────────────────
# v4 color scheme (white → red, tanh/MAD scaling)
# ─────────────────────────────────────────────────────────────
REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])


def preprocess(matrix, scale=0.1, epsilon=1e-9):
    '''
    tanh/MAD contrast scaling (verbatim from anchor_hic_plots_v4.py).
    Returns a matrix in roughly [-1, 1] suitable for REDMAP display.
    '''
    if np.abs(np.sum(matrix)) < epsilon:
        return matrix
    flat = matrix.flatten()
    flat = flat[flat > 0]
    if len(flat) == 0:
        return matrix
    median = np.median(flat)
    mad    = np.median(np.abs(flat - median))
    if mad == 0:
        mad = epsilon
    return np.tanh(scale * (matrix - median) / mad)


def compute_anchor_sum(matrix, bin_r0, bin_r1, bin_c0, bin_c1):
    '''
    Returns the sum of raw contacts within the loop anchor region.

    bin_r0/r1 — row (y-axis) bin range for anchor-0 start/end
    bin_c0/c1 — col (x-axis) bin range for anchor-1 start/end

    Uses floor/ceil so anchors smaller than one resolution bin still
    contribute at least 1 bin on each axis (avoids returning 0 for tiny
    anchors).  Values are clamped to the matrix bounds.
    '''
    ri0 = max(0,               int(np.floor(bin_r0)))
    ri1 = min(matrix.shape[0], max(ri0 + 1, int(np.ceil(bin_r1))))
    ci0 = max(0,               int(np.floor(bin_c0)))
    ci1 = min(matrix.shape[1], max(ci0 + 1, int(np.ceil(bin_c1))))
    return float(matrix[ri0:ri1, ci0:ci1].sum())


def _build_header(anchor, page_num=None, total_pages=None,
                  anc_ref=None, anc_alt=None):
    '''Returns (anchor_bold, metrics_normal, line2_normal).

    anchor_bold   : "Anchor: chrX:start–end"         → rendered bold
    metrics_normal: "   anc_REF: X   anc_ALT: X"     → normal, same line
    line2_normal  : "AlphaGenome pred: X  |  ..."     → normal, second line
    '''
    anchor_bold = f"Anchor: {anchor['chr']}:{anchor['start']:,}–{anchor['end']:,}"

    if anc_ref is not None and anc_alt is not None:
        metrics_normal = f"   anc_REF: {anc_ref:,.0f}   anc_ALT: {anc_alt:,.0f}"
    else:
        metrics_normal = ""

    parts2 = []
    if anchor.get('ag') is not None:
        parts2.append(f"AlphaGenome pred: {anchor['ag']:.3g}")
    if anchor.get('chipseq') is not None:
        parts2.append(f"empiric ChIP-seq: {anchor['chipseq']:.3g}")
    if anc_ref is not None and anc_alt is not None:
        if anc_ref > 0 and anc_alt > 0:
            parts2.append(f"Hi-C: {np.log2(anc_ref / anc_alt):+.2f}")
        elif anc_ref == 0 and anc_alt == 0:
            parts2.append("Hi-C: N/A")
        elif anc_alt == 0:
            parts2.append("Hi-C: +\u221e")
        else:
            parts2.append("Hi-C: \u2212\u221e")
    if page_num is not None and total_pages is not None and total_pages > 1:
        parts2.append(f"page {page_num} of {total_pages}")

    return anchor_bold, metrics_normal, "   |   ".join(parts2)


def _render_header(fig, fig_height, anchor_bold, metrics_normal, line2_normal):
    '''Renders the two-line page header with mixed bold/normal on line 1.

    Bold anchor label is right-aligned at x=0.5; normal metrics are
    left-aligned at x=0.5 — visually centered around the midpoint.
    '''
    y1 = 1.0 - 0.10 / fig_height   # ~0.10 in from top
    y2 = 1.0 - 0.28 / fig_height   # ~0.28 in from top (~0.18 in below line 1)

    fig.text(0.5, y1, anchor_bold,
             ha='right', va='top', fontsize=9, fontweight='bold',
             transform=fig.transFigure)
    if metrics_normal:
        fig.text(0.5, y1, metrics_normal,
                 ha='left', va='top', fontsize=9,
                 transform=fig.transFigure)
    if line2_normal:
        fig.text(0.5, y2, line2_normal,
                 ha='center', va='top', fontsize=8.5,
                 transform=fig.transFigure)


def render_anchor_header(fig, fig_height, anchor, anc_ref=None, anc_alt=None,
                         page_num=None, total_pages=None):
    '''Convenience wrapper: builds and renders the per-page anchor header.'''
    _render_header(fig, fig_height,
                   *_build_header(anchor, page_num, total_pages,
                                  anc_ref=anc_ref, anc_alt=anc_alt))


# ─────────────────────────────────────────────────────────────
# Cache objects so we don't reopen the hic file repeatedly
# ─────────────────────────────────────────────────────────────
_hic_file_cache = {}   # path -> HiCFile


def get_matrix_object(hic_path, chr_name, resolution, norm):
    if hic_path not in _hic_file_cache:
        _hic_file_cache[hic_path] = hicstraw.HiCFile(hic_path)

    hf = _hic_file_cache[hic_path]

    return hf.getMatrixZoomData(
        chr_name,
        chr_name,
        "observed",
        norm,
        "BP",
        resolution
    )


# ─────────────────────────────────────────────────────────────
# Helper: fetch and trim matrix for one hic file
# ─────────────────────────────────────────────────────────────
def _get_matrix(hic_path, chr_name, start0, end0, start1, end1, resolution, norm):
    print(f"Getting matrix for {chr_name} {start0, end0, start1, end1}", flush=True)
    matrix_obj = get_matrix_object(hic_path, chr_name, resolution, norm)
    matrix = matrix_obj.getRecordsAsMatrix(start0, end0, start1, end1)
    n_bins_x = math.ceil((end0 - start0) / resolution)
    n_bins_y = math.ceil((end1 - start1) / resolution)
    return matrix[:n_bins_x, :n_bins_y]


# ─────────────────────────────────────────────────────────────
# Helper: draw one panel onto an existing Axes
# ─────────────────────────────────────────────────────────────
def _draw_panel(ax, matrix, chr_name, start0, end0, start1, end1,
                resolution, vmin, vmax, cmap, loop, title,
                box_color="cyan", box_lw=2):

    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest"
    )

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel(f"{chr_name}:{start1:,}–{end1:,}", fontsize=8)
    ax.set_ylabel(f"{chr_name}:{start0:,}–{end0:,}", fontsize=8)
    ax.tick_params(labelsize=7)

    plt.colorbar(im, ax=ax, label="Contact count", fraction=0.046, pad=0.04)

    if loop is not None:
        x = (loop["start1"] - start1) / resolution
        y = (loop["start0"] - start0) / resolution
        w = (loop["end1"] - loop["start1"]) / resolution
        h = (loop["end0"] - loop["start0"]) / resolution
        ax.add_patch(Rectangle(
            (x, y), w, h,
            fill=False, edgecolor=box_color, linewidth=box_lw
        ))


# ─────────────────────────────────────────────────────────────
# Draw one loop row onto three provided Axes objects.
# Used by anchor_loop_hic.py to build multi-row per-anchor pages.
# ─────────────────────────────────────────────────────────────
def draw_hic_row(
    hic_overview,
    hic_path1,
    hic_path2,
    chr_name,
    loop,
    axes,                   # sequence of exactly 3 Axes
    resolution=1000,
    norm="NONE",
    vmax=1,                 # used when use_preprocess=False
    cmap="Reds",            # used when use_preprocess=False
    overview_pad=5000,
    zoom_pad=500,
    title0="Overview",
    title1="Sample 1",
    title2="Sample 2",
    use_preprocess=False,   # True → REDMAP + tanh/MAD (v4 style)
    overview_scale=0.1,     # tanh scale for overview panel (use_preprocess only)
    zoom_scale=1.0,         # tanh scale for zoom panels   (use_preprocess only)
    row_idx=0,              # 0-based row index used for the row label
    label_ax=None,          # if given, row label is rendered there; else floats on axes[0]
):
    lo   = min(loop["start0"], loop["start1"])
    hi   = max(loop["end0"],   loop["end1"])
    mid0 = (loop["start0"] + loop["end0"]) // 2
    mid1 = (loop["start1"] + loop["end1"]) // 2

    ov_s = ((lo - overview_pad) // resolution) * resolution
    ov_e = math.ceil((hi + overview_pad) / resolution) * resolution

    zm_s = ((lo - zoom_pad) // resolution) * resolution
    zm_e = math.ceil((hi + zoom_pad) / resolution) * resolution

    print(f"OV region: {chr_name}:{ov_s}-{ov_e}", flush=True)
    print(f"ZM region: {chr_name}:{zm_s}-{zm_e}", flush=True)
    mat_ov = _get_matrix(hic_overview, chr_name, ov_s, ov_e, ov_s, ov_e, resolution, norm)
    mat_z1 = _get_matrix(hic_path1,    chr_name, zm_s, zm_e, zm_s, zm_e, resolution, norm)
    mat_z2 = _get_matrix(hic_path2,    chr_name, zm_s, zm_e, zm_s, zm_e, resolution, norm)

    # ── Σ contact sums within the loop anchor region ──────────────────────────
    sum_ov = compute_anchor_sum(mat_ov,
                 (loop['start0'] - ov_s) / resolution,
                 (loop['end0']   - ov_s) / resolution,
                 (loop['start1'] - ov_s) / resolution,
                 (loop['end1']   - ov_s) / resolution)
    sum_z1 = compute_anchor_sum(mat_z1,
                 (loop['start0'] - zm_s) / resolution,
                 (loop['end0']   - zm_s) / resolution,
                 (loop['start1'] - zm_s) / resolution,
                 (loop['end1']   - zm_s) / resolution)
    sum_z2 = compute_anchor_sum(mat_z2,
                 (loop['start0'] - zm_s) / resolution,
                 (loop['end0']   - zm_s) / resolution,
                 (loop['start1'] - zm_s) / resolution,
                 (loop['end1']   - zm_s) / resolution)

    # ── allelic metrics ───────────────────────────────────────────────────────
    if sum_z1 > 0 and sum_z2 > 0:
        ratio       = sum_z1 / sum_z2
        log2fc      = np.log2(ratio)
        allelic_str = f"R/A: {ratio:.2f}\u00d7   log2FC: {log2fc:+.2f}"
    elif sum_z1 == 0 and sum_z2 == 0:
        allelic_str = "R/A: N/A   log2FC: N/A"
    elif sum_z2 == 0:
        allelic_str = "R/A: \u221e   log2FC: +\u221e"
    else:
        allelic_str = "R/A: 0   log2FC: \u2212\u221e"

    # ── row label text (rendered on axes[0] left margin after drawing) ────────
    row_label = (
        f"Loop {row_idx + 1}\n"
        f"{chr_name}:{loop['start0']:,}-{loop['end0']:,}\n"
        f"\u00d7 {chr_name}:{loop['start1']:,}-{loop['end1']:,}\n"
        f"dist: {abs(mid1 - mid0) / 1000:.0f} kb\n"
        f"{allelic_str}"
    )

    if use_preprocess:
        # v4 style: tanh/MAD scaling, REDMAP, shared ref/alt scale, dodgerblue box
        proc_ov = preprocess(mat_ov, scale=overview_scale)
        proc_z1 = preprocess(mat_z1, scale=zoom_scale)
        proc_z2 = preprocess(mat_z2, scale=zoom_scale)
        ov_vmin, ov_vmax = proc_ov.min(), proc_ov.max()
        zm_vmin = min(proc_z1.min(), proc_z2.min())
        zm_vmax = max(proc_z1.max(), proc_z2.max())
        _draw_panel(axes[0], proc_ov, chr_name, ov_s, ov_e, ov_s, ov_e,
                    resolution, ov_vmin, ov_vmax, REDMAP, loop,
                    f"{title0}  \u03a3={sum_ov:,.0f}",
                    box_color="dodgerblue", box_lw=0.8)
        _draw_panel(axes[1], proc_z1, chr_name, zm_s, zm_e, zm_s, zm_e,
                    resolution, zm_vmin, zm_vmax, REDMAP, loop,
                    f"{title1}  \u03a3={sum_z1:,.0f}",
                    box_color="dodgerblue", box_lw=0.8)
        _draw_panel(axes[2], proc_z2, chr_name, zm_s, zm_e, zm_s, zm_e,
                    resolution, zm_vmin, zm_vmax, REDMAP, loop,
                    f"{title2}  \u03a3={sum_z2:,.0f}",
                    box_color="dodgerblue", box_lw=0.8)
    else:
        # original style: raw vmax, user-supplied cmap, cyan box
        _draw_panel(axes[0], mat_ov, chr_name, ov_s, ov_e, ov_s, ov_e,
                    resolution, 0, vmax, cmap, loop,
                    f"{title0}  \u03a3={sum_ov:,.0f}")
        _draw_panel(axes[1], mat_z1, chr_name, zm_s, zm_e, zm_s, zm_e,
                    resolution, 0, vmax, cmap, loop,
                    f"{title1}  \u03a3={sum_z1:,.0f}")
        _draw_panel(axes[2], mat_z2, chr_name, zm_s, zm_e, zm_s, zm_e,
                    resolution, 0, vmax, cmap, loop,
                    f"{title2}  \u03a3={sum_z2:,.0f}")

    # ── row label: dedicated axis column (preferred) or left margin fallback ──
    _label_target = label_ax if label_ax is not None else axes[0]
    if label_ax is not None:
        label_ax.axis('off')
        x_pos, y_pos, clip = 0.5, 0.5, False
    else:
        x_pos, y_pos, clip = -0.05, 0.5, False
    _label_target.text(
        x_pos, y_pos, row_label,
        transform=_label_target.transAxes,
        fontsize=6.5, ha='center', va='center',
        rotation=90, rotation_mode='anchor',
        fontfamily='monospace', clip_on=clip,
    )

    # ── view/loop xlabel on first panel (top edge) ────────────────────────────
    axes[0].set_xlabel(
        f"View: {chr_name}:{ov_s:,}\u2013{ov_e:,}"
        f"\nLoop: {chr_name}:{loop['start1']:,}\u2013{loop['end1']:,}",
        fontsize=6, color='black',
    )
    axes[0].xaxis.set_label_position('top')

    return sum_ov, sum_z1, sum_z2


# ─────────────────────────────────────────────────────────────
# Main plotting function
# Panel 0: hic_overview  — wide symmetric region around the loop
# Panel 1: hic_path1     — zoomed in on loop anchors
# Panel 2: hic_path2     — zoomed in on loop anchors
# ─────────────────────────────────────────────────────────────
def plot_hic_region(
    hic_overview,
    hic_path1,
    hic_path2,
    chr_name,
    resolution=1000,
    norm="NONE",
    vmax=1,
    cmap="Reds",
    loop=None,
    overview_pad=5000,              # bp padding around loop bounding box for panel 0
    zoom_pad=500,                   # bp padding around loop bounding box for panels 1+2
    title0="Overview",
    title1="Sample 1",
    title2="Sample 2",
    pdf=None,                       # PdfPages object; if None, saves to output_file
    output_file="hic_plot.pdf"
):

    # ── derive both regions from loop bounding box + respective padding
    if loop is not None:
        lo = min(loop["start0"], loop["start1"])
        hi = max(loop["end0"],   loop["end1"])

        ov_s = ((lo - overview_pad) // resolution) * resolution
        ov_e = math.ceil((hi + overview_pad) / resolution) * resolution

        zm_s = ((lo - zoom_pad) // resolution) * resolution
        zm_e = math.ceil((hi + zoom_pad) / resolution) * resolution

        loop_label = (f"{chr_name}:{loop['start0']:,}–{loop['end0']:,}"
                      f"  ×  {chr_name}:{loop['start1']:,}–{loop['end1']:,}")
    else:
        raise ValueError("loop must be provided")

    print(f"OV region: {chr_name}:{ov_s}-{ov_e}", flush=True)
    print(f"ZM region: {chr_name}:{zm_s}-{zm_e}", flush=True)
    mat_ov = _get_matrix(hic_overview, chr_name, ov_s, ov_e, ov_s, ov_e, resolution, norm)
    mat_z1 = _get_matrix(hic_path1,    chr_name, zm_s, zm_e, zm_s, zm_e, resolution, norm)
    mat_z2 = _get_matrix(hic_path2,    chr_name, zm_s, zm_e, zm_s, zm_e, resolution, norm)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    _draw_panel(axes[0], mat_ov, chr_name, ov_s, ov_e, ov_s, ov_e,
                resolution, 0, vmax, cmap, loop, title0)
    _draw_panel(axes[1], mat_z1, chr_name, zm_s, zm_e, zm_s, zm_e,
                resolution, 0, vmax, cmap, loop,
                f"{title1}\n{loop_label}" if loop_label else title1)
    _draw_panel(axes[2], mat_z2, chr_name, zm_s, zm_e, zm_s, zm_e,
                resolution, 0, vmax, cmap, loop,
                f"{title2}\n{loop_label}" if loop_label else title2)

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig)
    else:
        plt.savefig(output_file)
        print("Saved:", output_file)

    plt.close()


# ─────────────────────────────────────────────────────────────
# Example usage (your Juicebox region)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    hic_overview = "/gpfs0/work/suhas/temp_processing/Juicer2_hg38_reprocess/combined/LCL/GM12878/combined_new_8.7.23/total/inter_30.hic"
    hic_file1 = "/mnt/altnas/work2/suhas/temp_processing/intact_HiC_GRCh38_juicer2/LCLs/diploid_maps/GM12878/total/r.hic"
    hic_file2 = "/mnt/altnas/work2/suhas/temp_processing/intact_HiC_GRCh38_juicer2/LCLs/diploid_maps/GM12878/total/a.hic"

    loops = [
        {"chr": "chr14", "start0": 106565460, "end0": 106565670, "start1": 106564000, "end1": 106566000},
        {"chr": "chr14", "start0": 106564200, "end0": 106564275, "start1": 106564000, "end1": 106566000},
        # add more loops here...
    ]

    output_file = "hic_loops.pdf"

    with PdfPages(output_file) as pdf:
        for loop in loops:
            plot_hic_region(
                hic_overview,
                hic_file1,
                hic_file2,
                loop["chr"],
                resolution=10,
                norm="NONE",
                vmax=1,
                loop=loop,
                overview_pad=5000,
                zoom_pad=100,
                title0="Unphased",
                title1="Ref",
                title2="Alt",
                pdf=pdf
            )

    print("Saved:", output_file)
