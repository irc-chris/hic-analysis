import hicstraw
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages


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
                resolution, vmax, cmap, loop, title):

    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=0,
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
            fill=False, edgecolor="cyan", linewidth=2
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
    axes,               # sequence of exactly 3 Axes
    resolution=1000,
    norm="NONE",
    vmax=1,
    cmap="Reds",
    overview_pad=5000,
    zoom_pad=500,
    title0="Overview",
    title1="Sample 1",
    title2="Sample 2",
):
    lo = min(loop["start0"], loop["start1"])
    hi = max(loop["end0"],   loop["end1"])

    ov_s = ((lo - overview_pad) // resolution) * resolution
    ov_e = math.ceil((hi + overview_pad) / resolution) * resolution

    zm_s = ((lo - zoom_pad) // resolution) * resolution
    zm_e = math.ceil((hi + zoom_pad) / resolution) * resolution

    loop_label = (f"{chr_name}:{loop['start0']:,}–{loop['end0']:,}"
                  f"  ×  {chr_name}:{loop['start1']:,}–{loop['end1']:,}")

    print(f"OV region: {chr_name}:{ov_s}-{ov_e}", flush=True)
    print(f"ZM region: {chr_name}:{zm_s}-{zm_e}", flush=True)
    mat_ov = _get_matrix(hic_overview, chr_name, ov_s, ov_e, ov_s, ov_e, resolution, norm)
    mat_z1 = _get_matrix(hic_path1,    chr_name, zm_s, zm_e, zm_s, zm_e, resolution, norm)
    mat_z2 = _get_matrix(hic_path2,    chr_name, zm_s, zm_e, zm_s, zm_e, resolution, norm)

    _draw_panel(axes[0], mat_ov, chr_name, ov_s, ov_e, ov_s, ov_e,
                resolution, vmax, cmap, loop, title0)
    _draw_panel(axes[1], mat_z1, chr_name, zm_s, zm_e, zm_s, zm_e,
                resolution, vmax, cmap, loop, f"{title1}\n{loop_label}")
    _draw_panel(axes[2], mat_z2, chr_name, zm_s, zm_e, zm_s, zm_e,
                resolution, vmax, cmap, loop, f"{title2}\n{loop_label}")


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
                resolution, vmax, cmap, loop, title0)
    _draw_panel(axes[1], mat_z1, chr_name, zm_s, zm_e, zm_s, zm_e,
                resolution, vmax, cmap, loop,
                f"{title1}\n{loop_label}" if loop_label else title1)
    _draw_panel(axes[2], mat_z2, chr_name, zm_s, zm_e, zm_s, zm_e,
                resolution, vmax, cmap, loop,
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
