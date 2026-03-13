#!/gpfs0/apps/x86/anaconda3/bin/python3
#SBATCH --job-name=anchor_loop_hic
#SBATCH --output=anchor_loop_hic_%j.out
#SBATCH --error=anchor_loop_hic_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=int

import os
import sys
import argparse
import logging

# SLURM copies the script to a spool dir, so __file__ won't point to the
# original location. SLURM_SUBMIT_DIR is the directory sbatch was called from
# and is always set by SLURM; fall back to __file__ for direct python runs.
_script_dir = os.environ.get('SLURM_SUBMIT_DIR', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _script_dir)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plot_hic_region import draw_hic_row

# ── logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


# ── parsing (verbatim from anchor_hic_plots_v4.py) ────────────────────────────

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
                'chr1':   parts[4],
                'start1': int(parts[5]),
                'end1':   int(parts[6]),
            })
    return loops


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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate anchor-centric Hi-C plots (3-panel: Unphased | Ref | Alt). '
                    'Produces one PDF page per loop, organised by belt anchor.\n\n'
                    'All parameters can be supplied via a JSON config file '
                    '(--config config.json). CLI args override config file values.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--config',       metavar='CONFIG.JSON',
                        help='JSON config file; CLI args override values set here.')
    parser.add_argument('--anchor_bed',   default=None,
                        help='BED: CHR START END [AG [CHIPSEQ]]')
    parser.add_argument('--loop_bedpe',   default=None,
                        help='BEDPE: CHR START0 END0 CHR START1 END1 ...')
    parser.add_argument('--unphased_hic', default=None)
    parser.add_argument('--ref_hic',      default=None)
    parser.add_argument('--alt_hic',      default=None)
    parser.add_argument('--output',       default=None)
    parser.add_argument('--resolution',   default=None, type=int,
                        help='Resolution in bp (default: 1000)')
    parser.add_argument('--norm',         default=None,
                        help='Hi-C normalisation (default: NONE)')
    parser.add_argument('--vmax',         default=None, type=float,
                        help='Colour scale max (default: 1.0)')
    parser.add_argument('--overview_pad', default=None, type=int,
                        help='bp padding around loop bounding box for overview panel (default: 5000)')
    parser.add_argument('--zoom_pad',     default=None, type=int,
                        help='bp padding around loop bounding box for zoomed panels (default: 100)')

    # two-pass: config file sets defaults, then CLI overrides
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        import json
        with open(pre_args.config) as fh:
            parser.set_defaults(**json.load(fh))

    args = parser.parse_args()

    # apply built-in defaults for anything still unset
    if args.output       is None: args.output       = 'anchor_loop_hic.pdf'
    if args.resolution   is None: args.resolution   = 1000
    if args.norm         is None: args.norm         = 'NONE'
    if args.vmax         is None: args.vmax         = 1.0
    if args.overview_pad is None: args.overview_pad = 5000
    if args.zoom_pad     is None: args.zoom_pad     = 100

    missing = [n for n, v in [
        ('--anchor_bed',   args.anchor_bed),
        ('--loop_bedpe',   args.loop_bedpe),
        ('--unphased_hic', args.unphased_hic),
        ('--ref_hic',      args.ref_hic),
        ('--alt_hic',      args.alt_hic),
    ] if v is None]
    if missing:
        parser.error("Missing required args (set via CLI or config):\n" + '\n'.join(f'  {m}' for m in missing))

    logger.info(f"Anchor BED:   {args.anchor_bed}")
    logger.info(f"Loop BEDPE:   {args.loop_bedpe}")
    logger.info(f"Unphased HiC: {args.unphased_hic}")
    logger.info(f"Ref HiC:      {args.ref_hic}")
    logger.info(f"Alt HiC:      {args.alt_hic}")
    logger.info(f"Resolution:   {args.resolution} bp")
    logger.info(f"Norm:         {args.norm}")
    logger.info(f"vmax:         {args.vmax}")
    logger.info(f"Overview pad: {args.overview_pad} bp")
    logger.info(f"Zoom pad:     {args.zoom_pad} bp")
    logger.info(f"Output:       {args.output}")

    anchors       = parse_anchor_bed(args.anchor_bed)
    loops         = parse_loop_bedpe(args.loop_bedpe)
    anchor_groups = group_loops_by_anchor(loops, anchors)

    logger.info(f"{len(anchors)} anchor(s), {len(loops)} loop(s) loaded.")

    n_pages = 0

    with PdfPages(args.output) as pdf:
        for anchor in anchors:
            key     = (anchor['chr'], anchor['start'], anchor['end'])
            entries = anchor_groups[key]

            if not entries:
                logger.info(f"  {key[0]}:{key[1]:,}-{key[2]:,} — no loops, skipping.")
                continue

            n = len(entries)
            logger.info(f"  {key[0]}:{key[1]:,}-{key[2]:,} — {n} loop(s).")

            try:
                fig, axes_grid = plt.subplots(n, 3, figsize=(18, 6 * n),
                                              squeeze=False)
                fig.suptitle(
                    f"Anchor: {anchor['chr']}:{anchor['start']:,}–{anchor['end']:,}",
                    fontsize=13, y=1.0
                )

                for row_idx, entry in enumerate(entries):
                    loop = entry['loop']
                    draw_hic_row(
                        args.unphased_hic,
                        args.ref_hic,
                        args.alt_hic,
                        loop['chr0'],
                        loop,
                        axes_grid[row_idx],
                        resolution=args.resolution,
                        norm=args.norm,
                        vmax=args.vmax,
                        overview_pad=args.overview_pad,
                        zoom_pad=args.zoom_pad,
                        title0="Unphased",
                        title1="Ref",
                        title2="Alt",
                    )

                plt.tight_layout()
                pdf.savefig(fig)
                n_pages += 1

            except Exception:
                logger.exception(
                    f"  FAILED — {key[0]}:{key[1]:,}-{key[2]:,} — skipping anchor."
                )
            finally:
                plt.close('all')

    logger.info(f"Done. {n_pages} page(s) written to {args.output}")


if __name__ == '__main__':
    main()
