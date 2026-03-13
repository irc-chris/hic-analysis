import hicstraw
import math

files = {
    "unphased": "/gpfs0/work/suhas/temp_processing/Juicer2_hg38_reprocess/combined/LCL/GM12878/combined_new_8.7.23/total/inter_30.hic",
    "ref":      "/mnt/altnas/work2/suhas/temp_processing/intact_HiC_GRCh38_juicer2/LCLs/diploid_maps/GM12878/total/r.hic",
    "alt":      "/mnt/altnas/work2/suhas/temp_processing/intact_HiC_GRCh38_juicer2/LCLs/diploid_maps/GM12878/total/a.hic",
}

# simulate the first anchor's loops
regions = [
    ("chr1", 1005000, 1026000),
    ("chr1", 1009900, 1021100),
]

hic_files = {name: hicstraw.HiCFile(path) for name, path in files.items()}

for i, (chrom, start, end) in enumerate(regions):
    for name, hf in hic_files.items():
        print(f"Loop {i} | {name} | {chrom}:{start}-{end}", flush=True)
        mzd = hf.getMatrixZoomData(chrom, chrom, "observed", "NONE", "BP", 1000)
        print(f"  got zoom data", flush=True)
        mat = mzd.getRecordsAsMatrix(start, end, start, end)
        print(f"  matrix shape: {mat.shape}", flush=True)

print("Done", flush=True)