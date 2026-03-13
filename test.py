import hicstraw

files = {
    "unphased": "/gpfs0/work/suhas/temp_processing/Juicer2_hg38_reprocess/combined/LCL/GM12878/combined_new_8.7.23/total/inter_30.hic",
    "ref":      "/mnt/altnas/work2/suhas/temp_processing/intact_HiC_GRCh38_juicer2/LCLs/diploid_maps/GM12878/total/r.hic",
    "alt":      "/mnt/altnas/work2/suhas/temp_processing/intact_HiC_GRCh38_juicer2/LCLs/diploid_maps/GM12878/total/a.hic",
}

for name, path in files.items():
    print(f"\nTesting {name}: {path}", flush=True)
    hf = hicstraw.HiCFile(path)
    print(f"  chromosomes: {[c.name for c in hf.getChromosomes()]}", flush=True)
    mzd = hf.getMatrixZoomData("chr1", "chr1", "observed", "NONE", "BP", 1000)
    print(f"  got zoom data", flush=True)
    mat = mzd.getRecordsAsMatrix(1005000, 1026000, 1005000, 1026000)
    print(f"  matrix shape: {mat.shape}", flush=True)