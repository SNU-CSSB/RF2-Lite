import os
import glob

MSA_dir = "/projects/omics/yeast/test_new_RF/pair_msas"
negative_fn = "/projects/omics/yeast/test_new_RF/all_inputs"

wrt = ''
with open(negative_fn) as fp:
    for line in fp:
        x = line.split()
        fn = glob.glob("%s/%s_*.a3m"%(MSA_dir, x[0]))[0]
        wrt += "%s %d %d outs/%s\n"%(fn, int(x[1]), int(x[2]), x[0])

with open("negative.inputs", 'wt') as fp:
    fp.write(wrt)

