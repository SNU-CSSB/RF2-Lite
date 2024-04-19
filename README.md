# RF2-Lite
GitHub repo for RF2-Lite, a rapid method for screening protein-protein interactions.

## Installation

1. Clone the package
```
git clone https://github.com/SNU-CSSB/RF2-Lite.git
cd RF2-Lite
```

2. Create conda environment
```
# create conda environment for RF2-Lite
conda env create -f RF2Lite-linux.yml
```
You also need to install NVIDIA's SE(3)-Transformer (**please use SE3Transformer in this repo to install**).
```
conda activate RF2Lite
cd SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
```

3. Download pre-trained weights under network directory
```
cd RF2-Lite/networks/
wget http://files.ipd.uw.edu/pub/pathogens/weights.tar.gz
tar xfz weights.tar.gz
```

## Usage
RF2-Lite requires 1) a paired multiple seqeunce alignment and 2) the length of each chain to predict if two or more proteins are interacting. Paired MSAs for RF2-Lite are concatenated alignments with no character deliminating the chain breaks.

See examples in RF2-Lite/example/

For pair inference:
```
python predict_complex.py -msa {paired MSA} -out {output prefix} -L1 {length of first chain}
e.g. python predict_complex.py -msa P77499_P77689.fas -out test -L1 248
```

For batch-run inference:

_Note: Batching will pad to the residue length of the longest example (len(p1) + len(p2)) in the list, increasing compute if there is a large discrepancy between protein lengths in a batch._
```
python RF2-Lite/predict_complex_list.py -list %s -p cuda:0

Where %s is a list file of:

{paired_msa.a3m} {lenp1} {lenp2} {output_prefix}

e.g.
P50599_Q9HYT7.c75i90.a3m 146 102 rf_out/P50599_Q9HYT7
P50599_Q9HZR0.c75i90.a3m 146 102 rf_out/P50599_Q9HZR0
P50599_Q9I0H9.c75i90.a3m 146 102 rf_out/P50599_Q9I0H9
```

## Expected outputs
The user will obtain an npz file containing the following objects: 'dist', 'plddt'.
The 'dist' matrix is of shape length(protein1) x length(protein2).

An interaction score may be extracted from this 'dist' matrix by:
```
int_score = np.max(npz_file['dist'][:-10,10:])
```

## Related work
Computed structures for binary pathogen interactions reported in Essential and virulence-related protein interactions of pathogens revealed through deep learning may be found at [ModelArchives]( https://modelarchive.org/doi/10.5452/ma-bak-evip). Models of higher order complexes may be found on [CongLab](https://conglab.swmed.edu/pathogens/).

This model heavily builds on improvements made in [RoseTTAFold2](https://www.biorxiv.org/content/10.1101/2023.05.24.542179v1).

## References
I.R. Humphreys, J. Zhang, M. Baek, Y. Wang, et al., Essential and virulence-related protein interactions of pathogens revealed through deep learning, BioRxiv (2024). [Link](https://www.biorxiv.org/content/10.1101/2024.04.12.589144v1) 
