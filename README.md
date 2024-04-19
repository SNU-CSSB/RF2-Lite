# RF2-Lite
GitHub repo for RF2-Lite

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
