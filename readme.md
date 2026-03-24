# Vineside - a microscopy dataset of anatomy of grapevine wood over 6 months
*Authors: Célia Perrin, Jean-Baptiste Courbot, Yann Leva, and Romain Pierron.*

This repository is part of the **Vineside** project:
- The database is available on [Zenodo](https://zenodo.org/records/18850060)
- The corresponding preprint, describing the experimental conditions and results, is online on [BioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.20.713180v1).
- The code for reproducing the numerical experiments in the paper is available on this repository.

## Code description

This Github repository provides the Python code running the image processing tasks described in the paper. The repository contains three Jupyter notebooks, that should be the entry point to run the code. The repository also contains:
- `segmentation_tools.py` that implement all of the image processing techniques we used.
- an `input images` folder containing a few test images that are used within the notebooks. Other images are available in our 4771-images database on Zenodo.
- `mobile_sam.pt` contains the weights used to run MobileSAM as part of the segmentation.

It requires standard Python packages, as well as:
- Tifffile for reading .tif images (https://pypi.org/project/tifffile/).
- MobileSAM as provided at https://github.com/ChaoningZhang/MobileSAM

The code was written using the following package versions:
- Matplotlib: 3.10.6
- Numpy: 2.3.4
- Pandas: 2.3.3
- Python: 3.12.12
- Scipy: 1.16.3
- Tifffile: 2024.12.12



