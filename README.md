# Stochastic Gauss-Newton Methods


## Introduction

This package is the implementation of "Stochastic Gauss-Newton Algorithms for Nonconvex Compositional Optimization" including 3 different variants to solve the class of stochastic nonconvex compositional optimization problems. The package includes two numerical examples on compositional optimization problems as presented in the paper. The code is tested using Python 3.7.1.

## How to Run

- First we need the following dependency

`
pip install pandas sklearn matplotlib
`

- Then we need to download the datasets. Assuming that we are at ```SGN_Code``` folder, now we go to data folder and execute the ```get_data.sh``` script
`
cd data
sh get_data.sh
`
or download the ```w8a``` and ```covtype``` datasets at [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) and place them in the ```SGN_Code/data``` folder.

- For ```covtype``` dataset, we need to decompress the file and rename it from ```covtype.bz2``` to ```covtype``` for consistency.

The notebooks to run each example are place in ```notebook``` folder. Please refer to them for more details.

## Code Usage

We hope that this program will be useful to others, and we would like to hear about your experience with it. If you found it helpful and are using it within our software please cite the following publication:

* Q. Tran-Dinh, N. H. Pham, and L. M. Nguyen, **[Stochastic Gauss-Newton Algorithms for Nonconvex Compositional Optimization](https://arxiv.org/abs/2002.07290)**, _arXiv preprint arXiv:2002.07290_, 2020.
