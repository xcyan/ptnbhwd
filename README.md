# Perspective Transformer Layer

## Build
To build the ptn libriary, run the following script.
```
luarocks make ptnbhwd-scm-1.rockspec
```

## Main modules

This is the torch implementation of the [Perspective Transformer Layer](https://papers.nips.cc/paper/6206-perspective-transformer-nets-learning-single-view-3d-object-reconstruction-without-3d-supervision.pdf), which is built on top of the [STN torch implementation](https://github.com/qassemoquab/stnbhwd).

``` lua
require 'ptn'

nn.PerspectiveGridGenerator(depth, height, width, focal_length)
-- takes B x 4 x 4 affine transform matrices as input, 
-- outputs a depth x height x width grid in normalized [dmin,dmax] x [-1,1] x [-1,1] coordinates, where dmin and dmax represent the minimal and maximal disparity.

nn.BilinearSamplerPerspective()
-- takes a table {inputImages, grids} as inputs
-- outputs the interpolated images according to the grids
-- inputImages is a batch of samples in BHWD layout
-- grids is a batch of grids (output of PerspectiveGridGenerator)
-- output is also BHWD
```

## Citation
If you find this useful, please cite our work as follows:
```
@incollection{NIPS2016_6206,
title = {Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction without 3D Supervision},
author = {Yan, Xinchen and Yang, Jimei and Yumer, Ersin and Guo, Yijie and Lee, Honglak},
booktitle = {Advances in Neural Information Processing Systems 29},
editor = {D. D. Lee and M. Sugiyama and U. V. Luxburg and I. Guyon and R. Garnett},
pages = {1696--1704},
year = {2016},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/6206-perspective-transformer-nets-learning-single-view-3d-object-reconstruction-without-3d-supervision.pdf}
}
```


