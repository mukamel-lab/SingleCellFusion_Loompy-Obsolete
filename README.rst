SingleCellFusion
================

SingleCellFusion is a package for computational integration and analysis of single cell multiomics data sets, including
transcriptom (RNA-Seq), DNA methylome (mC-Seq), and chromatin accessibility (ATAC-Seq). For a given pair of data sets,
SingleCellFusion finds the best matching pairs of cells in each modality (i.e. nearest neighbors) by taking advantage of
the correlation of gene expression with epigenomic marks across the gene body. Neighbors are used to impute counts for
each data set. For example, if integrating scRNA-seq and snATAC-seq cells, SingleCellFusion will generate imputed
scRNA-seq counts for the snATAC-seq profiled cells and snATAC-seq counts for the scRNA-seq profiled cells.
Cells profiled by each technique can then be analyzed together in a joint, lower dimensional space.


The package is still under active development and function parameters will continue to change over time. A changelog of
SingleCellFusion's development can be viewed
`here <docs/changelog.rst>`_.


Requirements
------------
* python 3
* loompy
* numpy
* scikit-learn
* scipy
* pandas
* numba
* fbpca


Installation
------------
Currently, the only method of installing SingleCellFusion is to clone the github repository.
Enter the directory where you would like to install SingleCellFusion and enter
the following commands on the command line::

    git clone https://github.com/mukamel-lab/SingleCellFusion.git
    cd SingleCellFusion
    python setup.py install

If you have trouble with dependencies, we have a
`guide <https://github.com/mukamel-lab/mop/blob/master/docs/mop_conda_guide.rst>`_
to generating a usable conda environment in SingleCellFusion's sister repository
`MoP <https://github.com/mukamel-lab/mop/>`_.

Basic Usage
-----------
This will be uploaded soon!

Tutorials and FAQs
-------------------
Our `FAQs <docs/faqs.rst>`_ have some basic information on running SingleCellFusion.

For a brief description of how SingleCellFusion works please check out this
`link <docs/scf_description.rst>`_.

If you need information on performing preliminary analyses on loom files, please check out
SingleCellFusion's sister repository `MoP <https://github.com/mukamel-lab/mop/>`_.


Authors
-------

`SingleCellFusion` was written by `Wayne Doyle <widoyle@ucsd.edu>`_,
`Fangming Xie <f7xie@ucsd.edu>`_, `Ethan Armand <earmand@ucsd.edu>`_.
All authors are members of the `Mukamel Lab <https://brainome.ucsd.edu>`_.


Acknowledgments
---------------
We are grateful for support from the Chan-Zuckerberg Initiative program on `Collaborative computational tools for the human cell atlas <https://grants.czi.technology/>`_ (grant 183111) and from the NIH
BRAIN Initiative U19 Center for Epigenomics of the Mouse Brain Atlas
(`CEMBA <https://biccn.org/teams/u19-ecker/>`_).
