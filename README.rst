SingleCellFusion_Loompy
================

SingleCellFusion is a package for computational integration and analysis of single cell multiomics data sets, including
transcriptom (RNA-Seq), DNA methylome (mC-Seq), and chromatin accessibility (ATAC-Seq). For a given pair of data sets,
SingleCellFusion finds the best matching pairs of cells in each modality (i.e. nearest neighbors) by taking advantage of
the correlation of gene expression with epigenomic marks across the gene body. Neighbors are used to impute counts for
each data set. For example, if integrating scRNA-seq and snATAC-seq cells, SingleCellFusion_Loompy will generate imputed
scRNA-seq counts for the snATAC-seq profiled cells and snATAC-seq counts for the scRNA-seq profiled cells.
Cells profiled by each technique can then be analyzed together in a joint, lower dimensional space.


The package is the implementation of SingleCellFusion based on `Loompy <http://loompy.org/>`_. It is still under development and function parameters will continue to change over time. A changelog of
SingleCellFusion_Loompy's development can be viewed
`here <docs/changelog.rst>`_. For a lite implementation of this computational tool, please go to `SingleCellFusion <https://github.com/mukamel-lab/SingleCellFusion/>`_.

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
Currently, the only method of installing SingleCellFusion_Loompy is to clone the github repository.
Enter the directory where you would like to install SingleCellFusion_Loompy and enter
the following commands on the command line::

    git clone https://github.com/mukamel-lab/SingleCellFusion_Loompy.git
    cd SingleCellFusion_Loompy
    python setup.py install

If you have trouble with dependencies, we have a
`guide <https://github.com/mukamel-lab/mop/blob/master/docs/mop_conda_guide.rst>`_
to generating a usable conda environment in SingleCellFusion_Loompy's sister repository
`MoP <https://github.com/mukamel-lab/mop/>`_.

Basic Usage
-----------
The simplest way to use SingleCellFusion_Loompy is to use the function fuse_data. As an example, say you have three datasets:
an RNA-seq dataset stored in loom_rna, a methylome-seq dataset stored in loom_mc, and an ATAC-seq dataset stored in
loom_atac. If your goal is to get imputed RNA data for the methylome and chromatin accessibility data, then
fuse_data can be run with the following commands in a Python script::

    import singlecellfusion as scf
    scf.fuse_data(loom_source=loom_rna, #Path to the RNA loom file, observed data for imputation
              loom_target=[loom_mc,loom_atac], #Paths to methylome and ATAC files, will receive imputed counts
              correlation=['negative','positive'], #Expected correlation to RNA, methylome is negative and ATAC is positive
              loom_output=loom_output, #Path to output loom file containing observed/imputed RNA counts
              layer_source='', #Layer in loom_rna containing observed RNA counts
              layer_target=['',''], #Layers in loom_mc and loom_atac containing observed data
              layer_impute = 'imputed_rna', #Layers added to loom_target, will contain imputed RNA counts
              feat_source='Accession', #Row attribute specifying unique feature IDs in loom_rna
              feat_target = ['Accession','Accession'], #Row attributes specifying unique feature IDs in loom_mc and loom_atac
              cell_source = 'CellID', #Column attribute specifying unique cell IDs in loom_rna
              cell_target = ['CellID','CellID'], #Column attributes specifying unique cell IDs in loom_mc and loom_atac
              cluster_source = 'ClusterID', #Column attribute containing cluster assignments for cells in loom_rna
              cluster_target = ['ClusterID','ClusterID'], #Column attributes containing cluster assignments for loom_mc and loom_atac
              valid_ra_source=None, #Optional, row attribute of 0s and 1s specifying features that should be used
              valid_ra_target=[None,None], #Optional, valid features for loom_mc and loom_atac
              valid_ca_source=None, #Optional, column attribute of 0s and 1s specifying cells that should be used
              valid_ca_target=[None,None], #Optional, valid cells for loom_mc and loom_atac
              label_source='scRNA-seq', #Optional, labels for cells from loom_rna
              label_target=['snmC-seq','snATAC-seq'], #Optional, labels for loom_mc and loom_atac
              var_method='kruskal', #Method for determining variable features for integration, Kruskal-Wallis is recommended
              kruskal_n=8000, #Number of variable features to find and intersect per modality
              neighbor_method='knn', #Method for finding neighbors, knn is recommended
              n_neighbors=20, #Number of neighbors to find between each modality
              relaxation=10, #Flexibility parameter, a higher number enables more connections per cell
              remove_version=False, #If true, removes GENCODE version IDs. Useful if datasets have different annotations
              low_mem=False, #If true, runs in a low memory fashion. Will be slow but can handle large datasets
              batch_size=5000, #Size of batches if low_mem is True. Larger sizes go faster but with more memory
              verbose=True) #Print logging messages


Tutorials and FAQs
-------------------
* An `example walkthrough <docs/snmc2t_example.rst>`_ of SingleCellFusion_Loompy using data from this `bioRxiv preprint <https://doi.org/10.1101/434845>`_.
* For a brief description of how SingleCellFusion_Loompy works please check out this `link <docs/scf_description.rst>`_.
* Our `FAQs <docs/faqs.rst>`_ have some answers to common questions that come up while using SingleCellFusion_Loompy.
* If you need information on performing preliminary analyses on loom files, please check out SingleCellFusion_Loompy's sister repository `MoP <https://github.com/mukamel-lab/mop/>`_.


Authors
-------

`SingleCellFusion_Loompy` was developed by the `Mukamel Lab <https://brainome.ucsd.edu>`_.
Code was specifically written by `Wayne Doyle <https://github.com/wayneidoyle>`_, `Fangming Xie <f7xie@ucsd.edu>`_,
and `Ethan Armand <earmand@ucsd.edu>`_.

Further description and citation
--------------------------------

- `Luo, C. et al. Single nucleus multi-omics links human cortical cell regulatory genome diversity to disease risk variants. bioRxiv 2019.12.11.873398 (2019) doi:10.1101/2019.12.11.873398 <https://www.biorxiv.org/content/10.1101/2019.12.11.873398v1>`_

- `Yao, Z. et al. An integrated transcriptomic and epigenomic atlas of mouse primary motor cortex cell types. bioRxiv 2020.02.29.970558v2 (2020) doi:10.1101/2020.02.29.970558 <https://www.biorxiv.org/content/10.1101/2019.12.11.873398v1>`_

Acknowledgments
---------------
We are grateful for support from the Chan-Zuckerberg Initiative program on `Collaborative Computational Tools for the Human Cell Atlas <https://grants.czi.technology/>`_
(grant 183111) and from the NIH BRAIN Initiative U19 Center for Epigenomics of the Mouse Brain Atlas
(`CEMBA <https://biccn.org/teams/u19-ecker/>`_).
