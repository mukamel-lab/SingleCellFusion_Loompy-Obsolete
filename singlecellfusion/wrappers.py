"""
Functions used to run multiple SingleCellFusion steps at once

Below code was written/developed by Fangming Xie, Ethan Armand, and Wayne Doyle

(C) 2019 Mukamel Lab GPLv2
"""

import logging
from . import features
from . import imputation
from . import integration
from . import utils

# Start log
wrap_log = logging.getLogger(__name__)


def fuse_data(loom_source,
              loom_target,
              correlation,
              loom_output='integrated.loom',
              layer_source='',
              layer_target='',
              layer_impute='imputed',
              feat_source='Accession',
              feat_target='Accession',
              cell_source='CellID',
              cell_target='CellID',
              cluster_source='ClusterID',
              cluster_target='ClusterID',
              pca_target=None,
              valid_ra_source=None,
              valid_ra_target=None,
              valid_ca_source=None,
              valid_ca_target=None,
              label_source=None,
              label_target=None,
              var_method='kruskal',
              common_attr='CommonVariable',
              variable_attr='VariableFeatures',
              kruskal_n=8000,
              var_percentile=30,
              neighbor_method='knn',
              n_neighbors=20,
              relaxation=10,
              speed_factor=10,
              n_trees=10,
              remove_version=False,
              low_mem=False,
              batch_size=5000,
              tmp_dir=None,
              seed=23,
              verbose=True):
    """
    Wrapper for running all steps of SingleCellFusion. Finds common variable features, imputes data, and integrates.
    
    Args:
        loom_source (str): Path to loom file that will provide counts to others
        loom_target (str/list): Path(s) to loom files that will receive imputed counts
        correlation (str/list): Expected correlation between loom_source and loom_target
            positive/+ for RNA-seq and ATAC-seq
            negative/- for RNA-seq or ATAC-seq and snmC-seq
        loom_output (str): Path to output loom file that contains observed/imputed counts for a given modality
        layer_source (str): Layer in loom_source containing observed counts
        layer_target (str/list): Layer(s) in loom_target containing imputed counts
        layer_impute (str/list): Output layer in loom_target
            A row attribute with the format Valid_{layer_out} will be added
            A col attribute with the format Valid_{layer_out} will be added
        feat_source (str): Row attribute specifying unique feature names in loom_source
        feat_target (str/list): Row attribute(s) specifying unique feature names in loom_target
        cell_source (str): Column attribute containing unique cell IDs in loom_source
            IDs will be included in loom_output under the column attribute CellID
        cell_target (str/list): Column attribute(s) containing unique cell IDs in loom_target
        cluster_source (str/list/None): Column attribute(s) specifying cluster assignments for each cell in loom_source
            Necessary if the kruskal method is used
        cluster_target (str/list/None): Column attribute(s) specifying cluster assignments for each cell in loom_target
            Necessary if the kruskal method is used
        pca_target (str/list/None): Column attribute containing PCs in loom_target
            Used if neighbor_method is mnn_rescue
        valid_ra_source (str): Row attribute specifying valid features in loom_source
            Should point to a boolean array
        valid_ra_target (str/list): Row attribute(s) specifying valid features in loom_target
            Should point to a boolean array
        valid_ca_source (str): Column attribute specifying valid cells in loom_source
            Should point to a boolean array
        valid_ca_target (str/list): Column attribute(s) specifying valid cells in loom_target
            Should point to a boolean array
        label_source (str/None): Optional, labels to be added to cells from loom_source
            Will be saved in the column attribute Modality
            If provided, label_target must also be provided
        label_target (str/list/None): Optional, labels to be added to cells from loom_target
        var_method (str): Method for finding variable genes
            vmr: variance mean ratio (found in deciles)
            sd: standard deviation (found in deciles)
            kruskal: Kruskal-Wallis test (recommended)
        common_attr (str): Output row attribute which will specify valid common features
        variable_attr (str/None): Output row attribute which will specify valid, variable features
            Only used if method is sd or vmr
            If method is kruskal the following row attributes will automatically be added
                 kruskal_H: H statistic from Kruskal-Wallis test
                 kruskal_pval: p-value from Kruskal-Wallis test
                 kruskal_max_cluster_pct: Percentage of cells with non-zero counts in clusters
                    Percent is from the cluster that has the largest number of non-zero counts
                 kruskal_cluster_attr: Cluster attribute used for performing Kruskal-Wallis test
        kruskal_n (int/None): Use the top kruskal_n number of genes from each loom file to find common features
            Only used if method is kruskal
        var_percentile (int): Whole number (0-100) percent of variable genes to select per decile of expression
            Only used if method is sd or vmr
        neighbor_method (str): Method for performing imputation
                knn: constrained knn
                mnn_direct: mutual nearest neighbors with rescue of cells (can only be done if low_mem = False)
                mnn_rescue: mutual nearest neighbors with cells that made direct neighbors (only if low_mem = False)
        n_neighbors (int/list): Minimum amount of neighbors that can be made
        relaxation (int/list): Relax search for constrained kNN by this factor
            Increases the number of neighbors that a source cell can make before saturation
            Only used if method is knn
        speed_factor (int/list): Speed up search of constrained kNN by this factor
            Will increase memory but decrease running time
            Only used if method is knn
        n_trees (int): Number of trees for approximate kNN search
            See Annoy documentation
        remove_version (bool/list): If true remove version number
            Anything after the first period is dropped (useful for GENCODE IDs)
            If a list, will behave differently for each loom file
            If a boolean, will behave the same for each loom_file
        low_mem (bool): If true, performs imputation in a slow but less memory-intensive manner
        batch_size (int): Chunk size for batches, used if low_mem
            Higher values mean faster code, but more memory
        tmp_dir (str): Path to directory for temporary files
            If None, uses system defaults
        seed (int): Seed for randomization
        verbose (bool): Print logging messages
    """
    # Check inputs
    is_a_list = False
    if isinstance(loom_target, list):
        utils.all_same_type_size(parameters=[loom_target,
                                             layer_target,
                                             correlation],
                                 expected_type='list',
                                 confirm_size=True)
        check_parameters = [feat_target,
                            cell_target,
                            valid_ra_target,
                            valid_ca_target,
                            remove_version,
                            label_target,
                            layer_impute,
                            pca_target]
        checked = utils.mimic_list(parameters=check_parameters,
                                   list_len=len(loom_target))
        feat_target = checked[0]
        cell_target = checked[1]
        valid_ra_target = checked[2]
        valid_ca_target = checked[3]
        remove_version = checked[4]
        label_target = checked[5]
        layer_impute = checked[6]
        pca_target = checked[7]
        is_a_list = True
    elif isinstance(loom_target, str):
        utils.all_same_type_size(parameters=[loom_target,
                                             layer_target,
                                             correlation,
                                             feat_target,
                                             cell_target],
                                 expected_type='str',
                                 confirm_size=False)

    # Prep for finding common features
    if is_a_list:
        loom_files = loom_target.copy()
        layers = layer_target.copy()
        cell_attrs = cell_target.copy()
        feat_attrs = feat_target.copy()
        cluster_attrs = cluster_target.copy()
        valid_ras = valid_ra_target.copy()
        valid_cas = valid_ca_target.copy()
    else:
        loom_files = [loom_target]
        layers = [layer_target]
        cell_attrs = [cell_target]
        feat_attrs = [feat_target]
        cluster_attrs = [cluster_target]
        valid_ras = [valid_ra_target]
        valid_cas = [valid_ca_target]
    loom_files.append(loom_source)
    layers.append(layer_source)
    cell_attrs.append(cell_source)
    feat_attrs.append(feat_source)
    cluster_attrs.append(cluster_source)
    valid_ras.append(valid_ra_source)
    valid_cas.append(valid_ca_source)
    # Find common features
    features.find_common_variable(loom_files=loom_files,
                                  layers=layers,
                                  method=var_method,
                                  cell_attrs=cell_attrs,
                                  feat_attrs=feat_attrs,
                                  cluster_attrs=cluster_attrs,
                                  common_attr=common_attr,
                                  variable_attr=variable_attr,
                                  valid_ras=valid_ras,
                                  valid_cas=valid_cas,
                                  kruskal_n=kruskal_n,
                                  percentile=var_percentile,
                                  low_mem=low_mem,
                                  batch_size=batch_size,
                                  remove_version=remove_version,
                                  verbose=verbose)

    # Impute data
    imputation.perform_imputation(loom_source=loom_source,
                                  loom_target=loom_target,
                                  method=neighbor_method,
                                  layer_source=layer_source,
                                  layer_target=layer_target,
                                  layer_impute=layer_impute,
                                  correlation=correlation,
                                  feat_source=feat_source,
                                  feat_target=feat_target,
                                  cell_source=cell_source,
                                  cell_target=cell_target,
                                  pca_target=pca_target,
                                  valid_ra_source=common_attr,
                                  valid_ra_target=common_attr,
                                  valid_ca_source=valid_ca_source,
                                  valid_ca_target=valid_ca_target,
                                  n_neighbors=n_neighbors,
                                  relaxation=relaxation,
                                  speed_factor=speed_factor,
                                  n_trees=n_trees,
                                  remove_version=remove_version,
                                  low_mem=low_mem,
                                  batch_size=batch_size,
                                  tmp_dir=tmp_dir,
                                  seed=seed,
                                  verbose=verbose)
    # Integrate data
    integration.integrate_data(loom_source=loom_source,
                               loom_target=loom_target,
                               loom_output=loom_output,
                               layer_source=layer_source,
                               layer_target=layer_target,
                               feat_source=feat_source,
                               feat_target=feat_target,
                               cell_source=cell_source,
                               cell_target=cell_target,
                               label_source=label_source,
                               label_target=label_target,
                               valid_ra_source=common_attr,
                               valid_ra_target=common_attr,
                               valid_ca_source=valid_ca_source,
                               valid_ca_target=valid_ca_target,
                               remove_version=remove_version,
                               low_mem=low_mem,
                               batch_size=batch_size,
                               verbose=verbose)
