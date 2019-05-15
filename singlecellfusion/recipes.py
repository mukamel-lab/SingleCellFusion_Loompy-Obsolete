"""
Collection of recipes used for basic analyses of sequencing data

Written by Wayne Doyle unless noted

(C) 2019 Mukamel Lab GPLv2
"""

from . import imputation
from . import decomposition


def pairwise_impute(loom_x,
                    layer_x,
                    loom_y,
                    layer_y,
                    imputed_x='imputed',
                    imputed_y='imputed',
                    neighbor_method='mnn',
                    correlation='positive',
                    gen_pca_x=False,
                    gen_pca_y=False,
                    pca_attr_x='PCA',
                    pca_attr_y='PCA',
                    n_pca_x=50,
                    n_pca_y=50,
                    var_measure_x='vmr',
                    var_measure_y='vmr',
                    feature_id_x='Accession',
                    feature_id_y='Accession',
                    per_decile_x=True,
                    per_decile_y=True,
                    feat_pct_x=30,
                    feat_pct_y=30,
                    n_feat_x=8000,
                    n_feat_y=8000,
                    k_x_to_y='auto',
                    k_y_to_x='auto',
                    ka_x=5,
                    ka_y=5,
                    rescue_k_x=10,
                    rescue_k_y=10,
                    constraint_relaxation=1.1,
                    valid_ca_x=None,
                    valid_ca_y=None,
                    valid_ra_x=None,
                    valid_ra_y=None,
                    batch_x=3000,
                    batch_y=3000,
                    remove_id_version=False,
                    verbose=True):
    """
    Performs a pairwise imputation between two data sets (x and y)
    
    Args:
        loom_x (str): Path to loom file for data set x
        layer_x (str): Layer in loom_x containing data for imputation
        loom_y (str): Path to loom file for data set y
        layer_y (str): Layer in loom_y containing data for imputation
        imputed_x (str): Output layer in loom_x containing imputed data
        imputed_y (str): Output layer in loom_y containing imputed data
        neighbor_method (str): Method for finding nearest neighbors
            mnn: Imputes data for cells that make direct MNNs
            rescue: Imputes data using direct and indirect MNNs
            knn: Imputes data using a kNN graph
        correlation (str): Expected correlation between data
            positive or +: Expected correlation is positive
                Examples: scRNA-seq vs snRNA-seq, snATAC-seq vs scRNA-seq
            negative or -: Expected correlation is negative
                Examples: snmC-seq vs scRNA-seq, snmC-seq vs snATAC-seq
        gen_pca_x (bool): Perform PCA on layer_x and add to pca_attr_x
        gen_pca_y (bool): Perform PCA on layer_y and add to pca_attr_y
        pca_attr_x (str): Attribute containing PCs for loom_x
            Used if neighbor_method is rescue
        pca_attr_y (str): Attribute containing PCs for loom_y
        n_pca_x (int): Number of PCs for loom_x
        n_pca_y (int): Number of PCs for loom_y
        var_measure_x (str): Method for identifying variable genes in loom_x
            vmr: variance mean ratio (useful for RNA-seq datasets)
            sd: standard deviation (useful for snmC-seq datasets)
        var_measure_y (str): Method for identifying variable genes in loom_y
            vmr: variance mean ratio (useful for RNA-seq datasets)
            sd: standard deviation (useful for snmC-seq datasets)
        feature_id_x (str): Row attribute for unique feature IDs for loom_x
            IDs must be able to be matched with feature_id_y
        feature_id_y (str): Row attribute for unique feature IDs for loom_y
        per_decile_x (bool): If true, find variable features per decile
        per_decile_y (bool): If true, find variable features per decile
        feat_pct_x (int): Gets feat_pct_x percent of features per decile
        feat_pct_y (int): Gets feat_pct_y percent of features per decile
        n_feat_x (int): If not per_decile_x, number of variable features to find
        n_feat_y (int): If not per_decile_y, number of variable features to find
        k_x_to_y (int): Number of neighbors from loom_x to loom_y
        k_y_to_x (int): Number of neighbors from loom_x to loom_y
        ka_x (int): Weight nearest neighbors to this neighbor
        ka_y (int): Weight nearest neighbors to this neighbor
        rescue_k_x (int): Number of nearest neighbors within modality
            Used if neighbor_method is rescue
        rescue_k_y (int): Number of nearest neighbors within modality
            Used if neighbor_method is rescue
        constraint_relaxation: Specifies ratio of neighbors formed by cells
            Used if neighbor_method is knn
        valid_ca_x (str): Column attribute specifying valid loom_x cells
        valid_ca_y (str): Column attribute specifying valid loom_y cells
        valid_ra_x (str): Column attribute specifying valid loom_x features
        valid_ra_y (str): Column attribute specifying valid_loom_y features
        batch_x (int): Size of batches for loom_x
        batch_y (int): Size of batches for loom_y
        remove_id_version (bool): Remove GENCODE ID version number
        verbose (bool): Enable logging
    """
    # Prepare for imputation
    imputation.prep_for_imputation(loom_x=loom_x,
                                   loom_y=loom_y,
                                   layer_x=layer_x,
                                   layer_y=layer_y,
                                   gen_var_x=True,
                                   gen_var_y=True,
                                   var_attr_x='highly_variable',
                                   var_attr_y='highly_variable',
                                   feature_id_x=feature_id_x,
                                   feature_id_y=feature_id_y,
                                   var_measure_x=var_measure_x,
                                   var_measure_y=var_measure_y,
                                   per_decile_x=per_decile_x,
                                   per_decile_y=per_decile_y,
                                   n_feat_x=n_feat_x,
                                   n_feat_y=n_feat_y,
                                   feat_pct_x=feat_pct_x,
                                   feat_pct_y=feat_pct_y,
                                   remove_id_version=remove_id_version,
                                   find_common=True,
                                   common_attr='common_variable',
                                   gen_pca_x=gen_pca_x,
                                   pca_attr_x=pca_attr_x,
                                   n_pca_x=n_pca_x,
                                   gen_pca_y=gen_pca_y,
                                   pca_attr_y=pca_attr_y,
                                   n_pca_y=n_pca_y,
                                   valid_ca_x=valid_ca_x,
                                   valid_ca_y=valid_ca_y,
                                   valid_ra_x=valid_ra_x,
                                   valid_ra_y=valid_ra_y,
                                   batch_x=batch_x,
                                   batch_y=batch_y,
                                   verbose=verbose)

    # Impute
    imputation.impute_between_datasets(loom_x=loom_x,
                                       loom_y=loom_y,
                                       layer_x=layer_x,
                                       layer_y=layer_y,
                                       imputed_x=imputed_x,
                                       imputed_y=imputed_y,
                                       common_attr='common_variable',
                                       correlation=correlation,
                                       neighbor_index_x='corr_indices',
                                       neighbor_index_y='corr_indices',
                                       neighbor_distance_x='corr_distances',
                                       neighbor_distance_y='corr_distances',
                                       neighbor_method=neighbor_method,
                                       constraint_relaxation=constraint_relaxation,
                                       feature_id_x=feature_id_x,
                                       feature_id_y=feature_id_y,
                                       mutual_k_x_to_y=k_x_to_y,
                                       mutual_k_y_to_x=k_y_to_x,
                                       rescue_k_x=rescue_k_x,
                                       rescue_k_y=rescue_k_y,
                                       ka_x=ka_x,
                                       ka_y=ka_y,
                                       epsilon_x=1,
                                       epsilon_y=1,
                                       pca_attr_x='PCA',
                                       pca_attr_y='PCA',
                                       valid_ca_x=valid_ca_x,
                                       valid_ca_y=valid_ca_y,
                                       valid_ra_x=valid_ra_x,
                                       valid_ra_y=valid_ra_y,
                                       batch_x=batch_x,
                                       batch_y=batch_y,
                                       remove_id_version=remove_id_version,
                                       verbose=verbose)
