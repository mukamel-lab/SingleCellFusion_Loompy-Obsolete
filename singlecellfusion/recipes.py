"""
Collection of recipes used for basic analyses of sequencing data

Written by Wayne Doyle unless noted

(C) 2019 Mukamel Lab GPLv2
"""

from . import imputation
from . import decomposition


def pairwise_impute(loom_x,
                    observed_x,
                    imputed_x,
                    pca_attr_x,
                    loom_y,
                    observed_y,
                    imputed_y,
                    pca_attr_y,
                    correlation='positive',
                    neighbor_method='mnn',
                    perform_pca_x=False,
                    perform_pca_y=False,
                    n_pca=50,
                    var_measure_x='vmr',
                    var_measure_y='vmr',
                    common_attr='common_variable_features',
                    corr_idx_attr='correlation_indices',
                    corr_dist_attr='correlation_distances',
                    feature_id_x='Accession',
                    feature_id_y='Accession',
                    k_x_to_y='auto',
                    k_y_to_x='auto',
                    valid_ca_x=None,
                    valid_ca_y=None,
                    valid_ra_x=None,
                    valid_ra_y=None,
                    batch_x=3000,
                    batch_y=3000,
                    verbose=True):
    """
    Performs a pairwise imputation between two data sets (x and y)
    
    Args:
        loom_x (str): Path to loom file for data set x
        observed_x (str): Layer in loom_x containing data for imputation
        imputed_x (str): Output layer in loom_x, will contain imputed loom_y data
        pca_attr_x (str): Attribute in loom_x containing PCs
            If perform_pca_x is true, PCA is run and outputs are stored here
        loom_y (str): Path to loom file for data set y
        observed_y (str): Layer in loom_y containing data for imputation
        imputed_y (str): Output layer in loom_y, will contain imputed loom_x data
        pca_attr_y (str): Attribute in loom_y containing PCs
            If perform_pca_y is true, PCA is run and outputs are stored here
        correlation (str): Expected correlation between data in loom_x and loom_y
            positive or +: Expected correlation is positive
                Examples: scRNA-seq versus snRNA-seq, snATAC-seq versus scRNA-seq
            negative or -: Expected correlation is negative
                Examples: snmC-seq versus scRNA-seq, snmC-seq versus snATAC-seq
        neighbor_method (str): Method for finding nearest neighbors
            mnn: Imputes data for cells that make direct MNNs
            rescue: Imputes data using direct and indirect MNNs
            knn: Imputes data using a kNN graph
        perform_pca_x (bool): Perform PCA on observed_x and add to pca_attr_x
        perform_pca_y (bool): Perform PCA on observed_y and add to pca_attr_y
        n_pca (int): Number of PCs if perform_pca_x or perform_pca_y
        var_measure_x (str): Method for identifying variable genes in loom_x
            vmr: variance mean ratio (useful for RNA-seq datasets)
            sd: standard deviation (useful for snmC-seq datasets)
        var_measure_y (str): Method for identifying variable genes in loom_y
            vmr: variance mean ratio (useful for RNA-seq datasets)
            sd: standard deviation (useful for snmC-seq datasets)
        common_attr (str): Output row attribute specifying common variable features
        corr_idx_attr (str): Output column attribute for correlation indices
        corr_dist_attr (str): Output column attribute for correlation distances
        feature_id_x (str): Row attribute for unique feature IDs for loom_x
            IDs must be able to be matched with feature_id_y
        feature_id_y (str): Row attribute for unique feature IDs for loom_y
        k_x_to_y (int): Number of neighbors from loom_x to loom_y
        k_y_to_x (int): Number of neighbors from loom_x to loom_y
        valid_ca_x (str): Column attribute specifying valid loom_x cells
        valid_ca_y (str): Column attribute specifying valid loom_y cells
        valid_ra_x (str): Column attribute specifying valid loom_x features
        valid_ra_y (str): Column attribute specifying valid_loom_y features
        batch_x (int): Size of batches for loom_x
        batch_y (int): Size of batches for loom_y
        verbose (bool): Enable logging
    """
    # Perform PCA (if needed)
    if perform_pca_x:
        decomposition.batch_pca(loom_file=loom_x,
                                layer=observed_x,
                                out_attr=pca_attr_x,
                                valid_ca=valid_ca_x,
                                valid_ra=valid_ra_x,
                                scale_attr=None,
                                n_pca=n_pca,
                                drop_first=False,
                                batch_size=batch_x,
                                verbose=verbose)
    if perform_pca_y:
        decomposition.batch_pca(loom_file=loom_y,
                                layer=observed_y,
                                out_attr=pca_attr_y,
                                valid_ca=valid_ca_y,
                                valid_ra=valid_ra_y,
                                scale_attr=None,
                                n_pca=n_pca,
                                drop_first=False,
                                batch_size=batch_y,
                                verbose=verbose)
    # Prepare for imputation
    imputation.prep_for_imputation(loom_x=loom_x,
                                   loom_y=loom_y,
                                   observed_x=observed_x,
                                   observed_y=observed_y,
                                   mutual_k_x_to_y=k_x_to_y,
                                   mutual_k_y_to_x=k_y_to_x,
                                   gen_var_x=True,
                                   gen_var_y=True,
                                   var_attr_x='hvf_8000',
                                   var_attr_y='hvf_8000',
                                   feature_id_x=feature_id_x,
                                   feature_id_y=feature_id_y,
                                   n_feat_x=8000,
                                   n_feat_y=8000,
                                   var_measure_x=var_measure_x,
                                   var_measure_y=var_measure_y,
                                   remove_id_version=False,
                                   find_common=True,
                                   common_attr=common_attr,
                                   gen_corr=True,
                                   direction=correlation,
                                   corr_dist_x=corr_dist_attr,
                                   corr_dist_y=corr_dist_attr,
                                   corr_idx_x=corr_idx_attr,
                                   corr_idx_y=corr_idx_attr,
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
                                       observed_x=observed_x,
                                       observed_y=observed_y,
                                       imputed_x=imputed_x,
                                       imputed_y=imputed_y,
                                       mnn_index_x=corr_idx_attr,
                                       mnn_index_y=corr_idx_attr,
                                       mnn_distance_x=corr_dist_attr,
                                       mnn_distance_y=corr_dist_attr,
                                       neighbor_method=neighbor_method,
                                       remove_id_version=False,
                                       constraint_relaxation=1.1,
                                       feature_id_x=feature_id_x,
                                       feature_id_y=feature_id_y,
                                       rescue_metric='euclidean',
                                       mutual_k_x_to_y=k_x_to_y,
                                       mutual_k_y_to_x=k_y_to_x,
                                       rescue_k_x=10,
                                       rescue_k_y=10,
                                       ka_x=5,
                                       ka_y=5,
                                       epsilon_x=1,
                                       epsilon_y=1,
                                       pca_attr_x=pca_attr_x,
                                       pca_attr_y=pca_attr_y,
                                       valid_ca_x=valid_ca_x,
                                       valid_ca_y=valid_ca_y,
                                       valid_ra_x=valid_ra_x,
                                       valid_ra_y=valid_ra_y,
                                       batch_x=batch_x,
                                       batch_y=batch_y,
                                       offset=1e-5,
                                       verbose=verbose)
