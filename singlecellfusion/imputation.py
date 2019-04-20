"""
Functions used to prepare for and impute data across sequencing modalities

The idea of using MNNs and a Gaussian kernel to impute across modalities is
based on ideas from the Marioni, Krishnaswamy, and Pe'er groups. The relevant
citations are:

'Batch effects in single-cell RNA sequencing data are corrected by matching
mutual nearest neighbors' by Laleh Haghverdi, Aaron TL Lun, Michael D Morgan,
and John C Marioni. Published in Nature Biotechnology. DOI:
https://doi.org/10.1038/nbt.4091.

'MAGIC: A diffusion-based imputation method reveals gene-gene interactions in
single-cell RNA-sequencing data.' The publication was authored by: David van
Dijk, Juozas Nainys, Roshan Sharma, Pooja Kathail, Ambrose J Carr, Kevin R Moon,
Linas Mazutis, Guy Wolf, Smita Krishnaswamy, Dana Pe'er. Published in Cell.
DOI: https://doi.org/10.1101/111591.

Below code was written/developed by Fangming Xie, Ethan Armand, and Wayne Doyle

(C) 2018 Mukamel Lab GPLv2
"""

import numpy as np
import time
import logging
from . import general_utils
from . import imputation_helpers as ih

# Start log
imp_log = logging.getLogger(__name__)


def prep_for_imputation(loom_x,
                        loom_y,
                        observed_x,
                        observed_y,
                        mutual_k_x_to_y='auto',
                        mutual_k_y_to_x='auto',
                        gen_var_x=True,
                        gen_var_y=True,
                        csls=False,
                        csls_weight=.1,
                        csls_k=20,
                        var_attr_x='highly_variable',
                        var_attr_y='highly_variable',
                        feature_id_x='Accession',
                        feature_id_y='Accession',
                        var_measure_x='vmr',
                        var_measure_y='vmr',
                        per_decile_x=False,
                        per_decile_y=False,
                        n_feat_x=8000,
                        n_feat_y=8000,
                        feat_pct_x=30,
                        feat_pct_y=30,
                        remove_id_version=False,
                        find_common=True,
                        common_attr='common_variable',
                        distance_metric="correlation",
                        gen_corr=True,
                        direction='positive',
                        corr_dist_x='corr_distances',
                        corr_dist_y='corr_distances',
                        corr_idx_x='corr_indices',
                        corr_idx_y='corr_indices',
                        valid_ca_x=None,
                        valid_ca_y=None,
                        valid_ra_x=None,
                        valid_ra_y=None,
                        batch_x=512,
                        batch_y=512,
                        verbose=False):
    """
    Pre-processes data for performing imputation between two datasets (x and y)
    
    Args:
        loom_x (str): Path to loom file
        loom_y (str): Path to loom file
        observed_x (str/list): Layer(s) containing observed data
        observed_y (str/list): Layer(s) containing observed data
        mutual_k_x_to_y (int/str): Number of nearest neighbors from x to y
            auto will automatically determine a k value
        mutual_k_y_to_x (int/str): Number of nearest neighbors from y to x
            auto will automatically determine a k value
        gen_var_x (bool): Find highly variable features for dataset x
        gen_var_y (bool): Find highly variable features for dataset y
        csls (bool) : whether to find adjusted distance matrices using csls
        csls_weight (float): a wighting factor which decides how much cells
            benefit from having a large average distance used for csls
        csls_k (int) : a number of nearest neighbors used to find average
            cross modality distances used for csls
        var_attr_x (str): Attribute specifying highly variable features
        var_attr_y (str): Attribute specifying highly variable features
        feature_id_x (str): Attribute containing unique feature IDs
        feature_id_y (str): Attribute containing unique feature IDs
        var_measure_x (str): Method for determining highly variable features
            vmr: variance mean ratio
            sd or std: standard deviation
            cv: coefficient of variation
        var_measure_y (str): Method for determining highly variable features
            vmr: variance mean ratio
            sd or std: standard deviation
            cv: coeffecient of variation
        per_decile_x (bool): Specifies how to determine variable features
            if True: get variable features per decile of mean gene expression
            if False: get variable features irregardless of mean gene expression
        per_decile_y (bool): Specifies how to determine variable features
            if True: get variable features per decile of mean gene expression
            if False: get variable features irregardless of mean gene expression
        n_feat_x (int): Number of highly variable features
            Used if per_decile_x is False
        n_feat_y (int): Number of highly variable features
            Used if per_decile_y is False
        feat_pct_x (int): Gets feat_pct_x percent of features per decile
            Used if per_decile_x is True
        feat_pct_y (int): Gets feat_pct_y percent of features per decile
            Used if per_decile_y is True
        remove_id_version (bool): Remove GENCODE ID versions from feature IDs
        find_common (bool): Find highly variable features that are in common
        common_attr (str): Name of attribute specifying common features
        distance_metric (str): Metric for calculating distances
            euclidean
            manhattan
            cosine
            correlation
        gen_corr (bool): Calculate Spearman's rank correlations between cells
        direction (str): Expected direction of correlation
            positive
            negative
        corr_dist_x (str): Attribute specifying correlation values
        corr_dist_y (str): Attribute specifying correlation values
        corr_idx_x (str): Attribute specifying correlation indices
        corr_idx_y (str): Attribute specifying correlation indices
        valid_ca_x (str): Attribute specifying cells to include
        valid_ca_y (str): Attribute specifying cells to include
        valid_ra_x (str): Attribute specifying features to include
        valid_ra_y (str): Attribute specifying features to include
        batch_x (int): Size of batches
        batch_y (int): Size of batches
        verbose (bool): Print logging messages
    """
    # Start log
    if verbose:
        imp_log.info('Preparing to impute between {0} and {1}'.format(loom_x,
                                                                      loom_y))
        t0 = time.time()
    # Get values for k
    if mutual_k_x_to_y == 'auto':
        mutual_k_x_to_y = ih.auto_find_mutual_k(loom_file=loom_y,
                                                valid_ca=valid_ca_y,
                                                verbose=verbose)
    if mutual_k_y_to_x == 'auto':
        mutual_k_y_to_x = ih.auto_find_mutual_k(loom_file=loom_x,
                                                valid_ca=valid_ca_x,
                                                verbose=verbose)
    max_k = np.max([mutual_k_x_to_y, mutual_k_y_to_x])
    # Find highly variable features
    if gen_var_x:
        if per_decile_x:
            ih.get_decile_variable(loom_file=loom_x,
                                   layer=observed_x,
                                   out_attr=var_attr_x,
                                   id_attr=feature_id_x,
                                   percentile=feat_pct_x,
                                   measure=var_measure_x,
                                   valid_ra=valid_ra_x,
                                   valid_ca=valid_ca_x,
                                   batch_size=batch_x,
                                   verbose=verbose)
        else:
            ih.get_n_variable_features(loom_file=loom_x,
                                       layer=observed_x,
                                       out_attr=var_attr_x,
                                       id_attr=feature_id_x,
                                       n_feat=n_feat_x,
                                       measure=var_measure_x,
                                       valid_ra=valid_ra_x,
                                       valid_ca=valid_ca_x,
                                       batch_size=batch_x,
                                       verbose=verbose)
    if gen_var_y:
        if per_decile_y:
            ih.get_decile_variable(loom_file=loom_y,
                                   layer=observed_y,
                                   out_attr=var_attr_y,
                                   id_attr=feature_id_y,
                                   percentile=feat_pct_y,
                                   measure=var_measure_y,
                                   valid_ra=valid_ra_y,
                                   valid_ca=valid_ca_y,
                                   batch_size=batch_y,
                                   verbose=verbose)
        else:
            ih.get_n_variable_features(loom_file=loom_y,
                                       layer=observed_y,
                                       out_attr=var_attr_y,
                                       id_attr=feature_id_y,
                                       n_feat=n_feat_y,
                                       measure=var_measure_y,
                                       valid_ra=valid_ra_y,
                                       valid_ca=valid_ca_y,
                                       batch_size=batch_y,
                                       verbose=verbose)
    # Find common variable features
    if find_common:
        ih.find_common_features(loom_x,
                                loom_y,
                                out_attr=common_attr,
                                feature_id_x=feature_id_x,
                                feature_id_y=feature_id_y,
                                valid_ra_x=var_attr_x,
                                valid_ra_y=var_attr_y,
                                remove_version=remove_id_version,
                                verbose=verbose)
    # Generate correlations
    if gen_corr:
        ih.generate_distances(loom_x=loom_x,
                              observed_x=observed_x,
                              dist_x=corr_dist_x,
                              idx_x=corr_idx_x,
                              max_k_x=max_k,
                              loom_y=loom_y,
                              observed_y=observed_y,
                              dist_y=corr_dist_y,
                              idx_y=corr_idx_y,
                              max_k_y=max_k,
                              direction=direction,
                              metric=distance_metric,
                              feature_id_x=feature_id_x,
                              feature_id_y=feature_id_y,
                              valid_ca_x=valid_ca_x,
                              ra_x=common_attr,
                              valid_ca_y=valid_ca_y,
                              ra_y=common_attr,
                              batch_x=batch_x,
                              batch_y=batch_y,
                              remove_version=remove_id_version,
                              verbose=verbose)
    if csls:
        ih.generate_csls_distance(loom_x=loom_x,
                                  observed_x=observed_x,
                                  dist_x=corr_dist_x,
                                  idx_x=corr_idx_x,
                                  max_k_x=max_k,
                                  loom_y=loom_y,
                                  observed_y=observed_y,
                                  dist_y=corr_dist_y,
                                  idx_y=corr_idx_y,
                                  max_k_y=max_k,
                                  direction=direction,
                                  feature_id_x=feature_id_x,
                                  feature_id_y=feature_id_y,
                                  avg_weight=csls_weight,
                                  check_k=csls_k,
                                  metric=distance_metric,
                                  valid_ca_x=valid_ca_x,
                                  ra_x=common_attr,
                                  valid_ca_y=valid_ca_y,
                                  ra_y=common_attr,
                                  batch_x=batch_x,
                                  batch_y=batch_y,
                                  remove_version=remove_id_version,
                                  verbose=verbose)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        imp_log.info(
            'Prepared for imputation in {0:.2f} {1}'.format(time_run, time_fmt))


def impute_between_datasets(loom_x,
                            loom_y,
                            observed_x,
                            observed_y,
                            imputed_x,
                            imputed_y,
                            mnn_index_x='corr_indices',
                            mnn_index_y='corr_indices',
                            mnn_distance_x='corr_distances',
                            mnn_distance_y='corr_distances',
                            neighbor_method="rescue",
                            remove_id_version=False,
                            constraint_relaxation=1.1,
                            feature_id_x='Accession',
                            feature_id_y='Accession',
                            rescue_metric='euclidean',
                            mutual_k_x_to_y='auto',
                            mutual_k_y_to_x='auto',
                            rescue_k_x=10,
                            rescue_k_y=10,
                            ka_x=5,
                            ka_y=5,
                            epsilon_x=1,
                            epsilon_y=1,
                            pca_attr_x='PCA',
                            pca_attr_y='PCA',
                            valid_ca_x=None,
                            valid_ca_y=None,
                            valid_ra_x=None,
                            valid_ra_y=None,
                            batch_x=512,
                            batch_y=512,
                            offset=1e-5,
                            verbose=False):
    """
    Imputes counts between dataset x and dataset y
    Args:
        loom_x (str): Path to loom file
        loom_y (str): Path to loom file
        observed_x (str/list): Layer(s) containing observed count data
        observed_y (str/list): Layer(s) containing observed count data
        imputed_x (str/list): Output layer(s) for imputed count data
        imputed_y (str/list): Output layer(s) for imputed count data
        mnn_index_x (str): Attribute containing indices for MNNs
            corr_idx_x in prep_for_imputation
        mnn_index_y (str): Attribute containing indices for MNNs
            corr_idx_y in prep_for_imputation
        mnn_distance_x (str): Attribute containing distances for MNNs
            corr_dist_x in prep_for_imputation
        mnn_distance_y (str): Attribute containing distances for MNNs
            corr_dist_y in prep_for imputation
        neighbor_method (str): How cells are chosen for imputation 
            rescue - include cells that did not make MNNs
            mnn - only include cells that made MNNs
            knn - use a restricted knn search to find neighbors
        remove_id_version (bool): Remove GENCODE version from feature IDs
        constraint_relaxation(float): used for knn imputation
            a ratio determining the number of neighbors that can be
            formed by cells in the other dataset. Increasing it means
            neighbors can be distributed more unevenly among cells, 
            one means each cell is used equally.
        feature_id_x (str): Attribute specifying unique feature IDs
        feature_id_y (str): Attribute specifying unique feature IDs
        rescue_metric (str): Metric for calculating distances if rescue
            euclidean
            manhattan
            cosine
        mutual_k_x_to_y (int/str): k value for MNNs
            auto will automatically select a k value
        mutual_k_y_to_x (int/str): k value for MNNs
            auto will automatically select a k value
        rescue_k_x (int/str): k value for rescue
             auto will automatically select a k value
        rescue_k_y (int/str): k value for rescue
            auto will automatically select a k value
        ka_x (int): Neighbor to normalize distances by
        ka_y (int): Neighbor to normalize distances by
        epsilon_x (float): Noise parameter for Gaussian kernel
        epsilon_y (float): Noise parameter for Gaussian kernel
        pca_attr_x (str): Attribute containing PCs for rescue
        pca_attr_y (str): Attribute containing PCs for rescue
        valid_ca_x (str): Attribute specifying valid cells
        valid_ca_y (str): Attribute specifying valid cells
        valid_ra_x (str): Attribute specifying valid features
        valid_ra_y (str): Attribute specifying valid features
        batch_x (int): Batch size
        batch_y (int): Batch size
        offset (float): Offset for Markov normalization
        verbose (bool): Print logging messages
    """
    if verbose:
        t0 = time.time()
    # Handle inputs
    if mutual_k_x_to_y == 'auto':
        mutual_k_x_to_y = ih.auto_find_mutual_k(loom_file=loom_y,
                                                valid_ca=valid_ca_y,
                                                verbose=verbose)
    if mutual_k_y_to_x == 'auto':
        mutual_k_y_to_x = ih.auto_find_mutual_k(loom_file=loom_x,
                                                valid_ca=valid_ca_x,
                                                verbose=verbose)
    if neighbor_method == "rescue":
        if rescue_k_x == 'auto':
            rescue_k_x = ih.auto_find_rescue_k(loom_file=loom_x,
                                               valid_ca=valid_ca_x,
                                               verbose=verbose)
        if rescue_k_y == 'auto':
            rescue_k_y = ih.auto_find_rescue_k(loom_file=loom_y,
                                               valid_ca=valid_ca_y,
                                               verbose=verbose)
        ka_x = ih.check_ka(k=rescue_k_x,
                           ka=ka_x)
        ka_y = ih.check_ka(k=rescue_k_y,
                           ka=ka_y)
        if pca_attr_x is None or pca_attr_y is None:
            err_msg = 'Missing pca_attr for rescue'
            if verbose:
                imp_log.error(err_msg)
            raise ValueError(err_msg)
    # Impute data for loom_x
    ih.loop_impute_data(loom_source=loom_y,
                        layer_source=observed_y,
                        id_source=feature_id_y,
                        cell_source=valid_ca_y,
                        feat_source=valid_ra_y,
                        loom_target=loom_x,
                        layer_target=imputed_x,
                        id_target=feature_id_x,
                        cell_target=valid_ca_x,
                        feat_target=valid_ra_x,
                        mnn_distance_target=mnn_distance_x,
                        mnn_distance_source=mnn_distance_y,
                        mnn_index_target=mnn_index_x,
                        mnn_index_source=mnn_index_y,

                        k_src_tar=mutual_k_y_to_x,
                        k_tar_src=mutual_k_x_to_y,
                        k_rescue=rescue_k_x,
                        ka=ka_x,
                        epsilon=epsilon_x,
                        pca_attr=pca_attr_x,
                        metric=rescue_metric,
                        neighbor_method=neighbor_method,
                        constraint_relaxation=constraint_relaxation,
                        remove_version=remove_id_version,
                        offset=offset,
                        batch_target=batch_x,
                        batch_source=batch_y,
                        verbose=verbose)
    # Impute data for loom_y
    ih.loop_impute_data(loom_source=loom_x,
                        layer_source=observed_x,
                        id_source=feature_id_x,
                        cell_source=valid_ca_x,
                        feat_source=valid_ra_x,
                        loom_target=loom_y,
                        layer_target=imputed_y,
                        id_target=feature_id_y,
                        cell_target=valid_ca_y,
                        feat_target=valid_ra_y,
                        mnn_distance_target=mnn_distance_y,
                        mnn_distance_source=mnn_distance_x,
                        mnn_index_target=mnn_index_y,
                        mnn_index_source=mnn_index_x,

                        k_src_tar=mutual_k_x_to_y,
                        k_tar_src=mutual_k_y_to_x,
                        k_rescue=rescue_k_y,
                        ka=ka_y,
                        epsilon=epsilon_y,
                        pca_attr=pca_attr_y,
                        metric=rescue_metric,
                        neighbor_method=neighbor_method,
                        constraint_relaxation=constraint_relaxation,
                        remove_version=remove_id_version,
                        offset=offset,
                        batch_target=batch_y,
                        batch_source=batch_x,
                        verbose=verbose)
    # Impute data for loom_y
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        imp_log.info('Completed imputation in {0:.2f} {1}'.format(time_run,
                                                                  time_fmt))
