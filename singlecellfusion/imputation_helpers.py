"""
Collection of functions used to perform imputation across datasets

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

Below code was written/developed by Fangming Xie and Wayne Doyle, unless noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
import pandas as pd
import time
from scipy import sparse
from sklearn.metrics import pairwise_distances
import functools
import logging
import gc
from numba import jit
from . import general_utils
from . import loom_utils

# Start log
ih_log = logging.getLogger(__name__)


def batch_mean_and_std(loom_file,
                       layer,
                       axis=None,
                       valid_ca=None,
                       valid_ra=None,
                       batch_size=512,
                       verbose=False):
    """
    Batch calculates mean and standard deviation

    Args:
        loom_file (str): Path to loom file containing mC/C counts
        layer (str): Layer containing mC/C counts
        axis (int): Axis to calculate mean and standard deviation
            None: values are for entire layer
            0: Statistics are for cells (will read all cells into memory)
            1: Statistics are for features (will read all features into memory)
        valid_ca (str): Optional, only use cells specified by valid_ca
        valid_ra (str): Optional, only use features specified by valid_ra
        batch_size (int): Number of elements per chunk
            If axis is None, chunks are number of cells
            If axis == 0, chunks are number of features
            If axis == 1, chunks are number of cells
        verbose (boolean): If true, print helpful progress messages

    Returns:
        mean (float): Mean value for specified layer
        std (float): Standard deviation value for specified layer

    Assumptions:
        (row/col)_attr specifies a boolean array attribute

    To Do:
        Make axis selection consistent across all functions

    Based on code from:
        http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    """
    # Set defaults
    old_mean = None
    old_std = None
    old_obs = None
    first_iter = True
    if axis is None:
        loom_axis = 1
    else:
        loom_axis = axis
    # Start log
    if verbose:
        ih_log.info('Calculating statistics for {}'.format(loom_file))
        t0 = time.time()
    # Get indices
    layers = loom_utils.make_layer_list(layers=layer)
    with loompy.connect(filename=loom_file, mode='r') as ds:
        for (_, selection, view) in ds.scan(axis=loom_axis,
                                            layers=layers,
                                            batch_size=batch_size):
            # Parse data
            dat = view.layers[layer][:, :]
            if valid_ca:
                col_idx = view.ca[valid_ca].astype(bool)
            else:
                col_idx = np.ones((view.shape[1],), dtype=bool)
            if valid_ra:
                row_idx = view.ra[valid_ra].astype(bool)
            else:
                row_idx = np.ones((view.shape[0],), dtype=bool)
            if not np.any(col_idx) or not np.any(row_idx):
                continue
            if axis is None:
                dat = dat[row_idx, :][:, col_idx]
            elif axis == 0:
                dat[:, np.logical_not(col_idx)] = 0
                dat = dat[row_idx, :]
            elif axis == 1:
                dat[np.logical_not(row_idx), :] = 0
                dat = dat[:, col_idx]
            # Get new values
            new_mean = np.mean(dat, axis=axis)
            new_std = np.std(dat, axis=axis)
            new_obs = dat.shape[1]
            # Update means
            if first_iter:
                old_mean = new_mean
                old_std = new_std
                old_obs = new_obs
                first_iter = False
            else:
                # Get updated values
                upd_mean = (old_obs / (old_obs + new_obs) * old_mean +
                            new_obs / (old_obs + new_obs) * new_mean)
                upd_std = np.sqrt(old_obs / (old_obs + new_obs) * old_std ** 2 +
                                  new_obs / (old_obs + new_obs) * new_std ** 2 +
                                  old_obs * new_obs / (old_obs + new_obs) ** 2 *
                                  (old_mean - new_mean) ** 2)
                upd_obs = old_obs + new_obs
                # Perform update
                old_mean = upd_mean
                old_std = upd_std
                old_obs = upd_obs
    # Set values
    my_mean = old_mean
    my_std = old_std
    # Restrict to valid cells/features
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ca,
                                        columns=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ra,
                                        columns=False,
                                        inverse=False)
    if axis == 0:
        my_mean = my_mean[col_idx]
        my_std = my_std[col_idx]
    elif axis == 1:
        my_mean = my_mean[row_idx]
        my_std = my_std[row_idx]
    if my_mean is None:
        raise ValueError('Could not calculate statistics')
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        ih_log.info(
            'Calculated statistics in {0:.2f} {1}'.format(time_run, time_fmt))
    return [my_mean, my_std]


def get_decile_variable(loom_file,
                        layer,
                        out_attr=None,
                        id_attr='Accession',
                        percentile=30,
                        measure='vmr',
                        valid_ra=None,
                        valid_ca=None,
                        batch_size=512,
                        verbose=False):
    """
    Generates an attribute indicating the highest variable features per decile

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer containing relevant counts
        out_attr (str): Name of output attribute which will specify features
            Defaults to hvf_{n}
        id_attr (str): Attribute specifying unique feature IDs
        percentile (int): Percent of highly variable features per decile
        measure (str): Method of measuring variance
            vmr: variance mean ratio
            sd/std: standard deviation
            cv: coefficient of variation
        valid_ra (str): Optional, attribute to restrict features by
        valid_ca (str): Optional, attribute to restrict cells by
        batch_size (int): Size of chunks
            Will generate a dense array of batch_size by cells
        verbose (bool): Print logging messages
    """
    if verbose:
        ih_log.info(
            'Finding {0}% variable features per decile for {1}'.format(
                percentile, loom_file))
        t0 = time.time()
    # Get valid indices
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ra,
                                        columns=False,
                                        as_bool=True,
                                        inverse=False)
    # Determine variability
    my_mean, my_std = batch_mean_and_std(loom_file=loom_file,
                                         layer=layer,
                                         axis=1,
                                         valid_ca=valid_ca,
                                         valid_ra=valid_ra,
                                         batch_size=batch_size,
                                         verbose=verbose)
    # Find highly variable
    with loompy.connect(filename=loom_file, mode='r') as ds:
        feat_labels = ds.ra[id_attr][row_idx]
        feat_idx = pd.Series(np.zeros((ds.shape[0])),
                             index=ds.ra[id_attr])
    if measure.lower() == 'sd' or measure.lower() == 'std':
        tmp_var = pd.Series(my_std, index=feat_labels)
    elif measure.lower() == 'vmr':
        my_var = my_std ** 2
        my_vmr = (my_var + 1) / (my_mean + 1)
        tmp_var = pd.Series(my_vmr, index=feat_labels)
    elif measure.lower() == 'cv':
        my_cv = (my_std + 1) / (my_mean + 1)
        tmp_var = pd.Series(my_cv, index=feat_labels)
    else:
        err_msg = 'Unsupported measure value ({})'.format(measure)
        if verbose:
            ih_log.error(err_msg)
        raise ValueError(err_msg)
    # Get variable
    my_mean = pd.Series(my_mean, index=feat_labels)
    feat_deciles = pd.qcut(my_mean,
                           q=10,
                           labels=False).to_frame('decile')
    hvf = []
    for decile, batch_info in feat_deciles.groupby('decile'):
        gene_group = batch_info.index.values
        var_gg = tmp_var.loc[gene_group]
        # Genes per decile
        hvf_group = gene_group[var_gg > np.percentile(var_gg, 100 - percentile)]
        if decile != 9:  # Ignore final decile
            hvf.append(hvf_group)
    # Add to loom file
    hvf = np.hstack(hvf)
    feat_idx.loc[hvf] = 1
    if out_attr is None:
        out_attr = 'hvf_decile_{}'.format(percentile)
    with loompy.connect(loom_file) as ds:
        ds.ra[out_attr] = feat_idx.values.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        ih_log.info(
            'Found {0} variable features in {1:.2f} {2}'.format(hvf.shape[0],
                                                                time_run,
                                                                time_fmt))


def get_n_variable_features(loom_file,
                            layer,
                            out_attr=None,
                            id_attr='Accession',
                            n_feat=4000,
                            measure='vmr',
                            valid_ra=None,
                            valid_ca=None,
                            batch_size=512,
                            verbose=False):
    """
    Generates an attribute indicating the n highest variable features

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer containing relevant counts
        out_attr (str): Name of output attribute which will specify features
            Defaults to hvf_{n}
        id_attr (str): Attribute specifying unique feature IDs
        n_feat (int): Number of highly variable features
        measure (str): Method of measuring variance
            vmr: variance mean ratio
            sd/std: standard deviation
            cv: coefficient of variation
        valid_ra (str): Optional, attribute to restrict features by
        valid_ca (str): Optional, attribute to restrict cells by
        batch_size (int): Size of chunks
            Will generate a dense array of batch_size by cells
        verbose (bool): Print logging messages
    """
    if verbose:
        ih_log.info(
            'Finding {} variable features for {}'.format(n_feat, loom_file))
        t0 = time.time()
    # Get valid indices
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ra,
                                        columns=False,
                                        as_bool=True,
                                        inverse=False)
    # Determine variability
    my_mean, my_std = batch_mean_and_std(loom_file=loom_file,
                                         layer=layer,
                                         axis=1,
                                         valid_ca=valid_ca,
                                         valid_ra=valid_ra,
                                         batch_size=batch_size,
                                         verbose=verbose)
    # Find highly variable
    with loompy.connect(filename=loom_file, mode='r') as ds:
        feat_labels = ds.ra[id_attr][row_idx]
        feat_idx = pd.Series(np.zeros((ds.shape[0])),
                             index=ds.ra[id_attr])
    if measure.lower() == 'sd' or measure.lower() == 'std':
        tmp_var = pd.Series(my_std, index=feat_labels)
    elif measure.lower() == 'vmr':
        my_var = my_std ** 2
        my_vmr = (my_var + 1) / (my_mean + 1)
        tmp_var = pd.Series(my_vmr, index=feat_labels)
    elif measure.lower() == 'cv':
        my_cv = (my_std + 1) / (my_mean + 1)
        tmp_var = pd.Series(my_cv, index=feat_labels)
    else:
        err_msg = 'Unsupported measure value ({})'.format(measure)
        if verbose:
            ih_log.error(err_msg)
        raise ValueError(err_msg)
    # Get top n variable features
    n_feat = min(n_feat, tmp_var.shape[0])
    hvf = tmp_var.sort_values(ascending=False).head(n_feat).index.values
    feat_idx.loc[hvf] = 1
    if out_attr is None:
        out_attr = 'hvf_nfeat_{}'.format(n_feat)
    with loompy.connect(loom_file) as ds:
        ds.ra[out_attr] = feat_idx.values.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        ih_log.info(
            'Found {0} variable features in {1:.2f} {2}'.format(hvf.shape[0],
                                                                time_run,
                                                                time_fmt))


def prep_for_common(loom_file,
                    id_attr='Accession',
                    valid_ra=None,
                    remove_version=False):
    """
    Generates objects for find_common_features

    Args:
        loom_file (str): Path to loom file
        id_attr (str): Attribute specifying unique feature IDs
        remove_version (bool): Remove GENCODE gene versions from IDs
        valid_ra (str): Optional, attribute that specifies desired features

    Returns:
        features (ndarray): Array of unique feature IDs
    """
    valid_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                          attr=valid_ra,
                                          columns=False,
                                          as_bool=True,
                                          inverse=False)
    with loompy.connect(filename=loom_file, mode='r') as ds:
        features = ds.ra[id_attr][valid_idx]
        if remove_version:
            features = general_utils.remove_gene_version(gene_ids=features)
    return features


def add_common_features(loom_file,
                        id_attr,
                        common_features,
                        out_attr,
                        remove_version=False):
    """
    Adds index of common features to loom file (run with find_common_features)

    Args:
        loom_file (str): Path to loom file
        id_attr (str): Name of attribute specifying unique feature IDs
        common_features (ndarray): Array of common features
        out_attr (str): Name of output attribute specifying common features

        remove_version (bool): If true remove version ID
            Anything after the first period is dropped
            Useful for GENCODE gene IDs
    """
    # Make logical index of desired features
    feat_ids = prep_for_common(loom_file=loom_file,
                               id_attr=id_attr,
                               remove_version=remove_version,
                               valid_ra=None)
    with loompy.connect(filename=loom_file) as ds:
        logical_idx = pd.Series(data=np.zeros((ds.shape[0],),
                                              dtype=int),
                                index=feat_ids,
                                dtype=int)
        logical_idx.loc[common_features] = 1
        ds.ra[out_attr] = logical_idx.values


def find_common_features(loom_x,
                         loom_y,
                         out_attr,
                         feature_id_x='Accession',
                         feature_id_y='Accession',
                         valid_ra_x=None,
                         valid_ra_y=None,
                         remove_version=False,
                         verbose=False):
    """
    Identifies common features between two loom files

    Args:
        loom_x (str): Path to first loom file
        loom_y (str): Path to second loom file
        out_attr (str): Name of output attribute indicating common IDs
            Will be a boolean array indicating IDs in feature_id_x/feature_id_y
        feature_id_x (str): Specifies attribute containing feature IDs
        feature_id_y (str): Specifies attribute containing feature IDs
        valid_ra_x (str): Optional, attribute that specifies desired features
        valid_ra_y (str): Optional, attribute that specifies desired features
        remove_version (bool): If true remove versioning
            Anything after the first period is dropped
            Useful for GENCODE gene IDs
        verbose (bool): If true, print logging messages
    """
    if verbose:
        ih_log.info('Finding common features')
    # Get features
    feat_x = prep_for_common(loom_file=loom_x,
                             id_attr=feature_id_x,
                             valid_ra=valid_ra_x,
                             remove_version=remove_version)
    feat_y = prep_for_common(loom_file=loom_y,
                             id_attr=feature_id_y,
                             valid_ra=valid_ra_y,
                             remove_version=remove_version)
    # Find common features
    feats = [feat_x, feat_y]
    common_feat = functools.reduce(np.intersect1d, feats)
    if common_feat.shape[0] == 0:
        err_msg = 'Could not identify any common features'
        if verbose:
            ih_log.error(err_msg)
        raise RuntimeError(err_msg)
    # Add indices
    add_common_features(loom_file=loom_x,
                        id_attr=feature_id_x,
                        common_features=common_feat,
                        out_attr=out_attr,
                        remove_version=True)
    add_common_features(loom_file=loom_y,
                        id_attr=feature_id_y,
                        common_features=common_feat,
                        out_attr=out_attr,
                        remove_version=True)
    if verbose:
        log_msg = ('Found {0} features in common ' +
                   '({1:.2f}% of features in {2}, ' +
                   '{3:.2f}% of features in {4})')
        num_comm = common_feat.shape[0]
        ih_log.info(log_msg.format(num_comm,
                                   loom_utils.get_pct(loom_file=loom_x,
                                                      num_val=num_comm,
                                                      columns=False),
                                   loom_x,
                                   loom_utils.get_pct(loom_file=loom_y,
                                                      num_val=num_comm,
                                                      columns=False),
                                   loom_y))


def normalize_adj(adj_mtx,
                  axis,
                  offset=1e-5):
    """
    Normalizes an adjacency matrix by its mean along an axis

    Args:
        adj_mtx (sparse matrix): Adjacency matrix
        axis (str/int): Axis to normalize along
            0 (int): Normalize along columns
            1 (int): Normalize along rows
            both (str): Normalize along columns, followed by rows
            None: Returns adj_mtx
        offset (float/int): Offset to avoid divide by zero errors

    Returns:
        norm_adj (sparse matrix): Normalized adjacency matrix
    """
    if axis == 0 or axis == 1:
        diags = sparse.diags(1 / (adj_mtx.sum(axis=axis) + offset).A.ravel())
        norm_adj = diags.dot(adj_mtx)
    elif axis == 'both':
        diags = sparse.diags(1 / (adj_mtx.sum(axis=0) + offset).A.ravel())
        norm_adj = diags.dot(adj_mtx)
        diags = sparse.diags(1 / (adj_mtx.sum(axis=1) + offset).A.ravel())
        norm_adj = diags.dot(norm_adj)
    elif axis is None:
        norm_adj = adj_mtx
    else:
        raise ValueError('Unsupported value for axis {}'.format(axis))
    return norm_adj


def update_markov_values(coeff,
                         self_index,
                         other_index,
                         k,
                         dist_vals,
                         idx_vals):
    """
    Updates distances and indices for MNNs

    Args:
        coeff (DataFrame): Correlation coefficients
        self_index (ndarray): Indices for data of interest
        other_index (ndarray): Indices for correlated data
        k (int): Number of mutual nearest neighbors
        dist_vals (ndarray): Distances for MNNs
        idx_vals (ndarray): Indices for MNNs

    Returns:
        dist_vals (ndarray): Updated distances for MNNs
        idx_vals (ndarray): Updated indices for MNNs
    """
    if coeff.shape[1] < k:
        self_k = coeff.shape[1]
        knn = np.ones((coeff.shape[0], coeff.shape[1]),
                      dtype=bool)
    else:
        self_k = k
        knn = ((-coeff).rank(axis=1, method='first') <= k).values.astype(bool)
    new_idx = other_index[np.where(knn)[1]]
    new_dist = coeff.values[knn]
    new_idx = np.reshape(new_idx,
                         newshape=(coeff.shape[0], self_k))
    new_dist = np.reshape(new_dist,
                          newshape=(coeff.shape[0], self_k))
    old_dist = dist_vals[self_index, :]
    old_idx = idx_vals[self_index, :]
    comb_dist = np.hstack([old_dist, new_dist])
    comb_idx = np.hstack([old_idx, new_idx])
    knn_comb = (pd.DataFrame(-comb_dist).rank(axis=1,
                                              method='first') <= k)
    knn_comb = knn_comb.values.astype(bool)
    comb_dist = comb_dist[knn_comb]
    comb_idx = comb_idx[knn_comb]
    comb_dist = np.reshape(comb_dist,
                           newshape=(coeff.shape[0], k))
    comb_idx = np.reshape(comb_idx,
                          newshape=(coeff.shape[0], k))
    idx_vals[self_index, :] = comb_idx
    dist_vals[self_index, :] = comb_dist
    return dist_vals, idx_vals


def get_normalized_dist(dat_x,
                        dat_y,
                        metric="euclidean"):
    """
    Calculates distances between each element in dat_x and dat_y
    based on the given metric

    Args:
        dat_x (ndarray): Values to be correlated
        dat_y (ndarray): Values to be correlated
        metric (str): Metric for calculating distances
            euclidean
            manhattan
            cosine
    Returns:
        dist (DataFrame): Distance between elements

    """
    # Get number of features
    if dat_x.shape[1] != dat_y.shape[1]:
        raise ValueError('dimension mismatch')
    dist = pairwise_distances(dat_x, dat_y, metric=metric)
    return pd.DataFrame(dist)


@jit
def generate_coefficients(dat_x,
                          dat_y):
    """
    Calculates correlation coefficients

    Args:
        dat_x (ndarray): Values to be correlated
        dat_y (ndarray): Values to be correlated

    Returns:
        coeff (DataFrame): Correlation coefficients

    Based on dbliss' answer on StackOverflow
    https://stackoverflow.com/questions/30143417/...
    computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
    """
    # Get number of features
    if dat_x.shape[1] == dat_y.shape[1]:
        n = dat_x.shape[1]
    else:
        raise ValueError('dimension mismatch')
    # Calculate coefficients

    mean_x = dat_x.mean(axis=1)
    mean_y = dat_y.mean(axis=1)
    std_x = dat_x.std(axis=1,
                      ddof=n - 1)
    std_y = dat_y.std(axis=1,
                      ddof=n - 1)
    cov = np.dot(dat_x, dat_y.T) - n * np.dot(mean_x[:, np.newaxis],
                                              mean_y[np.newaxis, :])
    coeff = cov / np.dot(std_x[:, np.newaxis],
                         std_y[np.newaxis, :])
    coeff = pd.DataFrame(coeff)
    return coeff


def generate_distances(loom_x,
                       observed_x,
                       dist_x,
                       idx_x,
                       max_k_x,
                       loom_y,
                       observed_y,
                       dist_y,
                       idx_y,
                       max_k_y,
                       direction,
                       feature_id_x,
                       feature_id_y,
                       metric="euclidian",
                       valid_ca_x=None,
                       ra_x=None,
                       valid_ca_y=None,
                       ra_y=None,
                       batch_x=512,
                       batch_y=512,
                       remove_version=False,
                       verbose=False):
    """
    Adds distnace matrices between two modalites to loom files based 
    on the given distance metrics

    Args:
        loom_x (str): Path to loom file
        observed_x (str): Name of layer containing counts
        dist_x (str): Name of distance attribute for correlations
        idx_x (str): Name of index attribute for correlations
        max_k_x (int): Maximum k needed
        loom_y (str): Path to loom file
        observed_y (str): Name of layer containing counts
        dist_y (str): Name of distance attribute for correlations
        idx_y (str): Name of index attribute for correlations
        max_k_y (int): Maximum k needed
        direction (str): Direction of expected correlation
            negative/- or positive/+
        feature_id_x (str): Attribute containing feature IDs
        feature_id_y (str): Attribute containing feature IDs
        metric (str): Metric for calculating distances
            euclidean
            manhattan
            cosine
            correlation
        valid_ca_x (str): Name of column attribute to restrict counts by
        ra_x (str): Name of row attribute to restrict counts by
        valid_ca_y (str): Name of column attribute to restrict counts by
        ra_y (str): Name of row attribute to restrict counts by
        batch_x (int): Chunk size for batches
        batch_y (int): Chunk size for batches
        remove_version (bool): If true, remove gene version number
        verbose (bool): Print logging messages
    """
    if verbose:
        ih_log.info('Generating {} distance matrix'.format(metric))
        t0 = time.time()
    if metric == "correlation":
        generate_correlations(loom_x=loom_x,
                              observed_x=observed_x,
                              corr_dist_x=dist_x,
                              corr_idx_x=idx_x,
                              max_k_x=max_k_x,
                              loom_y=loom_y,
                              observed_y=observed_y,
                              corr_dist_y=dist_y,
                              corr_idx_y=idx_y,
                              max_k_y=max_k_y,
                              direction=direction,
                              feature_id_x=feature_id_x,
                              feature_id_y=feature_id_y,
                              valid_ca_x=valid_ca_x,
                              ra_x=ra_x,
                              valid_ca_y=valid_ca_y,
                              ra_y=ra_y,
                              batch_x=batch_x,
                              batch_y=batch_y,
                              remove_version=remove_version,
                              verbose=False)
        return

    layers_x = loom_utils.make_layer_list(observed_x)
    col_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=valid_ca_x,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=ra_x,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    layers_y = loom_utils.make_layer_list(observed_y)
    col_y = loom_utils.get_attr_index(loom_file=loom_y,
                                      attr=valid_ca_y,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_y = loom_utils.get_attr_index(loom_file=loom_y,
                                      attr=ra_y,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    # Prepare for correlation matrix
    with loompy.connect(filename=loom_x) as ds_x:
        with loompy.connect(filename=loom_y) as ds_y:
            num_x = ds_x.shape[1]
            num_y = ds_y.shape[1]
            dist_x_ = general_utils.make_nan_array(num_rows=num_x,
                                                   num_cols=max_k_x)
            idx_x_ = general_utils.make_nan_array(num_rows=num_x,
                                                  num_cols=max_k_x)
            dist_y_ = general_utils.make_nan_array(num_rows=num_y,
                                                   num_cols=max_k_y)
            idx_y_ = general_utils.make_nan_array(num_rows=num_y,
                                                  num_cols=max_k_y)
            x_feat = ds_x.ra[feature_id_x][row_x]
            y_feat = ds_y.ra[feature_id_y][row_y]
            if remove_version:
                x_feat = general_utils.remove_gene_version(x_feat)
                y_feat = general_utils.remove_gene_version(y_feat)
    # Loop and make correlations
    with loompy.connect(filename=loom_x, mode='r') as ds_x:
        for (_, sel_x, dat_x) in ds_x.scan(axis=1,
                                           items=col_x,
                                           layers=layers_x,
                                           batch_size=batch_x):
            # Get data
            dat_x = dat_x.layers[observed_x][row_x, :].T
            dat_x = (dat_x - dat_x.mean()) / dat_x.std()
            if direction == '+' or direction == 'positive':
                pass
            elif direction == '-' or direction == 'negative':
                dat_x = -1 * dat_x
            else:
                err_msg = 'Unsupported direction value ({})'.format(direction)
                ih_log.error(err_msg)
                raise ValueError(err_msg)
            with loompy.connect(filename=loom_y, mode='r') as ds_y:
                for (_, sel_y, dat_y) in ds_y.scan(axis=1,
                                                   items=col_y,
                                                   layers=layers_y,
                                                   batch_size=batch_y):
                    dat_y = dat_y.layers[observed_y][row_y, :].T
                    dat_y = pd.DataFrame((dat_y - dat_y.mean()) / dat_y.std())
                    dat_y.columns = y_feat
                    dat_y = dat_y.loc[:, x_feat]
                    if dat_y.isnull().any().any():
                        err_msg = 'Feature mismatch for correlations'
                        if verbose:
                            ih_log.error(err_msg)
                        raise ValueError(err_msg)
                    dat_y = dat_y.values
                    coeff = get_normalized_dist(dat_x,
                                                dat_y,
                                                metric=metric)
                    dist_x_, idx_x_ = update_markov_values(coeff=coeff,
                                                           self_index=sel_x,
                                                           other_index=sel_y,
                                                           k=max_k_x,
                                                           dist_vals=dist_x_,
                                                           idx_vals=idx_x_)
                    dist_y_, idx_y_ = update_markov_values(coeff=coeff.T,
                                                           self_index=sel_y,
                                                           other_index=sel_x,
                                                           k=max_k_y,
                                                           dist_vals=dist_y_,
                                                           idx_vals=idx_y_)

                    del coeff
                    gc.collect()
    # Add data to files
    with loompy.connect(filename=loom_x) as ds:
        ds.ca[dist_x] = dist_x_
        ds.ca[idx_x] = idx_x_.astype(int)
    with loompy.connect(filename=loom_y) as ds:
        ds.ca[dist_y] = dist_y_
        ds.ca[idx_y] = idx_y_.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        ih_log.info(
            'Generated {0} distance in {1:.2f} {2}'.format(metric, time_run,
                                                           time_fmt))


def generate_correlations(loom_x,
                          observed_x,
                          corr_dist_x,
                          corr_idx_x,
                          max_k_x,
                          loom_y,
                          observed_y,
                          corr_dist_y,
                          corr_idx_y,
                          max_k_y,
                          direction,
                          feature_id_x,
                          feature_id_y,
                          valid_ca_x=None,
                          ra_x=None,
                          valid_ca_y=None,
                          ra_y=None,
                          batch_x=512,
                          batch_y=512,
                          remove_version=False,
                          verbose=False):
    """
    Adds correlation matrices between two modalites to loom files

    Args:
        loom_x (str): Path to loom file
        observed_x (str): Name of layer containing counts
        corr_dist_x (str): Name of distance attribute for correlations
        corr_idx_x (str): Name of index attribute for correlations
        max_k_x (int): Maximum k needed
        loom_y (str): Path to loom file
        observed_y (str): Name of layer containing counts
        corr_dist_y (str): Name of distance attribute for correlations
        corr_idx_y (str): Name of index attribute for correlations
        max_k_y (int): Maximum k needed
        direction (str): Direction of expected correlation
            negative/- or positive/+
        feature_id_x (str): Attribute containing feature IDs
        feature_id_y (str): Attribute containing feature IDs
        valid_ca_x (str): Name of column attribute to restrict counts by
        ra_x (str): Name of row attribute to restrict counts by
        valid_ca_y (str): Name of column attribute to restrict counts by
        ra_y (str): Name of row attribute to restrict counts by
        batch_x (int): Chunk size for batches
        batch_y (int): Chunk size for batches
        remove_version (bool): If true, remove gene version number
        verbose (bool): Print logging messages
    """
    if verbose:
        ih_log.info('Generating correlation matrix')
        t0 = time.time()
    layers_x = loom_utils.make_layer_list(observed_x)
    col_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=valid_ca_x,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=ra_x,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    layers_y = loom_utils.make_layer_list(observed_y)
    col_y = loom_utils.get_attr_index(loom_file=loom_y,
                                      attr=valid_ca_y,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_y = loom_utils.get_attr_index(loom_file=loom_y,
                                      attr=ra_y,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    # Prepare for correlation matrix
    with loompy.connect(filename=loom_x) as ds_x:
        with loompy.connect(filename=loom_y) as ds_y:
            num_x = ds_x.shape[1]
            num_y = ds_y.shape[1]
            dist_x = general_utils.make_nan_array(num_rows=num_x,
                                                  num_cols=max_k_x)
            idx_x = general_utils.make_nan_array(num_rows=num_x,
                                                 num_cols=max_k_x)
            dist_y = general_utils.make_nan_array(num_rows=num_y,
                                                  num_cols=max_k_y)
            idx_y = general_utils.make_nan_array(num_rows=num_y,
                                                 num_cols=max_k_y)
            x_feat = ds_x.ra[feature_id_x][row_x]
            y_feat = ds_y.ra[feature_id_y][row_y]
            if remove_version:
                x_feat = general_utils.remove_gene_version(x_feat)
                y_feat = general_utils.remove_gene_version(y_feat)
    # Loop and make correlations
    with loompy.connect(filename=loom_x, mode='r') as ds_x:
        for (_, sel_x, dat_x) in ds_x.scan(axis=1,
                                           items=col_x,
                                           layers=layers_x,
                                           batch_size=batch_x):
            # Get data
            dat_x = dat_x.layers[observed_x][row_x, :].T
            dat_x = pd.DataFrame(dat_x).rank(pct=True,
                                             method='first',
                                             axis=1).values
            if direction == '+' or direction == 'positive':
                pass
            elif direction == '-' or direction == 'negative':
                dat_x = 1 - dat_x
            else:
                raise ValueError('Unsupported direction value')
            with loompy.connect(filename=loom_y, mode='r') as ds_y:
                for (_, sel_y, dat_y) in ds_y.scan(axis=1,
                                                   items=col_y,
                                                   layers=layers_y,
                                                   batch_size=batch_y):
                    dat_y = dat_y.layers[observed_y][row_y, :].T
                    dat_y = pd.DataFrame(dat_y).rank(pct=True,
                                                     method='first',
                                                     axis=1)
                    dat_y.columns = y_feat
                    dat_y = dat_y.loc[:, x_feat]
                    if dat_y.isnull().any().any():
                        err_msg = 'Feature mismatch for correlations'
                        if verbose:
                            ih_log.error(err_msg)
                        raise ValueError(err_msg)
                    dat_y = dat_y.values
                    coeff = generate_coefficients(dat_x,
                                                  dat_y)
                    dist_x, idx_x = update_markov_values(coeff=coeff,
                                                         self_index=sel_x,
                                                         other_index=sel_y,
                                                         k=max_k_x,
                                                         dist_vals=dist_x,
                                                         idx_vals=idx_x)
                    dist_y, idx_y = update_markov_values(coeff=coeff.T,
                                                         self_index=sel_y,
                                                         other_index=sel_x,
                                                         k=max_k_y,
                                                         dist_vals=dist_y,
                                                         idx_vals=idx_y)

                    del coeff
                    gc.collect()
    # Add data to files
    with loompy.connect(filename=loom_x) as ds:
        ds.ca[corr_dist_x] = dist_x
        ds.ca[corr_idx_x] = idx_x.astype(int)
    with loompy.connect(filename=loom_y) as ds:
        ds.ca[corr_dist_y] = dist_y
        ds.ca[corr_idx_y] = idx_y.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        ih_log.info(
            'Generated correlations in {0:.2f} {1}'.format(time_run, time_fmt))


@jit
def csls_dist(distances, 
              r_x,
              r_y,
              avg_weight = .1):

    """
    finds adjusted csls distances

    Args:
        
        observed_x (str): Name of layer containing counts
        r_x (numpy array): Array of average cross modality distances
            for modality x
        r_y (numpy array): Array of average cross modality distances
            for modality y

    """
    csls_distances = 2 * distances
    
    for pair in zip(np.arange(distances.shape[0]), np.arange(distances.shape[1])):
        csls_distances[pair[0], pair[1]] = csls_distances[pair[0], pair[1]]\
                                        - avg_weight * (r_x[pair[0]] + r_y[pair[1]])

    return csls_distances
    
    
def generate_csls_distance(loom_x,
                           observed_x,
                           dist_x,
                           idx_x,
                           max_k_x,
                           loom_y,
                           observed_y,
                           dist_y,
                           idx_y,
                           max_k_y,
                           direction,
                           feature_id_x,
                           feature_id_y,
                           check_k = 20,
                           avg_weight = .1,
                           metric="correlation",
                           valid_ca_x=None,
                           ra_x=None,
                           valid_ca_y=None,
                           ra_y=None,
                           batch_x=512,
                           batch_y=512,
                           remove_version=False,
                           verbose=False):
    """
    Adds csls adjusted distances between two modalites to loom files

    Args:
        loom_x (str): Path to loom file
        observed_x (str): Name of layer containing counts
        corr_dist_x (str): Name of distance attribute for correlations
        corr_idx_x (str): Name of index attribute for correlations
        max_k_x (int): Maximum k needed
        loom_y (str): Path to loom file
        observed_y (str): Name of layer containing counts
        corr_dist_y (str): Name of distance attribute for correlations
        corr_idx_y (str): Name of index attribute for correlations
        max_k_y (int): Maximum k needed
        direction (str): Direction of expected correlation
            negative/- or positive/+
        feature_id_x (str): Attribute containing feature IDs
        feature_id_y (str): Attribute containing feature IDs
        check_k (int) : a number of nearest neighbors used to find average
            cross modality distances
        avg_weight (float) : a wighting factor which decides how much
            cells benefit from having a large avergae distance
        metric (str) : how distances bewteen cells are measured. 
            correlation by default. 
        valid_ca_x (str): Name of column attribute to restrict counts by
        ra_x (str): Name of row attribute to restrict counts by
        valid_ca_y (str): Name of column attribute to restrict counts by
        ra_y (str): Name of row attribute to restrict counts by
        batch_x (int): Chunk size for batches
        batch_y (int): Chunk size for batches
        remove_version (bool): If true, remove gene version number
        verbose (bool): Print logging messages
    """
    if verbose:
        ih_log.info('Generating csls distance matrix {}'.format(avg_weight))
        t0 = time.time()
    layers_x = loom_utils.make_layer_list(observed_x)
    col_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=valid_ca_x,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=ra_x,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    layers_y = loom_utils.make_layer_list(observed_y)
    col_y = loom_utils.get_attr_index(loom_file=loom_y,
                                      attr=valid_ca_y,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_y = loom_utils.get_attr_index(loom_file=loom_y,
                                      attr=ra_y,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)                
    
    with loompy.connect(filename=loom_x) as ds_x:
        with loompy.connect(filename=loom_y) as ds_y:
            num_x = ds_x.shape[1]
            num_y = ds_y.shape[1]
            dist_x_ = general_utils.make_nan_array(num_rows=num_x,
                                                   num_cols=max_k_x)
            idx_x_ = general_utils.make_nan_array(num_rows=num_x,
                                                  num_cols=max_k_x)
            dist_y_ = general_utils.make_nan_array(num_rows=num_y,
                                                   num_cols=max_k_y)
            idx_y_ = general_utils.make_nan_array(num_rows=num_y,
                                                  num_cols=max_k_y)
            x_feat = ds_x.ra[feature_id_x][row_x]
            y_feat = ds_y.ra[feature_id_y][row_y]
            if remove_version:
                x_feat = general_utils.remove_gene_version(x_feat)
                y_feat = general_utils.remove_gene_version(y_feat)
             
    # Loop and make correlations
    with loompy.connect(filename=loom_x, mode='r') as ds_x:
        for (_, sel_x, dat_x) in ds_x.scan(axis=1,
                                           items=col_x,
                                           layers=layers_x,
                                           batch_size=batch_x):
            # Get data
            r_x = np.array(dat_x.ca[dist_x][:, :check_k]).mean(axis=1)

            dat_x = dat_x.layers[observed_x][row_x, :].T
            
            if metric == "correlation": 
                dat_x = pd.DataFrame(dat_x).rank(pct=True,
                                                 method='first',
                                                 axis=1).values
                if direction == '+' or direction == 'positive':
                    pass
                elif direction == '-' or direction == 'negative':
                    dat_x = 1 - dat_x
                else:
                    raise ValueError('Unsupported direction value')
            else: 
                dat_x = (dat_x - dat_x.mean()) / dat_x.std()

                if direction == '+' or direction == 'positive':
                    pass
                elif direction == '-' or direction == 'negative':
                    dat_x = -1 * dat_x
                else:
                    err_msg = 'Unsupported direction value ({})'.format(direction)
                    ih_log.error(err_msg)
                    raise ValueError(err_msg)
            with loompy.connect(filename=loom_y, mode='r') as ds_y:
                for (_, sel_y, dat_y) in ds_y.scan(axis=1,
                                                   items=col_y,
                                                   layers=layers_y,
                                                   batch_size=batch_y):
                    
                    r_y = np.array(dat_y.ca[dist_y][:, :check_k]).mean(axis=1)
                    dat_y = dat_y.layers[observed_y][row_y, :].T
                    dat_y = pd.DataFrame((dat_y - dat_y.mean()) / dat_y.std())
                    dat_y.columns = y_feat
                    dat_y = dat_y.loc[:, x_feat]
                    
                    if dat_y.isnull().any().any():
                        err_msg = 'Feature mismatch for correlations'
                        if verbose:
                            ih_log.error(err_msg)
                        raise ValueError(err_msg)
                    dat_y = dat_y.values
                    if metric == "correlation":
                        coeff = generate_coefficients(dat_x,
                                                      dat_y)
                    else:
                        coeff = get_normalized_dist(dat_x,
                                                dat_y,
                                                metric=metric)
                    
                    coeff = csls_dist(distances = coeff.values,
                                      r_x = r_x,
                                      r_y = r_y,
                                      avg_weight = avg_weight)
                    
                    dist_x_, idx_x_ = update_markov_values(coeff=pd.DataFrame(coeff),
                                                           self_index=sel_x,
                                                           other_index=sel_y,
                                                           k=max_k_x,
                                                           dist_vals=dist_x_,
                                                           idx_vals=idx_x_)
                    dist_y_, idx_y_ = update_markov_values(coeff=pd.DataFrame(coeff.T),
                                                           self_index=sel_y,
                                                           other_index=sel_x,
                                                           k=max_k_y,
                                                           dist_vals=dist_y_,
                                                           idx_vals=idx_y_)

                    del coeff
                    gc.collect()
    # Add data to files
    with loompy.connect(filename=loom_x) as ds:
        ds.ca[dist_x] = dist_x_
        ds.ca[idx_x] = idx_x_.astype(int)
    with loompy.connect(filename=loom_y) as ds:
        ds.ca[dist_y] = dist_y_
        ds.ca[idx_y] = idx_y_.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        ih_log.info(
            'Generated {0} distance in {1:.2f} {2}'.format(metric, time_run,
                                                           time_fmt))
def multimodal_adjacency(distance_arr,
                         neighbor_arr,
                         num_col,
                         new_k=None):
    """
    Generates a sparse adjacency matrix from specified distances and neighbors
    Optionally, restricts to a new k nearest neighbors

    Args:
        distance_arr (ndarray): Distances between elements
        neighbor_arr (ndarray): Index of neighbors
        num_col (int): Number of output column in adjacency matrix
        new_k (int): Optional, restrict to this k

    Returns 
        A (sparse matrix): Adjacency matrix
    """
    if new_k is None:
        new_k = distance_arr.shape[1]
    if distance_arr.shape[1] != neighbor_arr.shape[1]:
        raise ValueError('Neighbors and distances must have same k!')
    if distance_arr.shape[1] < new_k:
        raise ValueError('new_k must be less than the current k')
    tmp = pd.DataFrame(distance_arr)
    new_k = int(new_k)
    knn = ((-tmp).rank(axis=1, method='first') <= new_k).values.astype(bool)
    if np.unique(np.sum(knn, axis=1)).shape[0] != 1:
        raise ValueError('k is inappropriate for data')
    a = sparse.csr_matrix(
        (np.ones((int(neighbor_arr.shape[0] * new_k),), dtype=int),
         (np.where(knn)[0], neighbor_arr[knn])),
        (neighbor_arr.shape[0], num_col))
    return a


@jit
def constrained_knn_search(distance_arr,
                           neighbor_arr,
                           num_other,
                           j_max,
                           saturated=None,
                           k=10):
    """
    Gets the top K neighbors in the other modality using a
    constrained search

    Args:
        distance_arr (ndarray): Distances between elements
        neighbor_arr (ndarray): Index of neighbors
        num_other (int): Number of output column in adjacency matrix
        j_max (int): the maximum number of neighbors cells in the 
            other modality can make with these cells
        saturated (array-like): cells in the other modality that are
            removed due to saturation constraint
        constraint relaxation(float): a ratio determining the number of
             neigbors that can be formed by cells in the other modality. 
             Increasing it means neighbors can be distributed more
             unevenly among cells, one means each cell is used equally.

        k (int): The number of nearest neibors to restrict to

    Returns
        knn (sparse matrix): the top k constrained neighbors for each cell

    """

    num_arr = neighbor_arr.shape[0]
    knn = np.zeros((num_arr, num_other))
    random_order = np.random.choice(range(neighbor_arr.shape[0]),
                                    size=neighbor_arr.shape[0],
                                    replace=False)
    if saturated is None:
        saturated = []
    else:
        to_drop = np.where(np.isin(neighbor_arr, saturated))
        distance_arr[to_drop[0], to_drop[1]] = np.inf
    # POSSIBLE BUG: neighbor is based on all cells not just valid cells
    for i in np.arange(k):  # get the ith nearest neighbor in the other dataset
        for cell in random_order:  # loop over all cells in rows
            j = distance_arr[cell, :].argmin()
            neighbor_idx = neighbor_arr[cell, j]
            knn[cell, neighbor_idx] = 1
            distance_arr[cell, j] = np.inf
            if np.sum(knn[:, neighbor_idx]) > j_max:
                to_drop = np.where(neighbor_arr == neighbor_idx)
                distance_arr[to_drop[0], to_drop[1]] = np.inf
                saturated.append(neighbor_idx)
    return sparse.lil_matrix(knn), saturated


def gen_impute_adj(loom_file,
                   neighbor_attr,
                   distance_attr,
                   k,
                   self_idx,
                   other_idx,
                   batch_size=512):
    """
    Generates adjacency matrix from a loom file
        Subfunction used in get_markov_impute

    Args:
        loom_file (str): Path to loom file
        neighbor_attr (str): Attribute specifying neighbors
        distance_attr (str): Attribute specifying distances
        k (int): k value for mutual nearest neighbors
        self_idx (ndarray): Rows in corr to include
        other_idx (ndarray) Columns in corr to include
        batch_size (int): Size of chunks

    Returns
        adj (sparse matrix): Adjacency matrix with k nearest 
                             neighbors of self in other.
    """
    adj = []
    num_other = other_idx.shape[0]
    num_self = self_idx.shape[0]
    with loompy.connect(filename=loom_file, mode='r') as ds:
        if num_self != ds.shape[1]:
            raise ValueError('Index does not match dimensions')
        for (_, selection, view) in ds.scan(axis=1,
                                            layers=[''],
                                            items=self_idx,
                                            batch_size=batch_size):
            adj.append(multimodal_adjacency(distance_arr=view.ca[distance_attr],
                                            neighbor_arr=view.ca[neighbor_attr],
                                            num_col=num_other,
                                            new_k=k))
    # Make matrices
    adj = sparse.vstack(adj)
    adj = adj.tocsc()[:, other_idx].tocsr()
    return adj


def gen_impute_knn(loom_target,
                   loom_source,
                   neighbor_attr,
                   distance_attr,
                   valid_target,
                   valid_source,
                   k=10,
                   constraint_relaxation=1.1,
                   offset=1e-5,
                   batch_size=512,
                   verbose=False):
    """
    Generates a restricted knn adjacentcy matrix from a loom file

    Args:
        loom_target (str): Path to loom file for target modality
        loom_source (str): Path to loom file for source modality
        neighbor_attr (str): Attribute specifying neighbors
        distance_attr (str): Attribute specifying distances
        k (int): The number of nearest neibors to restrict to
        valid_target (str): Attribute specifying cells to include in target
        valid_source (str): Attribute specifying cells to include in source
        constraint_relaxation(float): a ratio determining the number of
                                    neigbors that can be formed by 
                                    cells in the other modality. 
                                    Increasing it means neighbors can
                                    be distributed more unevenly among
                                    cells, one means each cell is used 
                                    equally.
        offset (float): Offset for normalization of adjacency matrix
        batch_size (int): Size of chunks
        verbose (bool): Print logging messages

     Returns:
        adj (sparse matrix): Adjacency matrix with k nearest 
                             neighbors of self in other.
    """

    if verbose:
        log_message = 'Generating restricted KNN adjacency matrix k={}'.format(
            k)
        ih_log.info(log_message)
    self_idx = loom_utils.get_attr_index(loom_file=loom_target,
                                         attr=valid_target,
                                         columns=True,
                                         as_bool=True,
                                         inverse=False)
    other_idx = loom_utils.get_attr_index(loom_file=loom_source,
                                          attr=valid_source,
                                          columns=True,
                                          as_bool=True,
                                          inverse=False)
    num_other = other_idx.astype(int).sum()
    constraint = np.ceil(
        (k * constraint_relaxation * self_idx.astype(int).sum()) / num_other)
    if verbose:
        log_message = 'Other cells are constrained to {} neighbors'.format(
            constraint)
        ih_log.info(log_message)
    adj = []
    with loompy.connect(filename=loom_target, mode='r') as ds:
        saturated = None
        for (_, selection, view) in ds.scan(axis=1,
                                            layers=[''],
                                            items=self_idx,
                                            batch_size=batch_size):
            knn, saturated = constrained_knn_search(
                distance_arr=view.ca[distance_attr],
                neighbor_arr=view.ca[neighbor_attr],
                saturated=saturated,
                num_other=other_idx.shape[0],
                j_max=constraint,
                k=k)
            adj.append(knn)
    # Make matrices
    adj = sparse.vstack(adj)
    adj = adj.tocsc()[:, other_idx].tocsr()

    # Normalize
    adj = normalize_adj(adj_mtx=adj,
                        axis=1,
                        offset=offset)
    return adj


def get_markov_impute(loom_target,
                      loom_source,
                      valid_target,
                      valid_source,
                      neighbor_target,
                      neighbor_source,
                      distance_target,
                      distance_source,
                      k_src_tar,
                      k_tar_src,
                      offset=1e-5,
                      batch_target=512,
                      batch_source=512,
                      verbose=False):
    """
    Generates mutual nearest neighbors Markov for imputation

    Args:
        loom_target (str): Path to loom file for target modality
        loom_source (str): Path to loom file for source modality
        valid_target (str): Attribute specifying cells to include in target
        valid_source (str): Attribute specifying cells to include in source
        neighbor_target (str): Attribute containing neighbor indices
        neighbor_source (str): Attribute containing neighbor indices
        distance_target (str): Attribute containing neighbor distances
        distance_source (str): Attribute containing neighbor distances
        k_src_tar (int): Number of nearest neighbors
        k_tar_src (int): Number of nearest neighbors
        offset (float): Offset for normalization of adjacency matrix
        batch_target (int): Size of batches
        batch_source (int): Size of batches
        verbose (bool): Print logging messages

    Returns:
        w_impute (sparse matrix): Markov matrix for imputation
    """
    if verbose:
        ih_log.info('Generating mutual adjacency matrix')
    cidx_target = loom_utils.get_attr_index(loom_file=loom_target,
                                            attr=valid_target,
                                            columns=True,
                                            as_bool=True,
                                            inverse=False)
    cidx_source = loom_utils.get_attr_index(loom_file=loom_source,
                                            attr=valid_source,
                                            columns=True,
                                            as_bool=True,
                                            inverse=False)

    # Make adjacency matrix
    ax_xy = gen_impute_adj(loom_file=loom_target,
                           neighbor_attr=neighbor_target,
                           distance_attr=distance_target,
                           k=k_tar_src,
                           self_idx=cidx_target,
                           other_idx=cidx_source,
                           batch_size=batch_target)
    ax_yx = gen_impute_adj(loom_file=loom_source,
                           neighbor_attr=neighbor_source,
                           distance_attr=distance_source,
                           k=k_src_tar,
                           self_idx=cidx_source,
                           other_idx=cidx_target,
                           batch_size=batch_source)
    # Generate mutual neighbors adjacency
    w_impute = (ax_xy.multiply(ax_yx.T))
    # Normalize
    w_impute = normalize_adj(adj_mtx=w_impute,
                             axis=1,
                             offset=offset)
    # Get cells
    c_x = len(np.sort(np.unique(w_impute.nonzero()[0])))
    if verbose:
        rec_msg = '{0}: {1} ({2:.2f}%) cells made direct MNNs'
        ih_log.info(rec_msg.format(loom_target,
                                   c_x,
                                   loom_utils.get_pct(loom_file=loom_target,
                                                      num_val=c_x,
                                                      columns=True)))
        k_msg = '{0} had a k of {1}'
        ih_log.info(k_msg.format(loom_target,
                                 k_tar_src))
        ih_log.info(k_msg.format(loom_source,
                                 k_src_tar))
    return w_impute


def gaussian_markov(loom_target,
                    valid_target,
                    mnns,
                    k,
                    ka,
                    epsilon,
                    pca_attr,
                    metric,
                    batch_size):
    """
    Generates Markov for rescuing cells

    Args:
        loom_target (str): Path to loom file for target modality
        valid_target (str): Attribute specifying cells to include
        mnns (str): Index of MNNs in np.where(valid_target)
        k (int): Number of neighbors for rescue
        ka (int): Normalize neighbor distances by the kath cell
        epsilon (float): Noise parameter for Gaussian kernel
        pca_attr (str): Attribute containing PCs
        metric (str): Metric for calculating distances
            euclidean
            manhattan
            cosine
        batch_size (int): Size of batches

    Returns:
        w (sparse matrix): Markov matrix for within-modality rescue

    This code originates from https://github.com/KrishnaswamyLab/MAGIC which is
    covered under a GNU General Public License version 2. The publication
    describing MAGIC is 'MAGIC: A diffusion-based imputation method
    reveals gene-gene interactions in single-cell RNA-sequencing data.' The
    publication was authored by: David van Dijk, Juozas Nainys, Roshan Sharma,
    Pooja Kathail, Ambrose J Carr, Kevin R Moon, Linas Mazutis, Guy Wolf,
    Smita Krishnaswamy, Dana Pe'er. The DOI is https://doi.org/10.1101/111591

    The concept of applying the Gaussian kernel originates from 'Batch effects
    in single-cell RNA sequencing data are corrected by matching mutual nearest
    neighbors' by Laleh Haghverdi, Aaron TL Lun, Michael D Morgan, and John C
    Marioni. It was published in Nature Biotechnology and the DOI is
    https://doi.org/10.1038/nbt.4091.
    """
    # Get neighbors and distances
    cidx_tar = loom_utils.get_attr_index(loom_file=loom_target,
                                         attr=valid_target,
                                         columns=True,
                                         as_bool=False,
                                         inverse=False)
    tot_n = cidx_tar.shape[0]
    distances = []
    indices = []
    with loompy.connect(loom_target) as ds:
        mnn_pcs = ds.ca[pca_attr][cidx_tar, :][mnns, :]
        for (_, selection, view) in ds.scan(items=cidx_tar,
                                            layers=[''],
                                            axis=1,
                                            batch_size=batch_size):
            tmp = pairwise_distances(X=view.ca[pca_attr],
                                     Y=mnn_pcs,
                                     metric=metric)
            knn = (pd.DataFrame(tmp).rank(axis=1, method='first') <= k)
            if np.unique(np.sum(knn, axis=1)).shape[0] != 1:
                raise ValueError('k is inappropriate for data')
            tmp_neighbor = np.reshape(mnns[np.where(knn)[1]],
                                      (selection.shape[0], k))
            tmp_distance = np.reshape(tmp[knn],
                                      (selection.shape[0], k))
            distances.append(tmp_distance)
            indices.append(tmp_neighbor)
    distances = np.vstack(distances)
    indices = np.vstack(indices)
    if ka > 0:
        distances = distances / (np.sort(distances,
                                         axis=1)[:, ka].reshape(-1, 1))
    # Calculate gaussian kernel
    adjs = np.exp(-((distances ** 2) / (epsilon ** 2)))
    # Construct W
    rows = np.repeat(np.arange(tot_n), k)
    cols = np.ravel(indices)
    vals = np.ravel(adjs)
    w = sparse.csr_matrix((vals, (rows, cols)), shape=(tot_n, tot_n))
    # Symmetrize W
    w = w + w.T
    # Normalize W
    divisor = np.ravel(np.repeat(w.sum(axis=1), w.getnnz(axis=1)))
    w.data /= divisor
    return w


def all_markov_self(loom_target,
                    valid_target,
                    loom_source,
                    valid_source,
                    neighbor_target,
                    neighbor_source,
                    distance_target,
                    distance_source,
                    k_src_tar,
                    k_tar_src,
                    k_rescue,
                    ka,
                    epsilon,
                    pca_attr,
                    metric,
                    offset=1e-5,
                    batch_target=512,
                    batch_source=512,
                    verbose=False):
    """
    Generates Markov used for imputation if all cells are included (rescue)

    Args:
        loom_target (str): Path to loom file for target modality
        valid_target (str): Attribute specifying cells to include
        loom_source (str): Path to loom file for source modality
        valid_source (str): Attribute specifying cells to include
        neighbor_target (str): Attribute specifying neighbor indices
        neighbor_source (str): Attribute specifying neighbor indices
        distance_target (str): Attribute specifying distance values
        distance_source (str): Attribute specifying distance values
        k_src_tar (int): Number of nearest neighbors for MNN
        k_tar_src (int): Number of nearest neighbors for MNN
        k_rescue (int): Number of nearest neighbors for rescue
        ka (int): Normalizes distance by kath cell's distance
        epsilon (float): Noise parameter for Gaussian kernel
        pca_attr (str): Attribute containing PCs
        metric (str): Metric for measuring distance
            euclidean
            manhattan
            cosine
        offset (float): Offset for Markov normalization
        batch_target (int): Size of batches
        batch_source (int): Size of batches
        verbose (bool): Print logging message

    Returns:
        w_use (sparse matrix): Markov matrix for imputing data
    """
    # Get w_impute and cells that formed MNNs
    w_impute = get_markov_impute(loom_target=loom_target,
                                 loom_source=loom_source,
                                 valid_target=valid_target,
                                 valid_source=valid_source,
                                 neighbor_target=neighbor_target,
                                 neighbor_source=neighbor_source,
                                 distance_target=distance_target,
                                 distance_source=distance_source,
                                 k_src_tar=k_src_tar,
                                 k_tar_src=k_tar_src,
                                 offset=offset,
                                 batch_target=batch_target,
                                 batch_source=batch_source,
                                 verbose=verbose)
    mnns = np.unique(w_impute.nonzero()[0])
    # Get w_self
    w_self = gaussian_markov(loom_target=loom_target,
                             valid_target=valid_target,
                             mnns=mnns,
                             k=k_rescue,
                             ka=ka,
                             epsilon=epsilon,
                             pca_attr=pca_attr,
                             metric=metric,
                             batch_size=batch_target)
    w_use = w_self.dot(w_impute)
    return w_use


def impute_data(loom_source,
                layer_source,
                id_source,
                cell_source,
                feat_source,
                loom_target,
                layer_target,
                id_target,
                cell_target,
                feat_target,
                mnn_distance_target,
                mnn_distance_source,
                mnn_index_target,
                mnn_index_source,
                k_src_tar,
                k_tar_src,
                k_rescue,
                ka,
                epsilon,
                pca_attr,
                metric,
                neighbor_method='rescue',
                constraint_relaxation=1.1,
                remove_version=False,
                offset=1e-5,
                batch_target=512,
                batch_source=512,
                verbose=False):
    """
    Performs imputation over a list (if provided) of layers imputing values in
    the source modality for the target data.

    Args:
        loom_source (str): Name of loom file that contains observed count data
        layer_source (str/list): Layer(s) containing observed count data
        id_source (str): Row attribute specifying unique feature IDs
        cell_source (str): Column attribute specifying columns to include
        feat_source (str): Row attribute specifying rows to include
        loom_target (str): Name of loom file that will receive imputed counts
        layer_target (str/list): Layer(s) that will contain imputed count data
        id_target (str): Row attribute specifying unique feature IDs
        cell_target (str): Column attribute specifying columns to include
        feat_target (str): Row attribute specifying rows to include
        mnn_distance_target (str): Attribute containing distances for MNNs
        mnn_distance_source (str): Attribute containing distances for MNNs
        mnn_index_target (str): Attribute containing indices for MNNs
        mnn_index_source (str): Attribute containing indices for MNNs
        k_src_tar (int): Number of nearest neighbors for MNNs
        k_tar_src (int): Number of nearest neighbors for MNNs
        k_rescue (int): Number of nearest neighbors for rescue
        ka (int): If rescue, neighbor to normalize by
        epsilon (float): If rescue, epsilon value for Gaussian kernel
        pca_attr (str): If rescue, attribute containing PCs
        metric (str): If rescue, method for calculating distances
            euclidean
            manhattan
            cosine
        neighbor_method (str): How cells are chosen for imputation
            rescue - include cells that did not make MNNs
            mnn - only include cells that made MNNs
            knn - use a restricted knn search to find neighbors
        constraint_relaxation(float): used for knn impuation
                                    a ratio determining the number of
                                    neigbors that can be formed by 
                                    cells in the other modality. 
                                    Increasing it means neighbors can
                                    be distributed more unevenly among
                                    cells, one means each cell is used 
                                    equally.
        remove_version (bool): Remove GENCODE version numbers from IDs
        offset (float): Offset for Markov normalization
        batch_target (int): Size of batches
        batch_source (int): Size of batches
        verbose (bool): Print logging messages

    To Do:
        Possibly allow additional restriction of features during imputation
        Batch impute to reduce memory
    """
    if verbose:
        ih_log.info('Generating imputed {}'.format(layer_target))
        t0 = time.time()
    # Get indices feature information
    out_idx = loom_utils.get_attr_index(loom_file=loom_target,
                                        attr=cell_target,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
    fidx_tar = loom_utils.get_attr_index(loom_file=loom_target,
                                         attr=feat_target,
                                         columns=False,
                                         as_bool=True,
                                         inverse=False)
    cidx_src = loom_utils.get_attr_index(loom_file=loom_source,
                                         attr=cell_source,
                                         columns=True,
                                         as_bool=True,
                                         inverse=False)
    fidx_src = loom_utils.get_attr_index(loom_file=loom_source,
                                         attr=feat_source,
                                         columns=False,
                                         as_bool=True,
                                         inverse=False)
    # Get relevant data from files
    with loompy.connect(filename=loom_source, mode='r') as ds:
        feat_src = ds.ra[id_source]
    with loompy.connect(filename=loom_target, mode='r') as ds:
        num_feat = ds.shape[0]
        feat_tar = ds.ra[id_target]
    # Determine features to include
    if remove_version:
        feat_tar = general_utils.remove_gene_version(gene_ids=feat_tar)
        feat_src = general_utils.remove_gene_version(gene_ids=feat_src)
    feat_tar = pd.DataFrame(np.arange(0, feat_tar.shape[0]),
                            index=feat_tar,
                            columns=['tar'])
    feat_src = pd.DataFrame(np.arange(0, feat_src.shape[0]),
                            index=feat_src,
                            columns=['src'])
    feat_tar = feat_tar.iloc[fidx_tar]
    feat_src = feat_src.iloc[fidx_src]
    feat_df = pd.merge(feat_tar,
                       feat_src,
                       left_index=True,
                       right_index=True,
                       how='inner')
    feat_df = feat_df.sort_values(by='tar')
    # Get Markov matrix
    if neighbor_method == "rescue":
        w_use = all_markov_self(loom_target=loom_target,
                                valid_target=cell_target,
                                loom_source=loom_source,
                                valid_source=cell_source,
                                neighbor_target=mnn_index_target,
                                neighbor_source=mnn_index_source,
                                distance_target=mnn_distance_target,
                                distance_source=mnn_distance_source,
                                k_src_tar=k_src_tar,
                                k_tar_src=k_tar_src,
                                k_rescue=k_rescue,
                                ka=ka,
                                epsilon=epsilon,
                                pca_attr=pca_attr,
                                metric=metric,
                                offset=offset,
                                batch_target=batch_target,
                                batch_source=batch_source,
                                verbose=verbose)
    elif neighbor_method == "mnn":
        w_use = get_markov_impute(loom_target=loom_target,
                                  loom_source=loom_source,
                                  valid_target=cell_target,
                                  valid_source=cell_source,
                                  neighbor_target=mnn_index_target,
                                  neighbor_source=mnn_index_source,
                                  distance_target=mnn_distance_target,
                                  distance_source=mnn_distance_source,
                                  k_src_tar=k_src_tar,
                                  k_tar_src=k_tar_src,
                                  offset=offset,
                                  batch_target=batch_target,
                                  batch_source=batch_source,
                                  verbose=verbose)
    elif neighbor_method == "knn":
        w_use = gen_impute_knn(loom_target=loom_target,
                               loom_source=loom_source,
                               neighbor_attr=mnn_index_target,
                               distance_attr=mnn_distance_target,
                               valid_target=cell_target,
                               valid_source=cell_source,
                               k=10,
                               constraint_relaxation=constraint_relaxation,
                               offset=offset,
                               batch_size=batch_target,
                               verbose=verbose)
    else:
        raise ValueError('Unsupported neighbor method')

    with loompy.connect(filename=loom_target) as ds_tar:
        # Make empty data
        ds_tar.layers[layer_target] = sparse.coo_matrix(ds_tar.shape,
                                                        dtype=float)
        # Get index for batches
        valid_idx = np.unique(w_use.nonzero()[0])
        batches = np.array_split(valid_idx,
                                 np.ceil(valid_idx.shape[0] / batch_target))
        for batch in batches:
            tmp_use = w_use[batch, :]
            use_idx = np.unique(tmp_use.nonzero()[1])
            use_src = np.where(cidx_src)[0][use_idx]
            with loompy.connect(filename=loom_source, mode='r') as ds_src:
                tmp_dat = ds_src.layers[layer_source][:, use_src][
                          feat_df['src'].values, :]
                tmp_dat = sparse.csr_matrix(tmp_dat.T)
            imputed = tmp_use[:, use_idx].dot(tmp_dat)
            imputed = general_utils.expand_sparse(mtx=imputed,
                                                  col_index=feat_df[
                                                      'tar'].values,
                                                  col_N=num_feat)
            imputed = imputed.transpose()
            # POSSIBLE BUG: w_use is restricted to just valid cells, this is
            # looking for locations to put the data into in the target loom file
            # based on indices in that restricted w_use matrix. This should
            # mean that the indices are all off.
            loc_idx = out_idx[batch]
            ds_tar.layers[layer_target][:, loc_idx] = imputed.toarray()
        valid_feat = np.zeros((ds_tar.shape[0],), dtype=int)
        valid_feat[feat_df['tar'].values] = 1
        ds_tar.ra['Valid_{}'.format(layer_target)] = valid_feat
        valid_cells = np.zeros((ds_tar.shape[1],), dtype=int)
        valid_cells[out_idx[valid_idx]] = 1
        ds_tar.ca['Valid_{}'.format(layer_target)] = valid_cells
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        ih_log.info('Imputed data in {0:.2f} {1}'.format(time_run, time_fmt))


def loop_impute_data(loom_source,
                     layer_source,
                     id_source,
                     cell_source,
                     feat_source,
                     loom_target,
                     layer_target,
                     id_target,
                     cell_target,
                     feat_target,
                     mnn_distance_target,
                     mnn_distance_source,
                     mnn_index_target,
                     mnn_index_source,
                     k_src_tar,
                     k_tar_src,
                     k_rescue,
                     ka,
                     epsilon,
                     pca_attr,
                     metric,
                     neighbor_method='rescue',
                     constraint_relaxation=1.1,
                     remove_version=False,
                     offset=1e-5,
                     batch_target=512,
                     batch_source=512,
                     verbose=False):
    """
    Performs imputation over a list (if provided) of layers

    Args:
        loom_source (str): Name of loom file that contains observed count data
        layer_source (str/list): Layer(s) containing observed count data
        id_source (str): Row attribute specifying unique feature IDs
        cell_source (str): Column attribute specifying columns to include
        feat_source (str): Row attribute specifying rows to include
        loom_target (str): Name of loom file that will receive imputed counts
        layer_target (str/list): Layer(s) that will contain imputed count data
        id_target (str): Row attribute specifying unique feature IDs
        cell_target (str): Column attribute specifying columns to include
        feat_target (str): Row attribute specifying rows to include
        mnn_distance_target (str): Attribute containing distances for MNNs
            corr_dist from prep_imputation
        mnn_distance_source (str): Attribute containing distances for MNNs
            corr_dist from prep_imputation
        mnn_index_target (str): Attribute containing indices for MNNs
            corr_idx from prep_imputation
        mnn_index_source (str): Attribute containing indices for MNNs
            corr_idx from prep_imputation
        k_src_tar (int): Number of mutual nearest neighbors
        k_tar_src (int): Number of mutual nearest neighbors
        k_rescue (int): Number of neighbors for rescue
        ka (int): If rescue, neighbor to normalize by
        epsilon (float): If rescue, epsilon value for Gaussian kernel
        pca_attr (str): If not rescue, attribute containing PCs
        metric (str): If not rescue, method for calculating distances
            euclidean
            manhattan
            cosine
        neighbor_method (str): How cells are chosen for imputation
            rescue - include cells that did not make MNNs
            mnn - only include cells that made MNNs
            knn - use a restricted knn search to find neighbors
        constraint_relaxation(float): used for knn impuation
            a ratio determining the number of neigbors that can be 
            formed by cells in the other dataset. Increasing it means
            neighbors can be distributed more unevenly among cells, 
            one means each cell is used equally.
        remove_version (bool): Remove GENCODE version numbers from IDs
        offset (float): Offset for Markov normalization
        batch_target (int): Size of chunks
        batch_source (int): Size of chunks
        verbose (bool): Print logging messages
    """
    if isinstance(layer_source, list) and isinstance(layer_target, list):
        if len(layer_source) != len(layer_target):
            err_msg = 'layer_source and layer_target must have same length'
            if verbose:
                ih_log.error(err_msg)
            raise ValueError(err_msg)
        for i in range(0, len(layer_source)):
            impute_data(loom_source=loom_source,
                        layer_source=layer_source[i],
                        id_source=id_source,
                        cell_source=cell_source,
                        feat_source=feat_source,
                        loom_target=loom_target,
                        layer_target=layer_target[i],
                        id_target=id_target,
                        cell_target=cell_target,
                        feat_target=feat_target,
                        mnn_distance_target=mnn_distance_target,
                        mnn_distance_source=mnn_distance_source,
                        mnn_index_target=mnn_index_target,
                        mnn_index_source=mnn_index_source,
                        k_src_tar=k_src_tar,
                        k_tar_src=k_tar_src,
                        k_rescue=k_rescue,
                        ka=ka,
                        epsilon=epsilon,
                        pca_attr=pca_attr,
                        metric=metric,
                        neighbor_method=neighbor_method,
                        constraint_relaxation=constraint_relaxation,
                        remove_version=remove_version,
                        offset=offset,
                        batch_target=batch_target,
                        batch_source=batch_source,
                        verbose=verbose)

    elif isinstance(layer_source, str) and isinstance(layer_target, str):
        impute_data(loom_source=loom_source,
                    layer_source=layer_source,
                    id_source=id_source,
                    cell_source=cell_source,
                    feat_source=feat_source,
                    loom_target=loom_target,
                    layer_target=layer_target,
                    id_target=id_target,
                    cell_target=cell_target,
                    feat_target=feat_target,
                    mnn_distance_target=mnn_distance_target,
                    mnn_distance_source=mnn_distance_source,
                    mnn_index_target=mnn_index_target,
                    mnn_index_source=mnn_index_source,
                    k_src_tar=k_src_tar,
                    k_tar_src=k_tar_src,
                    k_rescue=k_rescue,
                    ka=ka,
                    epsilon=epsilon,
                    pca_attr=pca_attr,
                    metric=metric,
                    neighbor_method=neighbor_method,
                    constraint_relaxation=constraint_relaxation,
                    remove_version=remove_version,
                    offset=offset,
                    batch_target=batch_target,
                    batch_source=batch_source,
                    verbose=verbose)
    else:
        err_msg = 'layer_source and layer_target must be consistent shapes'
        ih_log.error(err_msg)
        raise ValueError(err_msg)


def auto_find_mutual_k(loom_file,
                       valid_ca=None,
                       verbose=False):
    """
    Automatically determines the optimum k for mutual nearest neighbors

    Args:
        loom_file (str): Path to loom file
        valid_ca (str): Optional, attribute specifying cells to include
        verbose (bool): Print logging messages

    Returns:
        k (int): Optimum k for mutual nearest neighbors
    """
    valid_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                          attr=valid_ca,
                                          columns=True,
                                          as_bool=True,
                                          inverse=False)
    k = np.ceil(0.01 * np.sum(valid_idx))
    k = general_utils.round_unit(x=k,
                                 units=10,
                                 method='nearest')
    k = np.min([200, k])
    if verbose:
        ih_log.info('{0} mutual k: {1}'.format(loom_file, k))
    return k


def auto_find_rescue_k(loom_file,
                       valid_ca,
                       verbose=False):
    """
    Automatically determines the optimum k for rescuing non-MNNs

    Args:
        loom_file (str): Path to loom file
        valid_ca (str): Optional, attribute specifying cells to include
        verbose (bool): Print logging messages

    Returns:
        k (int): Optimum k for rescue
    """
    valid_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                          attr=valid_ca,
                                          columns=True,
                                          as_bool=True,
                                          inverse=False)
    k = np.ceil(0.001 * np.sum(valid_idx))
    k = general_utils.round_unit(x=k,
                                 units=10,
                                 method='nearest')
    k = np.min([50, k])
    if verbose:
        ih_log.info('{0} rescue k: {1}'.format(loom_file,
                                               k))
    return k


def check_ka(k,
             ka):
    """
    Checks if the ka value is appropiate for the provided k

    Args:
        k (int): Number of nearest neighbors
        ka (int): Nearest neighbor to normalize distances by

    Returns:
        ka (int): Nearest neighbor to normalize distances by
            Corrected if ka >= k
    """
    if ka >= k:
        ih_log.warning('ka is too large, resetting')
        ka = np.ceil(0.5 * k)
        ih_log.warning('New ka is {}'.format(ka))
    else:
        ka = ka
    return ka
