"""
Helper functions for performing integrative analyses
The typical user will NOT need to use these functions

Written by Wayne Doyle

(C) 2018 Mukamel Lab GPLv2
"""
import loompy
import numpy as np
import pandas as pd
import time
from scipy import sparse
from scipy.stats import zscore
import functools
import logging
from numba import jit
from annoy import AnnoyIndex
from . import general_utils
from . import loom_utils

help_log = logging.getLogger(__name__)


# Decomposition helpers
def check_pca_batches(loom_file,
                      n_pca=50,
                      batch_size=512,
                      verbose=False):
    """
    Checks and adjusts batch size for PCA

    Args:
        loom_file (str): Path to loom file
        n_pca (int): Number of components for PCA
        batch_size (int): Size of chunks
        verbose (bool): Print logging messages

    Returns:
        batch_size (int): Updated batch size to work with PCA
    """
    # Get the number of cells
    with loompy.connect(loom_file) as ds:
        num_total = ds.shape[1]
    # Check if batch_size and PCA are even reasonable
    if num_total < n_pca:
        err_msg = 'More PCA components {0} than samples {1}'.format(n_pca,
                                                                    num_total)
        if verbose:
            help_log.error(err_msg)
        raise ValueError(help_log)
    if batch_size < n_pca:
        batch_size = n_pca
    # Adjust based on expected size
    mod_total = num_total % batch_size
    adjusted_batch = False
    if mod_total < n_pca:
        adjusted_batch = True
        batch_size = batch_size - n_pca + mod_total
    if batch_size < n_pca:
        batch_size = num_total
    # Report to user
    if verbose and adjusted_batch:
        help_log.info('Adjusted batch size to {0} for PCA'.format(batch_size))
    # Return value
    return batch_size


def prep_pca(view,
             layer,
             row_idx,
             scale_attr=None):
    """
    Performs data processing for PCA on a given layer

    Args:
        view (object): Slice of loom file
        layer (str): Layer in view
        row_idx (array): Features to use
        scale_attr (str): If true, scale cells by this attribute
            Typically used in snmC-seq to scale by a cell's mC/C

    Returns:
        dat (matrix): Scaled data for PCA
    """
    dat = view.layers[layer][row_idx, :].copy()
    if scale_attr is not None:
        rel_scale = view.ca[scale_attr]
        dat = np.divide(dat, rel_scale)
    dat = dat.transpose()
    return dat


# Imputation helpers
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
        help_log.info('{0} mutual k: {1}'.format(loom_file, k))
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
        help_log.info('{0} rescue k: {1}'.format(loom_file,
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
        help_log.warning('ka is too large, resetting')
        ka = np.ceil(0.5 * k)
        help_log.warning('New ka is {}'.format(ka))
    else:
        ka = ka
    return ka


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
        help_log.info('Calculating statistics for {}'.format(loom_file))
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
        help_log.info(
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
        help_log.info(
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
            help_log.error(err_msg)
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
        help_log.info(
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
        help_log.info(
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
            help_log.error(err_msg)
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
        help_log.info(
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
        remove_version (bool): If true remove version number
            Anything after the first period is dropped
            Useful for GENCODE gene IDs
        verbose (bool): If true, print logging messages
    """
    if verbose:
        help_log.info('Finding common features')
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
            help_log.error(err_msg)
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
        help_log.info(log_msg.format(num_comm,
                                     loom_utils.get_pct(loom_file=loom_x,
                                                        num_val=num_comm,
                                                        columns=False),
                                     loom_x,
                                     loom_utils.get_pct(loom_file=loom_y,
                                                        num_val=num_comm,
                                                        columns=False),
                                     loom_y))


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

    This is a buggy-implementation, a more accurate version is coming soon

    Args:
        distance_arr (ndarray): Distances between elements
        neighbor_arr (ndarray): Index of neighbors
        num_other (int): Number of output column in adjacency matrix
        j_max (int): the maximum number of neighbors cells in the
            other modality can make with these cells
        saturated (array-like): cells in the other modality that are
            removed due to saturation constraint
        constraint relaxation(float): a ratio determining the number of
             neighbors that can be formed by cells in the other modality.
             Increasing it means neighbors can be distributed more
             unevenly among cells, one means each cell is used equally.

        k (int): The number of nearest neighbors to restrict to

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
    for _i in np.arange(k):  # get the ith nearest neighbor in the other dataset
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


def gen_impute_adj(loom_file,
                   neighbor_attr,
                   k,
                   self_idx,
                   other_idx):
    """
    Generates adjacency matrix from a loom file

    Args:
        loom_file (str): Path to loom file
        neighbor_attr (str): Attribute specifying neighbors
        k (int): k value for mutual nearest neighbors
        self_idx (ndarray): Rows in corr to include
        other_idx (ndarray) Columns in corr to include

    Returns
        adj (sparse matrix): Adjacency matrix with k nearest
                             neighbors of self in other.
    """
    # Get number of cells
    num_other = other_idx.shape[0]
    num_self = self_idx.shape[0]
    # Get row indices (where valid cells are located)
    row_inds = np.repeat(np.where(self_idx)[0], k)
    with loompy.connect(loom_file) as ds:
        col_inds = np.ravel(ds.ca[neighbor_attr][self_idx, :][:, np.arange(k)])
        # data = np.ravel(ds.ca[distance_attr][self_idx,:][:,np.arange(k)])
    data = [1] * len(row_inds)  # all neighbors have same weight
    adj = sparse.coo_matrix((data, (row_inds, col_inds)),
                            shape=(num_self, num_other))
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
    Generates a restricted knn adjacency matrix from a loom file

    Args:
        loom_target (str): Path to loom file for target modality
        loom_source (str): Path to loom file for source modality
        neighbor_attr (str): Attribute specifying neighbors
        distance_attr (str): Attribute specifying distances
        k (int): The number of nearest neighbors to restrict to
        valid_target (str): Attribute specifying cells to include in target
        valid_source (str): Attribute specifying cells to include in source
        constraint_relaxation(float): a ratio determining the number of
                                    neighbors that can be formed by
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
        help_log.info(log_message)
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
        help_log.info(log_message)
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
                      k_src_tar,
                      k_tar_src,
                      offset=1e-5,
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
        k_src_tar (int): Number of nearest neighbors
        k_tar_src (int): Number of nearest neighbors
        offset (float): Offset for normalization of adjacency matrix
        verbose (bool): Print logging messages

    Returns:
        w_impute (sparse matrix): Markov matrix for imputation
    """
    if verbose:
        help_log.info('Generating mutual adjacency matrix')
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
                           k=k_tar_src,
                           self_idx=cidx_target,
                           other_idx=cidx_source)
    ax_yx = gen_impute_adj(loom_file=loom_source,
                           neighbor_attr=neighbor_source,
                           k=k_src_tar,
                           self_idx=cidx_source,
                           other_idx=cidx_target)
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
        help_log.info(rec_msg.format(loom_target,
                                     c_x,
                                     loom_utils.get_pct(loom_file=loom_target,
                                                        num_val=c_x,
                                                        columns=True)))
        k_msg = '{0} had a k of {1}'
        help_log.info(k_msg.format(loom_target,
                                   k_tar_src))
        help_log.info(k_msg.format(loom_source,
                                   k_src_tar))
    return w_impute


def get_knn_dist_and_idx(t,
                         mat_test,
                         k,
                         search_k=-1,
                         include_distances=False,
                         verbose=False):
    """
    Gets the distances and indices from an Annoy kNN object

    Args:
        t (object): Index for an Annoy kNN
        mat_test (ndarray): Matrix of values to test against kNN
            Used to find neighbors
        k (int): Nearest number of neighbors
        search_k (int): Number of nodes to search
            -1 defaults to n_trees * n
        include_distances (bool): Return distances
            If false, only returns indices
        verbose (bool): Print logging messages

    Returns
        knn_dist (ndarray): Optional, distances for k nearest neighbors
        knn_idx (ndarray): Indices for k nearest neighbors
    """
    # Check data
    train_obs = t.get_n_items()
    train_f = t.f
    test_obs, test_f = mat_test.shape
    if train_f != test_f:
        raise ValueError('mat_test and mat_train dimensions are not identical')
    if k > train_obs:
        if verbose:
            help_log.info(
                'Changing k to reflect observations (k now equals {})'.format(
                    train_obs))
        k = train_obs
    # Set-up for output
    knn_idx = [0] * test_obs
    knn_dist = [0] * test_obs
    if include_distances:
        for i, vector in enumerate(mat_test):
            res = t.get_nns_by_vector(vector,
                                      k,
                                      search_k=search_k,
                                      include_distances=include_distances)
            knn_idx[i] = res[0]
            knn_dist[i] = res[1]
    else:
        for i, vector in enumerate(mat_test):
            res = t.get_nns_by_vector(vector,
                                      k,
                                      search_k=search_k,
                                      include_distances=include_distances)
            knn_idx[i] = res
    # Convert to arrays
    knn_idx = np.array(knn_idx)
    knn_dist = np.array(knn_dist)
    if include_distances:
        return knn_dist, knn_idx.astype(int)
    else:
        return knn_idx.astype(int)


def prep_knn_object(num_dim,
                    metric='euclidean'):
    """
    Generates an Annoy kNN index

    Args:
        num_dim: Number of dimensions for vectors in kNN
        metric (str): Distance metric for kNN
            angular, euclidean, manhattan, hamming, dot

    Returns:
        t (object): Annoy index for kNN
    """
    t = AnnoyIndex(num_dim,
                   metric=metric)
    t.set_seed(23)
    return t


def add_mat_to_knn(mat,
                   t,
                   start_idx=0):
    """
    Adds a matrix to an Annoy kNN index

    Args:
        mat (ndarray): Array of values to add to kNN
        t (object): Annoy index
        start_idx (int): Start index for mat in t
            Useful when adding in batches

    Returns:
        t (object): Annoy index
        new_idx (int): Last index added to t
    """
    for i, val in enumerate(mat):
        new_idx = i + start_idx
        t.add_item(new_idx, val)
    return t, new_idx


def build_knn(t,
              n_trees=10):
    """
    Builds a forest of tress for kNN

    Args:
        t (object): Annoy index for kNN
        n_trees (int): Number of trees to use for kNN
            More trees leads to higher precision

    Returns:
        t (object): Annoy index for kNN
    """
    t.build(n_trees)
    return t


def low_mem_distance_index(mat_train,
                           mat_test,
                           k,
                           metric='euclidean',
                           n_trees=10,
                           search_k=-1,
                           include_distances=True,
                           verbose=False):
    """
    Uses Annoy to find indices and distances for nearest neighbors
        This will hold everything in memory, so is recommended for things like
        finding neighbors in lower dimensional space

    Args:
        mat_train (ndarray): Matrix to train the kNN on
        mat_test (ndarray): Matrix to test the kNN on
        k (int): Number of nearest neighbors
        metric (str): Distance metric for kNN
            angular, euclidean, manhattan, hamming, dot
        n_trees (int): Number of trees for kNN
            more trees = more precision
        search_k (int): Number of nodes to use for searching kNN
            -1 = n_trees * n
        include_distances (bool): Return distances
            If false, only returns kNN indices
        verbose (bool): Print logging messages

    Returns:
        knn_res (tuple/ndarray): kNN indices and (include_distances) distances
    """
    # Get dimensions
    train_f = mat_train.shape[1]
    test_f = mat_test.shape[1]
    if train_f != test_f:
        raise ValueError('mat_train and mat_test dimensions are not identical')
    # Build kNN
    t = prep_knn_object(num_dim=train_f,
                        metric=metric)
    t, _x = add_mat_to_knn(mat=mat_train,
                           t=t)
    t = build_knn(t=t,
                  n_trees=n_trees)
    # Get distances and indices
    knn_res = get_knn_dist_and_idx(t=t,
                                   mat_test=mat_test,
                                   k=k,
                                   search_k=search_k,
                                   include_distances=include_distances,
                                   verbose=verbose)
    return knn_res


def train_knn(loom_file,
              layer,
              row_arr,
              col_arr,
              feat_attr,
              feat_select,
              reverse_rank,
              remove_version,
              batch_size):
    """
    Trains a kNN using loom data in batches

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer containing data to add to kNN
        row_arr (ndarray): Boolean vector of rows to include from loom_file
        col_arr (ndarray): Boolean vector of columns to include from loom_file
        feat_attr (str): Row attribute in loom_file specifying feature IDs
        feat_select (ndarray): Vector of features to include for kNN
        reverse_rank (bool): Reverse the ranking of features in a cell
            Used if expected correlation is negative
        remove_version (bool): Remove GENCODE version ID
        batch_size (int): Size of chunks for iterating over loom_file

    Returns:
        t (object): Annoy kNN index
    """
    # Prepare kNN object
    t = prep_knn_object(num_dim=feat_select.shape[0],
                        metric='dot')
    current_idx = 0
    # Get layers
    layers = loom_utils.make_layer_list(layer)
    # Train kNN object
    with loompy.connect(filename=loom_file, mode='r') as ds:
        for (_, selection, view) in ds.scan(axis=1,
                                            items=col_arr,
                                            layers=layers,
                                            batch_size=batch_size):
            # Get data
            dat = pd.DataFrame(view.layers[layer][row_arr, :].T,
                               columns=view.ra[feat_attr][row_arr])
            if remove_version:
                dat.columns = general_utils.remove_gene_version(dat.columns)
            dat = dat.loc[:, feat_select]
            dat = pd.DataFrame(dat).rank(pct=True, axis=1)
            dat = dat.apply(zscore, axis=1, result_type='expand').values
            if reverse_rank:
                dat = -1 * dat
            # Add to kNN
            t, current_idx = add_mat_to_knn(mat=dat,
                                            t=t,
                                            start_idx=current_idx)
            current_idx += 1  # Need to iterate by one to start at next element
    # Return kNN
    return t


def report_knn(loom_file,
               layer,
               row_arr,
               col_arr,
               feat_attr,
               feat_select,
               reverse_rank,
               k,
               t,
               batch_size,
               remove_version,
               verbose):
    """
    Gets distance and indices from kNN

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer with counts for kNN
        row_arr (ndarray): Boolean vector of rows to include in loom_file
        col_arr (ndarray): Boolean vector of columns to include in loom_file
        feat_attr (str): Row attribute specifying feature IDs in loom_file
        feat_select (ndarray): Vector of features to include from loom_file
        reverse_rank (bool): Reverse rank ordering of features per cell
            Useful if expected correlation is negative
        k (int): Number of nearest neighbors
        t (object): Annoy index
        batch_size (int): Size of chunks to iterate for loom file
        remove_version (bool): Remove GENCODE gene version ID
        verbose (bool): Print logging messages

    Returns:
        dist (ndarray): Array of distances for kNN
        idx (ndarray): Array of indices for kNN
    """
    # Make distance and index arrays
    with loompy.connect(loom_file) as ds:
        num_cells = ds.shape[1]
    dist = np.zeros((num_cells, k))
    idx = np.zeros((num_cells, k))
    # Get layers
    layers = loom_utils.make_layer_list(layer)
    # Get results of kNN object
    with loompy.connect(filename=loom_file, mode='r') as ds:
        for (_, selection, view) in ds.scan(axis=1,
                                            items=col_arr,
                                            layers=layers,
                                            batch_size=batch_size):
            # Get data
            dat = pd.DataFrame(view.layers[layer][row_arr, :].T,
                               columns=view.ra[feat_attr][row_arr])
            if remove_version:
                dat.columns = general_utils.remove_gene_version(dat.columns)
            dat = dat.loc[:, feat_select]
            dat = pd.DataFrame(dat).rank(pct=True, axis=1)
            dat = dat.apply(zscore,
                            axis=1,
                            result_type='expand').values
            if reverse_rank:
                dat = -1 * dat
            # Get distances and indices
            tmp_dist, tmp_idx = get_knn_dist_and_idx(t=t,
                                                     mat_test=dat,
                                                     k=k,
                                                     search_k=-1,
                                                     include_distances=True,
                                                     verbose=verbose)
            dist[selection, :] = tmp_dist
            idx[selection, :] = tmp_idx
    # Return values
    return dist, idx


def perform_loom_knn(loom_x,
                     layer_x,
                     neighbor_distance_x,
                     neighbor_index_x,
                     max_k_x,
                     loom_y,
                     layer_y,
                     neighbor_distance_y,
                     neighbor_index_y,
                     max_k_y,
                     direction,
                     feature_id_x,
                     feature_id_y,
                     valid_ca_x=None,
                     valid_ra_x=None,
                     valid_ca_y=None,
                     valid_ra_y=None,
                     n_trees=10,
                     batch_x=512,
                     batch_y=512,
                     remove_version=False,
                     verbose=False):
    """
    Gets kNN distances and indices by iterating over a loom file

    Args:
        loom_x (str): Path to loom file
        layer_x (str): Layer containing data for loom_x
        neighbor_distance_x (str): Output attribute for distances
        neighbor_index_x (str): Output attribute for indices
        max_k_x (int): Maximum number of nearest neighbors for x
        loom_y (str): Path to loom file
        layer_y (str): Layer containing data for loom_y
        neighbor_distance_y (str): Output attribute for distances
        neighbor_index_y (str): Output attribute for indices
        max_k_y  (int): Maximum number of nearest neighbors for y
        direction (str): Expected direction of relationship between x and y
            positive or +
            negative or -
        feature_id_x (str): Row attribute containing unique feature IDs
        feature_id_y (str): Row attribute containing unique feature IDs
        valid_ca_x (str): Column attribute specifying valid cells
        valid_ra_x (str): Row attribute specifying valid features
        valid_ca_y (str): Column attribute specifying valid cells
        valid_ra_y (str): Row attribute specifying valid features
        n_trees (int): Number of trees to use for kNN
            more trees = more precision
        batch_x (int): Size of chunks for iterating over loom_x
        batch_y (int): Size of chunks for iterating over loom_y
        remove_version (bool): Remove GENCODE version IDs
        verbose (bool): Print logging messages
    """
    # Prep for function
    if verbose:
        help_log.info('Finding kNN distances and indices')
        t0 = time.time()
    # Prep for kNN
    col_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=valid_ca_x,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=valid_ra_x,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    col_y = loom_utils.get_attr_index(loom_file=loom_y,
                                      attr=valid_ca_y,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_y = loom_utils.get_attr_index(loom_file=loom_y,
                                      attr=valid_ra_y,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    # Make lookup
    lookup_x = pd.Series(np.where(col_x)[0],
                         index=np.arange(np.sum(col_x)))
    lookup_y = pd.Series(np.where(col_y)[0],
                         index=np.arange(np.sum(col_y)))

    # Get features
    with loompy.connect(filename=loom_x) as ds_x:
        x_feat = ds_x.ra[feature_id_x][row_x]
    with loompy.connect(filename=loom_y) as ds_y:
        y_feat = ds_y.ra[feature_id_y][row_y]
    if remove_version:
        x_feat = general_utils.remove_gene_version(x_feat)
        y_feat = general_utils.remove_gene_version(y_feat)
    if np.any(np.sort(x_feat) != np.sort(y_feat)):
        raise ValueError('Feature mismatch!')
    # Train kNNs
    reverse_y = False
    if direction == '+' or direction == 'positive':
        reverse_x = False
    elif direction == '-' or direction == 'negative':
        reverse_x = True
    else:
        raise ValueError('Unsupported direction value')
    t_y2x = train_knn(loom_file=loom_x,
                      layer=layer_x,
                      row_arr=row_x,
                      col_arr=col_x,
                      feat_attr=feature_id_x,
                      feat_select=x_feat,
                      reverse_rank=reverse_x,
                      remove_version=remove_version,
                      batch_size=batch_x)
    t_x2y = train_knn(loom_file=loom_y,
                      layer=layer_y,
                      row_arr=row_y,
                      col_arr=col_y,
                      feat_attr=feature_id_y,
                      feat_select=x_feat,
                      reverse_rank=reverse_y,
                      remove_version=remove_version,
                      batch_size=batch_y)
    # Build trees
    t_x2y = build_knn(t=t_x2y,
                      n_trees=n_trees)
    t_y2x = build_knn(t=t_y2x,
                      n_trees=n_trees)
    # Get distances and indices
    dist_x, idx_x = report_knn(loom_file=loom_x,
                               layer=layer_x,
                               row_arr=row_x,
                               col_arr=col_x,
                               feat_attr=feature_id_x,
                               feat_select=x_feat,
                               reverse_rank=reverse_x,
                               k=max_k_x,
                               t=t_x2y,
                               batch_size=batch_x,
                               remove_version=remove_version,
                               verbose=verbose)
    dist_y, idx_y = report_knn(loom_file=loom_y,
                               layer=layer_y,
                               row_arr=row_y,
                               col_arr=col_y,
                               feat_attr=feature_id_y,
                               feat_select=x_feat,
                               reverse_rank=reverse_y,
                               k=max_k_y,
                               t=t_y2x,
                               batch_size=batch_y,
                               remove_version=remove_version,
                               verbose=verbose)
    # Get correct indices (import if restricted to valid cells)
    correct_idx_x = np.reshape(lookup_y.loc[np.ravel(idx_x).astype(int)].values,
                               idx_x.shape)
    correct_idx_y = np.reshape(lookup_x.loc[np.ravel(idx_y).astype(int)].values,
                               idx_y.shape)
    # Add data to files
    with loompy.connect(filename=loom_x) as ds:
        ds.ca[neighbor_distance_x] = dist_x
        ds.ca[neighbor_index_x] = correct_idx_x
    with loompy.connect(filename=loom_y) as ds:
        ds.ca[neighbor_distance_y] = dist_y
        ds.ca[neighbor_index_y] = correct_idx_y
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        help_log.info(
            'Found neighbors in {0:.2f} {1}'.format(time_run, time_fmt))


def rescue_markov(loom_target,
                  valid_target,
                  mnns,
                  k,
                  ka,
                  epsilon,
                  pca_attr,
                  offset = 1e-5,
                  verbose=False):
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
        offset (float): Offset for avoiding divide by zero errors
        verbose (bool): Print logging messages

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
    # Get neighbors and distance
    cidx_tar = loom_utils.get_attr_index(loom_file=loom_target,
                                         attr=valid_target,
                                         columns=True,
                                         as_bool=True,
                                         inverse=False)
    tot_n = cidx_tar.shape[0]
    # Get PCs
    with loompy.connect(loom_target) as ds:
        all_pcs = ds.ca[pca_attr][cidx_tar, :]
        mnn_pcs = ds.ca[pca_attr][mnns,:]
    # Get within-modality MNN
    distances, indices = low_mem_distance_index(mat_train=mnn_pcs,
                                                mat_test=all_pcs,
                                                k=k,
                                                metric='euclidean',
                                                n_trees=10,
                                                search_k=-1,
                                                verbose=verbose,
                                                include_distances=True)
    if ka > 0:
        distances = distances / (np.sort(distances,
                                         axis=1)[:, ka].reshape(-1, 1))
    # Calculate gaussian kernel
    adjs = np.exp(-((distances ** 2) / (2 * (epsilon ** 2))))
    # Construct W
    rows = np.repeat(np.where(cidx_tar)[0], k)
    cols = mnns[np.ravel(indices)]
    vals = np.ravel(adjs)
    w = sparse.csr_matrix((vals, (rows, cols)), shape=(tot_n, tot_n))
    # Normalize
    w = normalize_adj(adj_mtx=w,
                      axis=1,
                      offset=offset)
    return w


def all_markov_self(loom_target,
                    valid_target,
                    loom_source,
                    valid_source,
                    neighbor_target,
                    neighbor_source,
                    k_src_tar,
                    k_tar_src,
                    k_rescue,
                    ka,
                    epsilon,
                    pca_attr,
                    offset=1e-5,
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
        k_src_tar (int): Number of nearest neighbors for MNN
        k_tar_src (int): Number of nearest neighbors for MNN
        k_rescue (int): Number of nearest neighbors for rescue
        ka (int): Normalizes distance by kath cell's distance
        epsilon (float): Noise parameter for Gaussian kernel
        pca_attr (str): Attribute containing PCs
        offset (float): Offset for Markov normalization
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
                                 k_src_tar=k_src_tar,
                                 k_tar_src=k_tar_src,
                                 offset=offset,
                                 verbose=verbose)
    mnns = np.unique(w_impute.nonzero()[0])
    # Get w_self
    w_self = rescue_markov(loom_target=loom_target,
                           valid_target=valid_target,
                           mnns=mnns,
                           k=k_rescue,
                           ka=ka,
                           epsilon=epsilon,
                           pca_attr=pca_attr,
                           offset=offset,
                           verbose=verbose)
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
                neighbor_distance_target,
                neighbor_index_target,
                neighbor_index_source,
                k_src_tar,
                k_tar_src,
                k_rescue,
                ka,
                epsilon,
                pca_attr,
                neighbor_method='rescue',
                constraint_relaxation=1.1,
                remove_version=False,
                offset=1e-5,
                batch_target=512,
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
        neighbor_distance_target (str): Attribute containing distances for MNNs
        neighbor_index_target (str): Attribute containing indices for MNNs
        neighbor_index_source (str): Attribute containing indices for MNNs
        k_src_tar (int): Number of nearest neighbors for MNNs
        k_tar_src (int): Number of nearest neighbors for MNNs
        k_rescue (int): Number of nearest neighbors for rescue
        ka (int): If rescue, neighbor to normalize by
        epsilon (float): If rescue, epsilon value for Gaussian kernel
        pca_attr (str): If rescue, attribute containing PCs
        neighbor_method (str): How cells are chosen for imputation
            rescue - include cells that did not make MNNs
            mnn - only include cells that made MNNs
            knn - use a restricted knn search to find neighbors
        constraint_relaxation(float): ratio determining the number of neighbors
            that can be formed by cells in the other modality.
            Increasing it means neighbors can be distributed more unevenly among
            cells, one means each cell is used equally.
            Used for neighbor_method == knn
        remove_version (bool): Remove GENCODE version numbers from IDs
        offset (float): Offset for Markov normalization
        batch_target (int): Size of batches
        verbose (bool): Print logging messages
    """
    if verbose:
        help_log.info('Generating imputed {}'.format(layer_target))
        t0 = time.time()
    # Get indices feature information
    fidx_tar = loom_utils.get_attr_index(loom_file=loom_target,
                                         attr=feat_target,
                                         columns=False,
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
    if neighbor_method == 'rescue':
        w_use = all_markov_self(loom_target=loom_target,
                                valid_target=cell_target,
                                loom_source=loom_source,
                                valid_source=cell_source,
                                neighbor_target=neighbor_index_target,
                                neighbor_source=neighbor_index_source,
                                k_src_tar=k_src_tar,
                                k_tar_src=k_tar_src,
                                k_rescue=k_rescue,
                                ka=ka,
                                epsilon=epsilon,
                                pca_attr=pca_attr,
                                offset=offset,
                                verbose=verbose)
    elif neighbor_method == 'mnn':
        w_use = get_markov_impute(loom_target=loom_target,
                                  loom_source=loom_source,
                                  valid_target=cell_target,
                                  valid_source=cell_source,
                                  neighbor_target=neighbor_index_target,
                                  neighbor_source=neighbor_index_source,
                                  k_src_tar=k_src_tar,
                                  k_tar_src=k_tar_src,
                                  offset=offset,
                                  verbose=verbose)
    elif neighbor_method == 'knn':
        w_use = gen_impute_knn(loom_target=loom_target,
                               loom_source=loom_source,
                               neighbor_attr=neighbor_index_target,
                               distance_attr=neighbor_distance_target,
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
            tmp_w = w_use[batch, :]
            use_feat = np.unique(tmp_w.nonzero()[1])
            with loompy.connect(filename=loom_source, mode='r') as ds_src:
                tmp_dat = ds_src.layers[layer_source][:, use_feat][
                          feat_df['src'].values, :]
                tmp_dat = sparse.csr_matrix(tmp_dat.T)
            imputed = tmp_w[:, use_feat].dot(tmp_dat)
            imputed = general_utils.expand_sparse(mtx=imputed,
                                                  col_index=feat_df[
                                                      'tar'].values,
                                                  col_n=num_feat)
            imputed = imputed.transpose()
            ds_tar.layers[layer_target][:, batch] = imputed.toarray()
        valid_feat = np.zeros((ds_tar.shape[0],), dtype=int)
        valid_feat[feat_df['tar'].values] = 1
        ds_tar.ra['Valid_{}'.format(layer_target)] = valid_feat
        valid_cells = np.zeros((ds_tar.shape[1],), dtype=int)
        valid_cells[valid_idx] = 1
        ds_tar.ca['Valid_{}'.format(layer_target)] = valid_cells
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        help_log.info('Imputed data in {0:.2f} {1}'.format(time_run, time_fmt))


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
                     neighbor_distance_target,
                     neighbor_index_target,
                     neighbor_index_source,
                     k_src_tar,
                     k_tar_src,
                     k_rescue,
                     ka,
                     epsilon,
                     pca_attr,
                     neighbor_method='rescue',
                     constraint_relaxation=1.1,
                     remove_version=False,
                     offset=1e-5,
                     batch_target=512,
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
        neighbor_distance_target (str): Attribute containing distances for MNNs
            corr_dist from prep_imputation
        neighbor_index_target (str): Attribute containing indices for MNNs
            corr_idx from prep_imputation
        neighbor_index_source (str): Attribute containing indices for MNNs
            corr_idx from prep_imputation
        k_src_tar (int): Number of mutual nearest neighbors
        k_tar_src (int): Number of mutual nearest neighbors
        k_rescue (int): Number of neighbors for rescue
        ka (int): If rescue, neighbor to normalize by
        epsilon (float): If rescue, epsilon value for Gaussian kernel
        pca_attr (str): If not rescue, attribute containing PCs
        neighbor_method (str): How cells are chosen for imputation
            rescue - include cells that did not make MNNs
            mnn - only include cells that made MNNs
            knn - use a restricted knn search to find neighbors
        constraint_relaxation(float): used for knn imputation
            a ratio determining the number of neighbors that can be
            formed by cells in the other dataset. Increasing it means
            neighbors can be distributed more unevenly among cells,
            one means each cell is used equally.
        remove_version (bool): Remove GENCODE version numbers from IDs
        offset (float): Offset for Markov normalization
        batch_target (int): Size of chunks
        verbose (bool): Print logging messages
    """
    if isinstance(layer_source, list) and isinstance(layer_target, list):
        if len(layer_source) != len(layer_target):
            err_msg = 'layer_source and layer_target must have same length'
            if verbose:
                help_log.error(err_msg)
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
                        neighbor_distance_target=neighbor_distance_target,
                        neighbor_index_target=neighbor_index_target,
                        neighbor_index_source=neighbor_index_source,
                        k_src_tar=k_src_tar,
                        k_tar_src=k_tar_src,
                        k_rescue=k_rescue,
                        ka=ka,
                        epsilon=epsilon,
                        pca_attr=pca_attr,
                        neighbor_method=neighbor_method,
                        constraint_relaxation=constraint_relaxation,
                        remove_version=remove_version,
                        offset=offset,
                        batch_target=batch_target,
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
                    neighbor_distance_target=neighbor_distance_target,
                    neighbor_index_target=neighbor_index_target,
                    neighbor_index_source=neighbor_index_source,
                    k_src_tar=k_src_tar,
                    k_tar_src=k_tar_src,
                    k_rescue=k_rescue,
                    ka=ka,
                    epsilon=epsilon,
                    pca_attr=pca_attr,
                    neighbor_method=neighbor_method,
                    constraint_relaxation=constraint_relaxation,
                    remove_version=remove_version,
                    offset=offset,
                    batch_target=batch_target,
                    verbose=verbose)
    else:
        err_msg = 'layer_source and layer_target must be consistent shapes'
        help_log.error(err_msg)
        raise ValueError(err_msg)
