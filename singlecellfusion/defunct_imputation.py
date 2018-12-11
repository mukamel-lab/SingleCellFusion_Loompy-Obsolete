"""
Collection of functions used to perform imputation across datasets
    
Written/developed by Fangming Xie and Wayne Doyle

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
import pandas as pd
import time
from scipy import sparse
import functools
import logging
import gc
from . import general_utils
from . import loom_utils
from . import graphs

# Start log
imp_log = logging.getLogger(__name__)


def get_n_variable_features(loom_file,
                            layer,
                            out_attr=None,
                            id_attr='Accession',
                            n_feat=4000,
                            measure='vmr',
                            row_attr=None,
                            col_attr=None,
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
        row_attr (str): Optional, attribute to restrict features by
        col_attr (str): Optional, attribute to restrict cells by
        batch_size (int): Size of chunks
            Will generate a dense array of batch_size by cells
        verbose (bool): Print logging messages
    """
    # Get valid indices
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=col_attr,
                                        columns=True,
                                        as_bool=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=row_attr,
                                        columns=False,
                                        as_bool=True,
                                        inverse=False)
    layers = loom_utils.make_layer_list(layers=layer)
    if verbose:
        imp_log.info(
            'Finding {} variable features for {}'.format(n_feat, loom_file))
        t0 = time.time()
    # Determine variability
    with loompy.connect(filename=loom_file) as ds:
        var_df = pd.DataFrame({'var': np.zeros((ds.shape[0],), dtype=int),
                               'idx': np.zeros((ds.shape[0],), dtype=int)},
                              index=ds.ra[id_attr])
        for (_, selection, view) in ds.scan(items=row_idx,
                                            axis=0,
                                            layers=layers,
                                            batch_size=batch_size):
            dat = view.layers[layer][:, col_idx]
            if measure.lower() == 'sd' or measure.lower() == 'std':
                var_df['var'].iloc[selection] = np.std(dat, axis=1)
            elif measure.lower() == 'vmr':
                var_df['var'].iloc[selection] = np.var(dat, axis=1) / np.mean(
                    dat, axis=1)
            elif measure.lower() == 'cv':
                var_df['var'].iloc[selection] = np.std(dat, axis=1) / np.mean(
                    dat, axis=1)
            else:
                raise ValueError(
                    'Unsupported measure value ({})'.format(measure))
        # Get top n variable features
        n_feat = min(n_feat, var_df.shape[0])
        hvf = var_df['var'].sort_values(ascending=False).head(
            n_feat).index.values
        var_df.loc[hvf, 'idx'] = 1
        if out_attr is None:
            out_attr = 'hvf_{}'.format(n_feat)
        ds.ra[out_attr] = var_df['idx'].values.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        imp_log.info(
            'Found variable features in {0:.2f} {1}'.format(time_run, time_fmt))


def prep_for_common(loom_file,
                    id_attr='Accession',
                    valid_attr=None,
                    remove_version=False):
    """
    Generates objects for find_common_features
    
    Args:
        loom_file (str): Path to loom file
        id_attr (str): Attribute specifying unique feature IDs
        remove_version (bool): Remove GENCODE gene versions from IDs
        valid_attr (str): Optional, attribute that specifies desired features
    
    Returns:
        features (1D array): Array of unique feature IDs
    """
    valid_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                          attr=valid_attr,
                                          columns=False,
                                          as_bool=True,
                                          inverse=False)
    with loompy.connect(filename=loom_file,mode='r') as ds:
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
        common_features (1D array): Array of common features
        out_attr (str): Name of output attribute specifying common features
        
        remove_version (bool): If true remove version ID
            Anything after the first period is dropped
            Useful for GENCODE gene IDs
    """
    # Make logical index of desired features
    feat_ids = prep_for_common(loom_file=loom_file,
                               id_attr=id_attr,
                               remove_version=remove_version,
                               valid_attr=None)
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
                         id_x='Accession',
                         id_y='Accession',
                         valid_x=None,
                         valid_y=None,
                         remove_version=False,
                         verbose=False):
    """
    Identifies common features between two loom files
    
    Args:
        loom_x (str): Path to first loom file
        loom_y (str): Path to second loom file
        out_attr (str): Name of ouput attribute indicating common IDs
            Will be a boolean array indicating IDs in id_x/id_y
        id_x (str): Specifies attribute containing feature IDs
        id_y (str): Specifies attribute containing feature IDs
        valid_x (str): Optional, attribute that specifies desired features
        valid_y (str): Optional, attribute that specifies desired features
        remove_version (bool): If true remove versioning
            Anything after the first period is dropped
            Useful for GENCODE gene IDs
        verbose (bool): If true, print logging messages
    
    Assumptions:
        If true, remove_version is run on both loom files
    """
    if verbose:
        imp_log.info('Finding common features')
    # Get features
    feat_x = prep_for_common(loom_file=loom_x,
                             id_attr=id_x,
                             valid_attr=valid_x,
                             remove_version=remove_version)
    feat_y = prep_for_common(loom_file=loom_y,
                             id_attr=id_y,
                             valid_attr=valid_y,
                             remove_version=remove_version)
    # Find common features
    feats = [feat_x, feat_y]
    common_feat = functools.reduce(np.intersect1d, feats)
    if common_feat.shape[0] == 0:
        imp_log.error('Could not identify any common features')
        raise RuntimeError
    # Add indices
    add_common_features(loom_file=loom_x,
                        id_attr=id_x,
                        common_features=common_feat,
                        out_attr=out_attr,
                        remove_version=True)
    add_common_features(loom_file=loom_y,
                        id_attr=id_y,
                        common_features=common_feat,
                        out_attr=out_attr,
                        remove_version=True)
    if verbose:
        log_msg = 'Found {0} features ({1}%) in common'
        common_x = np.sum(ds.ra[out_attr])
        imp_log.info(log_msg.format(common_x,
                                    loom_utils.get_pct(loom_file=loom_file,
                                                       num_val=common_x,
                                                       columns=False)))

def add_distances_for_mnn(coeff,
                          self_index,
                          other_index,
                          k,
                          dist_vals,
                          idx_vals):
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


def generate_coefficients(dat_x,
                          dat_y):
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


def generate_correlations(loom_x,
                          layer_x,
                          corr_dist_x,
                          corr_idx_x,
                          max_k_x,
                          loom_y,
                          layer_y,
                          corr_dist_y,
                          corr_idx_y,
                          max_k_y,
                          direction,
                          id_x,
                          id_y,
                          ca_x=None,
                          ra_x=None,
                          ca_y=None,
                          ra_y=None,
                          batch_x=512,
                          batch_y=512,
                          remove_version=False,
                          verbose=False):
    """
    Adds correlation matrices between two modalites to loom files
    
    Args:
        loom_x (str): Path to loom file
        layer_x (str): Name of layer containing counts
        corr_dist_x (str): Name of distance attribute for correlations
        corr_idx_x (str): Name of index attribute for correlations
        max_k_x (int): Maximum k needed
        loom_y (str): Path to loom file
        layer_y (str): Name of layer containing counts
        corr_dist_y (str): Name of distance attribute for correlations
        corr_idx_y (str): Name of index attribute for correlations
        max_k_y (int): Maximum k needed
        direction (str): Direction of expected correlation
            negative/- or positive/+
        id_x (str): Attribute containing feature IDs
        id_y (str): Attribute containing feature IDs
        ca_x (str): Name of column attribute to restrict counts by
        ra_x (str): Name of row attribute to restrict counts by
        ca_y (str): Name of column attribute to restrict counts by
        ra_y (str): Name of row attribute to restrict counts by
        batch_x (int): Chunk size for batches
        batch_y (int): Chunk size for batches
        remove_version (bool): If true, remove gene version number
        verbose (bool): Print logging messages
    
    Notes
        Not very efficient, could generate dat_x and pass into subfunction
        To reduce memory could perform garbage collection after subfunction
    """
    if verbose:
        imp_log.info('Generating correlation matrix')
        t0 = time.time()
    layers_x = loom_utils.make_layer_list(layer_x)
    col_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=ca_x,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=ra_x,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    layers_y = loom_utils.make_layer_list(layer_y)
    col_y = loom_utils.get_attr_index(loom_file=loom_y,
                                      attr=ca_y,
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
            x_feat = ds_x.ra[id_x][row_x]
            y_feat = ds_y.ra[id_y][row_y]
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
            dat_x = dat_x.layers[layer_x][row_x, :].T
            dat_x = pd.DataFrame(dat_x).rank(pct=True, axis=1).values
            # Get ranks
            if direction == '+' or direction == 'positive':
                pass
            elif direction == '-' or direction == 'negative':
                dat_x = 1 - dat_x
            else:
                raise ValueError(
                    'Unsupported direction value ({})'.format(direction))
            with loompy.connect(filename=loom_y, mode='r') as ds_y:
                for (_, sel_y, dat_y) in ds_y.scan(axis=1,
                                                   items=col_y,
                                                   layers=layers_y,
                                                   batch_size=batch_y):
                    dat_y = dat_y.layers[layer_y][row_y, :].T
                    dat_y = pd.DataFrame(dat_y).rank(pct=True, axis=1)
                    dat_y.columns = y_feat
                    dat_y = dat_y.loc[:, x_feat]
                    if dat_y.isnull().any().any():
                        raise ValueError('Feature mismatch')
                    dat_y = dat_y.values
                    coeff = generate_coefficients(dat_x,
                                                  dat_y)
                    dist_x, idx_x = add_distances_for_mnn(coeff=coeff,
                                                          self_index=sel_x,
                                                          other_index=sel_y,
                                                          k=max_k_x,
                                                          dist_vals=dist_x,
                                                          idx_vals=idx_x)
                    dist_y, idx_y = add_distances_for_mnn(coeff=coeff.T,
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
        imp_log.info(
            'Generated correlations in {0:.2f} {1}'.format(time_run, time_fmt))


def gen_knn_from_corr(loom_file,
                      corr,
                      neighbor_attr,
                      distance_attr,
                      k,
                      self_idx,
                      other_idx,
                      batch_size=512):
    """
    Gets neighbors and distances from a correlation matrix
    
    Args:
        loom_file (str): Path to loom file
        corr (str): Name of attribute in loom_file containing correlation matrix
        neighbor_attr (str): Name for output attribute specifying neighbors
        distance_attr (str): Name of output attribute specifying distances
        k (int): Number of nearest neighbors 
        self_idx (array): Array of rows to include in correlation matrix
        other_idx (str): Array of rows to include in column matrix
        batch_size (int): Size of chunks
    """
    with loompy.connect(filename=loom_file) as ds:
        neighbors = np.zeros((ds.shape[1], k), dtype=int)
        distances = np.zeros((ds.shape[1], k), dtype=float)
        for (_, selection, view) in ds.scan(axis=1,
                                            layers=[''],
                                            items=self_idx,
                                            batch_size=batch_size):
            tmp = pd.DataFrame(view.ca[corr][:, other_idx])
            knn = ((-tmp).rank(axis=1) <= k).values.astype(bool)
            if np.unique(np.sum(knn, axis=1)).shape[0] != 1:
                raise ValueError('k is inappropriate for data')
            tmp_neighbor = np.reshape(np.where(knn)[1],
                                      (selection.shape[0], -1))
            tmp_distance = np.reshape(tmp.values[knn],
                                      (selection.shape[0], -1))
            neighbors[selection, :] = tmp_neighbor
            distances[selection, :] = tmp_distance
        ds.ca[neighbor_attr] = neighbors
        ds.ca[distance_attr] = distances


def multimodal_adjacency(distances,
                         neighbors,
                         num_col,
                         new_k=None):
    """
    Generates a sparse adjacency matrix from specified distances and neighbors
    Optionally, restricts to a new k nearest neighbors
    
    Args:
        distances (1D array): Distances between elements
        neighbors (1D array): Index of neighbors
        num_col (int): Number of output column in adjacency matrix
        new_k (int): Optional, restrict to this k
    
    Returns 
        A (sparse matrix): Adjacency matrix
    """
    if new_k is None:
        new_k = distances.shape[1]
    if distances.shape[1] != neighbors.shape[1]:
        raise ValueError('Neighbors and distances must have same k!')
    if distances.shape[1] < new_k:
        raise ValueError('new_k must be less than the current k')
    tmp = pd.DataFrame(distances)
    new_k = int(new_k)
    knn = ((-tmp).rank(axis=1, method='first') <= new_k).values.astype(bool)
    if np.unique(np.sum(knn, axis=1)).shape[0] != 1:
        raise ValueError('k is inappropriate for data')
    a = sparse.csr_matrix(
        (np.ones((int(neighbors.shape[0] * new_k),), dtype=int),
         (np.where(knn)[0], neighbors[knn])),
        (neighbors.shape[0], num_col))
    return a


def gen_impute_adj(loom_file,
                   neighbor_attr,
                   distance_attr,
                   k_1,
                   k_2,
                   self_idx,
                   other_idx,
                   batch_size=512):
    """
    Generates adjacency matrix from a loom file
        Subfunction used in perform_imputation
    
    Args:
        loom_file (str): Path to loom file
        neighbor_attr (str): Attribute specifying neighbors
        distance_attr (str): Attribute specifying distances
        k_1 (int): k for first kNN
        k_2 (int): k for second kNN
        self_idx (bool array): Rows in corr to include
        other_idx (bool array) Columns in corr to include
        batch_size (int): Size of chunks
    
    Returns
        adj_1 (sparse matrix): Adjacency matrix for k_1
        adj_2 (sparse_matrix): Adjacency matrix for k_2
    """
    adj_1 = []
    adj_2 = []
    num_other = other_idx.shape[0]
    num_self = self_idx.shape[0]
    with loompy.connect(filename=loom_file,mode='r') as ds:
        if num_self != ds.shape[1]:
            raise ValueError('Index does not match dimensions')
        for (_, selection, view) in ds.scan(axis=1,
                                            layers=[''],
                                            items=self_idx,
                                            batch_size=batch_size):
            adj_1.append(multimodal_adjacency(distances=view.ca[distance_attr],
                                              neighbors=view.ca[neighbor_attr],
                                              num_col=num_other,
                                              new_k=k_1))
            adj_2.append(multimodal_adjacency(distances=view.ca[distance_attr],
                                              neighbors=view.ca[neighbor_attr],
                                              num_col=num_other,
                                              new_k=k_2))
    # Make matrices
    adj_1 = sparse.vstack(adj_1)
    adj_2 = sparse.vstack(adj_2)
    adj_1 = general_utils.expand_sparse(mtx=adj_1,
                                        col_index=None,
                                        row_index=np.where(self_idx)[0],
                                        col_N=None,
                                        row_N=self_idx.shape[0])
    adj_2 = general_utils.expand_sparse(mtx=adj_2,
                                        col_index=None,
                                        row_index=np.where(self_idx)[0],
                                        col_N=None,
                                        row_N=self_idx.shape[0])
    return adj_1, adj_2


def gen_k_adj(loom_x,
              neighbor_x,
              distance_x,
              kx_xy,
              kx_yx,
              col_x,
              batch_x,
              loom_y,
              neighbor_y,
              distance_y,
              ky_yx,
              ky_xy,
              col_y,
              batch_y):
    """
    Generates adjacency matrix using mutual nearest neighbors
    
    Args:
        loom_x (str): Path to loom_file
        neighbor_x (str): Attribute specifying neighbors
        distance_x (str): Attribute specifying distances
        kx_xy (int): Number of nearest neighbors from x to y
        kx_yx (int): Number of mutual nearest neighbors from y to x
        col_x (bool array): Columns to include from loom_x
        batch_x (int): Size of chunks
        loom_y (str): Path to loom_file
        neighbor_y (str): Attribute specifying neighbors
        distance_y (str): Attribute specifying distances
        ky_yx (int): Number of nearest neighbors from y to x
        ky_xy (int): Number of mutual nearest neighbors from x to y
        col_y (bool array): Columns to include from loom_y
        batch_y (int): Size of chunks
    """
    # Get adjacency matrices
    ax_xy, ay_xy = gen_impute_adj(loom_file=loom_x,
                                  neighbor_attr=neighbor_x,
                                  distance_attr=distance_x,
                                  k_1=kx_xy,
                                  k_2=ky_xy,
                                  self_idx=col_x,
                                  other_idx=col_y,
                                  batch_size=batch_x)
    ax_yx, ay_yx = gen_impute_adj(loom_file=loom_y,
                                  neighbor_attr=neighbor_y,
                                  distance_attr=distance_y,
                                  k_1=kx_yx,
                                  k_2=ky_yx,
                                  self_idx=col_y,
                                  other_idx=col_x,
                                  batch_size=batch_y)
    # Generate mutual neighbors adjacency
    adj_xy = (ax_xy.multiply(ax_yx.T))
    adj_yx = (ay_yx.multiply(ay_xy.T))
    return adj_xy, adj_yx


def gen_mutual_adj(loom_x,
                   neighbor_x,
                   distance_x,
                   max_x_to_y,
                   step_x_to_y,
                   mutual_scale_x_to_y,
                   col_x,
                   batch_x,
                   loom_y,
                   neighbor_y,
                   distance_y,
                   max_y_to_x,
                   step_y_to_x,
                   mutual_scale_y_to_x,
                   col_y,
                   batch_y,
                   verbose=False):
    """
    Generates adjacnecy matrix based on mutual nearest neighbors
    
    Args:
        loom_x (str): Path to loom_file
        neighbor_x (str): Attribute specifying neighbors
        distance_x (str): Attribute specifying distances
        max_x_to_y (int): Maximum k value
        step_x_to_y (int): Steps for k
        mutual_scale_x_to_y (int): Scale for mutual k
        col_x (array): Columns to include from loom_x
        batch_x (int): Size of chunks
        loom_y (str): Path to loom_file
        neighbor_y (str): Attribute specifying neighbors
        distance_y (str): Attribute specifying distances
        max_y_to_x (int): Maximum k value
        step_y_to_x (int): Steps for k
        mutual_scale_y_to_x (int): Scale for mutual k
        col_y (array): Columns to include from loom_y
        batch_y (int): Size of chunks
        verbose (bool): If true, print logging messages
    """
    if verbose:
        imp_log.info('Generating mutual adjacency matrix')
    # Get x and y
    k_xy = np.arange(step_x_to_y,
                     max_x_to_y,
                     step_x_to_y)
    k_yx = np.arange(step_y_to_x,
                     max_y_to_x,
                     step_y_to_x)
    # Loop over k values
    for idx, (kx_xy, kx_yx, ky_xy, ky_yx) in enumerate(zip(k_xy,
                                                           mutual_scale_x_to_y * k_xy,
                                                           mutual_scale_y_to_x * k_yx,
                                                           k_yx)):
        # Make adjacency matrix
        adj_xy, adj_yx = gen_k_adj(loom_x=loom_x,
                                   neighbor_x=neighbor_x,
                                   distance_x=distance_x,
                                   kx_xy=kx_xy,
                                   kx_yx=kx_yx,
                                   col_x=col_x,
                                   batch_x=batch_x,
                                   loom_y=loom_y,
                                   neighbor_y=neighbor_y,
                                   distance_y=distance_y,
                                   ky_yx=ky_yx,
                                   ky_xy=ky_xy,
                                   col_y=col_y,
                                   batch_y=batch_y)
        # Get cells
        c_x = np.sort(np.unique(adj_xy.nonzero()[0]))
        c_y = np.sort(np.unique(adj_yx.nonzero()[0]))
        # Update mutual adjacency
        if idx == 0:
            gadj_xy = adj_xy.copy().tolil()
            gc_x = c_x.copy()
            gadj_yx = adj_yx.copy().tolil()
            gc_y = c_y.copy()
        else:
            for j in c_x:
                if j not in gc_x:
                    kcell = int(adj_xy[j, :].sum())
                    if kcell <= max_x_to_y:
                        gadj_xy[j, :] = adj_xy[j, :]
                        gc_x = np.append(gc_x, j)
            for j in c_y:
                if j not in gc_y:
                    kcell = int(adj_yx[j, :].sum())
                    if kcell <= max_y_to_x:
                        gadj_yx[j, :] = adj_yx[j, :]
                        gc_y = np.append(gc_y, j)
        if verbose:
            basic_msg = ('{0}: {1} ({2}%) cells with {3} k to other modality' +
                         'and {4} k back')
            imp_log.info(basic_msg.format(loom_x,
                                          len(gc_x),
                                          loom_utils.get_pct(loom_file=loom_x,
                                                             num_val=len(gc_x),
                                                             columns=True),
                                          kx_xy,
                                          kx_yx))
            imp_log.info(basic_msg.format(loom_y,
                                          len(gc_y),
                                          loom_utils.get_pct(loom_file=loom_y,
                                                             num_val=len(gc_x),
                                                             columns=True),
                                          ky_yx,
                                          ky_xy))
    return gadj_xy, gadj_yx, gc_x, gc_y


def gen_mutual_markov(loom_x,
                      neighbor_x,
                      distance_x,
                      mutual_x,
                      used_x,
                      max_x_to_y,
                      step_x_to_y,
                      mutual_scale_x_to_y,
                      col_x,
                      batch_x,
                      loom_y,
                      neighbor_y,
                      distance_y,
                      mutual_y,
                      used_y,
                      max_y_to_x,
                      step_y_to_x,
                      mutual_scale_y_to_x,
                      col_y,
                      batch_y,
                      offset=1e-5,
                      verbose=False):
    """
    Generates Markov matrix based on mutual nearest neighbors
    
    Args:
        loom_x (str): Path to loom_file
        neighbor_x (str): Attribute specifying neighbors
        distance_x (str): Attribute specifying distances
        mutual_x (str): Output Markov matrix attribute in loom_x
        used_x (str): Output attribute specifying cells with mutual neighbors
        max_x_to_y (int): Maximum k value
        step_x_to_y (int): Steps for k
        mutual_scale_x_to_y (int): Scale for mutual k
        col_x (array): Columns to include from loom_x
        batch_x (int): Size of chunks
        loom_y (str): Path to loom_file
        neighbor_y (str): Attribute specifying neighbors
        distance_y (str): Attribute specifying distances
        mutual_y (str): Name of output Markov matrix attribute in loom_y
        used_y (str): Output attribute specifying cells with mutual neighbors
        max_y_to_x (int): Maximum k value
        step_y_to_x (int): Steps for k
        mutual_scale_y_to_x (int): Scale for mutual k
        col_y (array): Columns to include from loom_y
        batch_y (int): Size of chunks
        offset (float): Size of offset for normalizing adjacency matrices
        verbose (bool): Print logging messages
    
    To Do:
        Add batch way to add Markov matrix
    """
    if verbose:
        imp_log.info('Generating mutual Markov')
        t0 = time.time()
    # Get adjacency matrices
    gadj_xy, gadj_yx, gc_x, gc_y = gen_mutual_adj(loom_x=loom_x,
                                                  neighbor_x=neighbor_x,
                                                  distance_x=distance_x,
                                                  max_x_to_y=max_x_to_y,
                                                  step_x_to_y=step_x_to_y,
                                                  mutual_scale_x_to_y=mutual_scale_x_to_y,
                                                  col_x=col_x,
                                                  batch_x=batch_x,
                                                  loom_y=loom_y,
                                                  neighbor_y=neighbor_y,
                                                  distance_y=distance_y,
                                                  max_y_to_x=max_y_to_x,
                                                  step_y_to_x=step_y_to_x,
                                                  mutual_scale_y_to_x=mutual_scale_y_to_x,
                                                  col_y=col_y,
                                                  batch_y=batch_y,
                                                  verbose=verbose)
    # Normalize adjacency matrices
    gadj_xy = graphs.normalize_adj(adj_mtx=gadj_xy,
                                   axis=1,
                                   offset=offset)
    gadj_yx = graphs.normalize_adj(adj_mtx=gadj_yx,
                                   axis=1,
                                   offset=offset)
    with loompy.connect(filename=loom_x) as ds:
        ds.ca[mutual_x] = gadj_xy.toarray()
        used_idx = np.zeros((ds.shape[1],), dtype=int)
        used_idx[gc_x] = 1
        ds.ca[used_x] = used_idx
    with loompy.connect(filename=loom_y) as ds:
        ds.ca[mutual_y] = gadj_yx.toarray()
        used_idx = np.zeros((ds.shape[1],), dtype=int)
        used_idx[gc_y] = 1
        ds.ca[used_y] = used_idx
    if verbose:
        basic_msg = 'From {0} obtained {1} cells ({2}%)'
        imp_log.info(basic_msg.format(loom_x,
                                      len(gc_x),
                                      loom_utils.get_pct(loom_file=loom_x,
                                                         num_val=len(gc_x),
                                                         columns=True)))
        imp_log.info(basic_msg.format(loom_y,
                                      len(gc_y),
                                      loom_utils.get_pct(loom_file=loom_y,
                                                         num_val=len(gc_y),
                                                         columns=True)))
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        imp_log.info(
            'Generated Markov in {0:.2f} {1}'.format(time_run, time_fmt))


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
                markov_mnn,
                markov_self,
                offset=1e-5,
                remove_version=False,
                batch_size=512,
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
        feat_target (str): Row atttribute specifying rows to include
        markov_mnn (str): Column attribute specifying MNN Markov in target
        markov_self (str): Optional, col_graph specifying target's Markov
        offset (float): Offset for normalizing adjacency matrices
        remove_version (bool): Remove GENCODE version numbers from IDs
        batch_size (int): Chunk size
        verbose (bool): Print logging messages
    
    To Do:
        Possibly allow additional restriction of features during imputation
        Batch impute to reduce memory
    """
    if verbose:
        imp_log.info('Generating imputed {}'.format(layer_target))
        t0 = time.time()
    # Get indices feature information
    cidx_tar = loom_utils.get_attr_index(loom_file=loom_target,
                                         attr=cell_target,
                                         columns=True,
                                         as_bool=True,
                                         inverse=False)
    fidx_tar = loom_utils.get_attr_index(loom_file=loom_target,
                                         attr=feat_target,
                                         columns=False,
                                         as_bool=True,
                                         inverse=False)
    out_idx = np.where(cidx_tar)[0]
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
    with loompy.connect(filename=loom_target,mode='r') as ds:
        num_feat = ds.shape[0]
        feat_tar = ds.ra[id_target]
        w_impute = []
        for (_, selection, view) in ds.scan(axis=1,
                                            items=cidx_tar,
                                            layers=[''],
                                            batch_size=batch_size):
            w_impute.append(sparse.csr_matrix(view.ca[markov_mnn][:, cidx_src]))
        w_impute = sparse.vstack(w_impute)
        del view
        gc.collect()
        if markov_self is not None:
            w_self = ds.col_graphs[markov_self].tolil()[cidx_tar, :][:, cidx_tar]
    with loompy.connect(filename=loom_source, mode='r') as ds:
        feat_src = ds.ra[id_source]
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
    # Update self Markov
    if markov_self is not None:
        if w_impute.shape[0] != w_self.shape[0]:
            raise ValueError('Dimensions of Markov must match!')
        gci = np.unique(w_impute.nonzero()[0])
        for i in range(w_self.shape[0]):
            if i in gci:
                w_self[i, :] = 0
                w_self[i, i] = 1
            else:
                w_self[:, i] = 0
        w_self = graphs.normalize_adj(adj_mtx=w_self,
                                      axis=1,
                                      offset=offset)
        w_use = w_self.dot(w_impute)
    else:
        w_use = w_impute
    with loompy.connect(filename=loom_target) as ds_tar:
        # Make empty data
        ds_tar.layers[layer_target] = sparse.coo_matrix(ds_tar.shape,
                                                        dtype=float)
        # Get index for batches
        valid_idx = np.unique(w_use.nonzero()[0])
        batches = np.array_split(valid_idx,
                                 np.ceil(valid_idx.shape[0] / batch_size))
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
        imp_log.info('Imputed data in {0:.2f} {1}'.format(time_run, time_fmt))


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
                     markov_mnn,
                     markov_self,
                     offset=1e-5,
                     remove_version=False,
                     batch_size=512,
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
        markov_mnn (str): Column attribute specifying MNN Markov in target
        markov_self (str): col_graph attribute specifying target's Markov
        offset (float): Offset for normalizing adjacency matrices
        remove_version (bool): Remove GENCODE version numbers from IDs
        batch_size (int): Size of chunks
        verbose (bool): Print logging messages
    """
    if isinstance(layer_source, list) and isinstance(layer_target, list):
        if len(layer_source) != len(layer_target):
            raise ValueError(
                'layer_source and layer_target should have same length')
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
                        markov_mnn=markov_mnn,
                        markov_self=markov_self,
                        offset=offset,
                        remove_version=remove_version,
                        batch_size=batch_size,
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
                    markov_mnn=markov_mnn,
                    markov_self=markov_self,
                    offset=offset,
                    remove_version=remove_version,
                    batch_size=batch_size,
                    verbose=verbose)
    else:
        raise ValueError(
            'layer_source and layer_target should be consistent shapes')


def prep_for_imputation(loom_x,
                        neighbor_x,
                        distance_x,
                        mutual_x,
                        used_x,
                        loom_y,
                        neighbor_y,
                        distance_y,
                        mutual_y,
                        used_y,
                        ca_x=None,
                        max_x_to_y=100,
                        step_x_to_y=10,
                        mutual_scale_x_to_y=2,
                        batch_x=512,
                        ca_y=None,
                        max_y_to_x=100,
                        step_y_to_x=10,
                        mutual_scale_y_to_x=2,
                        batch_y=512,
                        offset=1e-5,
                        verbose=False):
    """
    Generates mutual kNN and Markov for imputation
    
    Args:
        loom_x (str): Path to loom file containing one dataset
        neighbor_x (str): Name of kNN neighbors
            k = max_x_to_y * scale_x_to_y
        distance_x (str): Name of kNN distances
        mutual_x (str): Name of column attribute containing mutual Markov
        used_x (str): Name of column attribute containing cells that made MNNs
        loom_y (str): Path to loom file containing one dataset
        neighbor_y (str): Name of kNN neighbors
            k = max_y_to_x * scale_y_to_x
        distance_y (str): Name of kNN distances
        mutual_y (str): Name of column attribute containing mutual Markov
        used_y (str): Name of column attribute containing cells that made MNNs
        ca_x (str): Attribute specifying columns to include
        max_x_to_y (int): Maximum k for kNN
        step_x_to_y (int): Step for k values
        mutual_scale_x_to_y (int): Scaling factor for kNN for MNNs
        batch_x (int): Size of chunks
        ca_y (str): Attribute specifying columns to include
        max_y_to_x (int): Maximum k for kNN
        step_y_to_x (int): Step for k values
        mutual_scale_y_to_x (int): Scaling factor for kNN for MNNs
        batch_y (int): Size of chunks
        offset (float): Offset for normalizing adjacency matrices
        verbose (bool): Print logging messages
    """
    # Start log
    if verbose:
        imp_log.info('Preparing to impute between {0} and {1}'.format(loom_x,
                                                                     loom_y))
        t0 = time.time()
        # Get columns
    col_x = loom_utils.get_attr_index(loom_file=loom_x,
                                      attr=ca_x,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    col_y = loom_utils.get_attr_index(loom_file=loom_y,
                                      attr=ca_y,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    # Generate mutual nearest neighbors Markov matrix
    gen_mutual_markov(loom_x=loom_x,
                      neighbor_x=neighbor_x,
                      distance_x=distance_x,
                      mutual_x=mutual_x,
                      used_x=used_x,
                      max_x_to_y=max_x_to_y,
                      step_x_to_y=step_x_to_y,
                      mutual_scale_x_to_y=mutual_scale_x_to_y,
                      col_x=col_x,
                      batch_x=batch_x,
                      loom_y=loom_y,
                      neighbor_y=neighbor_y,
                      distance_y=distance_y,
                      mutual_y=mutual_y,
                      used_y=used_y,
                      max_y_to_x=max_y_to_x,
                      step_y_to_x=step_y_to_x,
                      mutual_scale_y_to_x=mutual_scale_y_to_x,
                      col_y=col_y,
                      batch_y=batch_y,
                      offset=offset,
                      verbose=verbose)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        imp_log.info(
            'Prepared for imputation in {0:.2f} {1}'.format(time_run, time_fmt))


def impute_between_datasets(loom_x,
                            neighbor_x,
                            distance_x,
                            self_x,
                            mutual_x,
                            used_x,
                            layer_x,
                            out_x,
                            loom_y,
                            neighbor_y,
                            distance_y,
                            self_y,
                            mutual_y,
                            used_y,
                            layer_y,
                            out_y,
                            ca_x=None,
                            ra_x=None,
                            id_x='Accession',
                            max_x_to_y=100,
                            step_x_to_y=10,
                            mutual_scale_x_to_y=2,
                            batch_x=512,
                            ca_y=None,
                            ra_y=None,
                            id_y='Accession',
                            max_y_to_x=100,
                            step_y_to_x=10,
                            mutual_scale_y_to_x=2,
                            batch_y=512,
                            offset=1e-5,
                            remove_version=False,
                            verbose=False):
    """
    Imputes data between datasets
    
    Args:
        loom_x (str): Path to loom file containing one dataset
        neighbor_x (str): Name of kNN neighbors
            k = max_x_to_y * scale_x_to_y
        distance_x (str): Name of kNN distances
        self_x (str): Name of col_graph specifying single dataset Markov
        mutual_x (str): Name of column attribute containing mutual Markov
        used_x (str): Name of column attribute containing cells that made MNNs
        layer_x (str/list): Layer(s) containing counts used for imputation
        out_x (str/list): Output layer(s) for imputed data
        loom_y (str): Path to loom file containing one dataset
        neighbor_y (str): Name of kNN neighbors
            k = max_y_to_x * scale_y_to_x
        distance_y (str): Name of kNN distances
        self_y (str): Name of col_graph specifying single dataset Markov
        mutual_y (str): Name of column attribute containing mutual Markov
        used_y (str): Name of column attribute containing cells that made MNNs
        layer_y (str/list): Layer(s) containing counts used for imputation
        out_y (str/list): Output layer(s) for imputed data
        ca_x (str): Attribute specifying columns to include
        ra_x (str): Attribute specifying rows to include
        id_x (str): Row attribute specifying unique feature IDs
        max_x_to_y (int): Maximum k for kNN
        step_x_to_y (int): Step for k values
        mutual_scale_x_to_y (int): Scaling factor for kNN for MNNs
        batch_x (int): Size of chunks
        ca_y (str): Attribute specifying columns to include
        ra_y (str): Attribute specifying rows to include
        id_y (str): Row attribute specifying unique feature IDs
        max_y_to_x (int): Maximum k for kNN
        step_y_to_x (int): Step for k values
        mutual_scale_y_to_x (int): Scaling factor for kNN for MNNs
        batch_y (int): Size of chunks
        offset (float): Offset for normalizing adjacency matrices
        remove_version (bool): Remove GENCODE version IDs
        verbose (bool): Print logging messages
    """
    if verbose:
        t0 = time.time()
    # Prepare for imputation
    prep_for_imputation(loom_x=loom_x,
                        neighbor_x=neighbor_x,
                        distance_x=distance_x,
                        mutual_x=mutual_x,
                        used_x=used_x,
                        loom_y=loom_y,
                        neighbor_y=neighbor_y,
                        distance_y=distance_y,
                        mutual_y=mutual_y,
                        used_y=used_y,
                        ca_x=ca_x,
                        max_x_to_y=max_x_to_y,
                        step_x_to_y=step_x_to_y,
                        mutual_scale_x_to_y=mutual_scale_x_to_y,
                        batch_x=batch_x,
                        ca_y=ca_y,
                        max_y_to_x=max_y_to_x,
                        step_y_to_x=step_y_to_x,
                        mutual_scale_y_to_x=mutual_scale_y_to_x,
                        batch_y=batch_y,
                        offset=offset,
                        verbose=verbose)
    # Impute data for loom_x
    loop_impute_data(loom_source=loom_y,
                     layer_source=layer_y,
                     id_source=id_y,
                     cell_source=ca_y,
                     feat_source=ra_y,
                     loom_target=loom_x,
                     layer_target=out_x,
                     id_target=id_x,
                     cell_target=ca_x,
                     feat_target=ra_x,
                     markov_mnn=mutual_x,
                     markov_self=self_x,
                     offset=offset,
                     remove_version=remove_version,
                     batch_size=batch_x,
                     verbose=verbose)
    # Impute data for loom_y
    loop_impute_data(loom_source=loom_x,
                     layer_source=layer_x,
                     id_source=id_x,
                     cell_source=ca_x,
                     feat_source=ra_y,
                     loom_target=loom_y,
                     layer_target=out_y,
                     id_target=id_y,
                     cell_target=ca_y,
                     feat_target=ra_y,
                     markov_mnn=mutual_y,
                     markov_self=self_y,
                     offset=offset,
                     remove_version=remove_version,
                     batch_size=batch_y,
                     verbose=verbose)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        imp_log.info('Completed imputation in {0:.2f} {1}'.format(time_run,
                                                                 time_fmt))
