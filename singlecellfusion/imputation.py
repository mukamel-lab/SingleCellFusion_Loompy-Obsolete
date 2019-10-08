"""
Functions used to imputing data across modalities

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

(C) 2019 Mukamel Lab GPLv2
"""

import loompy
import pandas as pd
import numpy as np
from scipy import stats
from scipy import sparse
from . import utils
from scipy.stats import zscore
import tempfile
from annoy import AnnoyIndex
import logging
import time
import os

# Start log
imp_log = logging.getLogger(__name__)


def temp_zscore_loom(loom_file,
                     raw_layer,
                     feat_attr='Accession',
                     valid_ca=None,
                     valid_ra=None,
                     batch_size=512,
                     tmp_dir=None,
                     verbose=False):
    if verbose:
        t0 = time.time()
        imp_log.info(
            'Generating temporary z-scored file for {}'.format(loom_file))
    # Prep
    col_idx = utils.get_attr_index(loom_file=loom_file,
                                   attr=valid_ca,
                                   columns=True,
                                   as_bool=False,
                                   inverse=False)
    row_idx = utils.get_attr_index(loom_file=loom_file,
                                   attr=valid_ra,
                                   columns=False,
                                   as_bool=False,
                                   inverse=False)
    layers = utils.make_layer_list(raw_layer)
    append_loom = False
    start_pos = 0
    # Make temporary loom file
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    tmp_loom = tempfile.mktemp(suffix='.loom', dir=tmp_dir)
    with loompy.connect(filename=loom_file, mode='r') as ds:
        for (_, selection, view) in ds.scan(axis=1,
                                            items=col_idx,
                                            layers=layers,
                                            batch_size=batch_size):
            # Get zscore
            dat = pd.DataFrame(view.layers[raw_layer][row_idx, :].T)
            dat = pd.DataFrame(dat).rank(pct=True, axis=1)
            dat = dat.apply(zscore, axis=1, result_type='expand').values.T
            # Reshape 
            dat = sparse.coo_matrix((np.ravel(dat),
                                     (np.repeat(row_idx, dat.shape[1]),
                                      np.tile(selection, row_idx.shape[0]))),
                                    shape=ds.shape)
            # Restrict for easy add
            dat = dat.tocsc()
            new_idx = np.arange(start=start_pos,
                                stop=selection[-1] + 1,
                                step=1)
            dat = dat[:, new_idx]
            utils.batch_add_sparse(loom_file=tmp_loom,
                                   layers={'': dat},
                                   row_attrs={feat_attr: ds.ra[feat_attr]},
                                   col_attrs={'FakeID': new_idx},
                                   append=append_loom,
                                   empty_base=False,
                                   batch_size=batch_size)
            append_loom = True
            start_pos = selection[-1] + 1
        if start_pos < ds.shape[1]:
            dat = sparse.coo_matrix((ds.shape[0], ds.shape[1] - start_pos))
            new_idx = np.arange(start=start_pos,
                                stop=ds.shape[1] + 1,
                                step=1)
            utils.batch_add_sparse(loom_file=tmp_loom,
                                   layers={'': dat},
                                   row_attrs={feat_attr: ds.ra[feat_attr]},
                                   col_attrs={'FakeID': new_idx},
                                   append=append_loom,
                                   empty_base=False,
                                   batch_size=batch_size)
    # Log
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        imp_log.info(
            'Made temporary loom file in {0:.2f} {1}'.format(time_run,
                                                             time_fmt))
    return tmp_loom


def get_knn_dist_and_idx(t,
                         mat_test,
                         k,
                         search_k=-1,
                         include_distances=False,
                         verbose=False):
    """
    Gets the distances and indices from an Annoy kNN object

    Args:
        t (Annoy object): Index for an Annoy kNN
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
            imp_log.info(
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


def low_mem_train_knn(loom_file,
                      layer,
                      row_arr,
                      col_arr,
                      feat_attr,
                      feat_select,
                      reverse_rank,
                      remove_version,
                      tmp_dir,
                      seed,
                      batch_size,
                      verbose):
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
        tmp_dir (str): Output directory for temporary files
            If None, writes to system's default
        seed (int): Seed for annoy
        batch_size (int): Size of chunks for iterating over loom_file
        verbose (bool): Print logging messages

    Returns:
        t (object): Annoy kNN index
        index_file (str): Path to file containing index
    """
    if verbose:
        imp_log.info('Training kNN')
    # Prepare kNN object
    t = AnnoyIndex(feat_select.shape[0],
                   metric='dot')
    # Low memory so build on disk
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as tmpfile:
        index_file = tmpfile.name
    t.on_disk_build(index_file)
    # Set seed
    if seed is not None:
        t.set_seed(seed)
    current_idx = 0
    # Get layers
    layers = utils.make_layer_list(layer)
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
                dat.columns = utils.remove_gene_version(dat.columns)
            dat = dat.loc[:, feat_select].values
            if reverse_rank:
                dat = -1 * dat
            # Add to kNN
            for _, val in enumerate(dat):
                t.add_item(current_idx, val)
                current_idx += 1
    # Return kNN
    return t, index_file


def low_mem_report_knn(loom_file,
                       layer,
                       row_arr,
                       col_arr,
                       feat_attr,
                       feat_select,
                       reverse_rank,
                       k,
                       t,
                       index_file,
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
        index_file (str): Path to on disk index file for kNN
        batch_size (int): Size of chunks to iterate for loom file
        remove_version (bool): Remove GENCODE gene version ID
        verbose (bool): Print logging messages

    Returns:
        dist (ndarray): Array of distances for kNN
        idx (ndarray): Array of indices for kNN
    """
    if verbose:
        imp_log.info('Querying kNN')
    # Make distance and index arrays
    with loompy.connect(loom_file) as ds:
        num_cells = ds.shape[1]
    dist = np.zeros((num_cells, k))
    idx = np.zeros((num_cells, k))
    # Get layers
    layers = utils.make_layer_list(layer)
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
                dat.columns = utils.remove_gene_version(dat.columns)
            dat = dat.loc[:, feat_select].values
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
    # Remove temporary file
    os.remove(index_file)
    # Return values
    return dist, idx


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


def rescue_markov(loom_target,
                  valid_target,
                  mnns,
                  k,
                  ka,
                  epsilon,
                  pca_attr,
                  offset=1e-5,
                  seed=None,
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
        seed (int): Seed for annoy
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
    cidx_tar = utils.get_attr_index(loom_file=loom_target,
                                    attr=valid_target,
                                    columns=True,
                                    as_bool=True,
                                    inverse=False)
    tot_n = cidx_tar.shape[0]
    # Get PCs
    with loompy.connect(loom_target) as ds:
        all_pcs = ds.ca[pca_attr][cidx_tar, :]
        mnn_pcs = ds.ca[pca_attr][mnns, :]
    # Get within-modality MNN
    distances, indices = low_mem_distance_index(mat_train=mnn_pcs,
                                                mat_test=all_pcs,
                                                k=k,
                                                metric='euclidean',
                                                n_trees=10,
                                                search_k=-1,
                                                verbose=verbose,
                                                include_distances=True,
                                                seed=seed)
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


def gen_impute_knn(loom_target,
                   loom_source,
                   neighbor_attr,
                   k,
                   valid_target,
                   valid_source,
                   offset=1e-5,
                   verbose=False):
    """
    Generates a restricted knn adjacency matrix from a loom file

    Args:
        loom_target (str): Path to loom file for target modality
        loom_source (str): Path to loom file for source modality
        neighbor_attr (str): Attribute specifying neighbors
        k (int): The number of nearest neighbors to restrict to
        valid_target (str): Attribute specifying cells to include in target
        valid_source (str): Attribute specifying cells to include in source
        offset (float): Offset for normalization of adjacency matrix
        verbose (bool): Print logging messages

     Returns:
        adj (sparse matrix): Adjacency matrix with k nearest
                             neighbors of self in other.
    """

    if verbose:
        log_message = 'Generating restricted KNN adjacency matrix k={}'.format(
            k)
        imp_log.info(log_message)
    # Get indices
    self_idx = utils.get_attr_index(loom_file=loom_target,
                                    attr=valid_target,
                                    columns=True,
                                    as_bool=True,
                                    inverse=False)
    other_idx = utils.get_attr_index(loom_file=loom_source,
                                     attr=valid_source,
                                     columns=True,
                                     as_bool=True,
                                     inverse=False)
    # Get adjacency
    adj = gen_impute_adj(loom_file=loom_target,
                         neighbor_attr=neighbor_attr,
                         k=k,
                         self_idx=self_idx,
                         other_idx=other_idx)
    # Normalize
    adj = normalize_adj(adj_mtx=adj,
                        axis=1,
                        offset=offset)
    return adj


def low_mem_distance_index(mat_train,
                           mat_test,
                           k,
                           metric='euclidean',
                           n_trees=10,
                           search_k=-1,
                           include_distances=True,
                           seed=None,
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
        seed (int): Seed for Annoy
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
    if verbose:
        imp_log.info('Building kNN')
    t = AnnoyIndex(train_f,
                   metric=metric)
    if seed is not None:
        t.set_seed(seed)
    for i, row in enumerate(mat_train):
        t.add_item(i, row)
    t.build(n_trees)
    # Get distances and indices
    knn_res = get_knn_dist_and_idx(t=t,
                                   mat_test=mat_test,
                                   k=k,
                                   search_k=search_k,
                                   include_distances=include_distances,
                                   verbose=verbose)
    return knn_res


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
        imp_log.info('Generating mutual adjacency matrix')
    cidx_target = utils.get_attr_index(loom_file=loom_target,
                                       attr=valid_target,
                                       columns=True,
                                       as_bool=True,
                                       inverse=False)
    cidx_source = utils.get_attr_index(loom_file=loom_source,
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
        imp_log.info(rec_msg.format(loom_target,
                                    c_x,
                                    utils.get_pct(loom_file=loom_target,
                                                  num_val=c_x,
                                                  axis=1)))
        k_msg = '{0} had a k of {1}'
        imp_log.info(k_msg.format(loom_target,
                                  k_tar_src))
        imp_log.info(k_msg.format(loom_source,
                                  k_src_tar))
    return w_impute


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
                    seed=None,
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
        seed (int): Seed for Annoy
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
                           seed=seed,
                           verbose=verbose)
    w_use = w_self.dot(w_impute)
    return w_use


def get_dat_df(loom_file,
               layer='',
               feat_attr='Accession',
               cell_attr='CellID',
               valid_ra=None,
               valid_ca=None,
               remove_gene_version=False):
    """
    Gets a sparse matrix containing specified data from a loom file

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer in loom file containing data for imputation
        feat_attr (str): Row attribute containing unique feature IDs
        cell_attr (str): Column attribute containing unique cell IDs
        valid_ra (str): Row attribute specifying valid features
        valid_ca (str): Column attribute specifying valid cells
        remove_gene_version (bool): Remove GENCODE version ID

    Returns:
        dat (df): Data frame of counts from layer

    """
    # Get indices
    ra_idx = utils.get_attr_index(loom_file=loom_file,
                                  attr=valid_ra,
                                  columns=False,
                                  as_bool=False,
                                  inverse=False)
    ca_idx = utils.get_attr_index(loom_file=loom_file,
                                  attr=valid_ca,
                                  columns=True,
                                  as_bool=False,
                                  inverse=False)
    # Get data in sparse format
    with loompy.connect(loom_file) as ds:
        dat = pd.DataFrame(ds.layers[layer].sparse(ra_idx, ca_idx).todense(),
                           index=ds.ra[feat_attr][ra_idx],
                           columns=ds.ca[cell_attr][ca_idx])
    if remove_gene_version:
        dat.index = utils.remove_gene_version(dat.index.values)
    dat = dat.T
    return dat


def knn_to_sparse(knn,
                  row_n,
                  col_n):
    # Check inputs
    if row_n != knn.shape[0]:
        raise ValueError('Dimension mismatch')
    # Get indices
    row_inds = np.repeat(np.arange(row_n), knn.shape[1])
    col_inds = np.ravel(knn)
    # Get values for sparse matrix (using value of 1)
    data = [1] * len(row_inds)
    knn_sparse = sparse.coo_matrix((data, (row_inds, col_inds)),
                                   shape=(row_n, col_n))
    return knn_sparse


def high_mem_iteration_knn(train_dat,
                           test_dat,
                           k,
                           n_trees=10,
                           verbose=False):
    # Start time
    if verbose:
        t0 = time.time()
    # Check inputs
    if train_dat.shape[1] != test_dat.shape[1]:
        raise ValueError('Different number of genes per dataset')
    # Build the kNN map
    t = AnnoyIndex(f=train_dat.shape[1],
                   metric='dot')
    for i, row in enumerate(train_dat):
        t.add_item(i, row)
    t.build(n_trees)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        imp_log.info('Built kNN tree in {0:.2f} {1}'.format(time_run,
                                                            time_fmt))
    # Check tree
    n_obs = t.get_n_items()
    if test_dat.shape[1] != t.f:
        raise ValueError('Different number of genes, this should have been caught')
    if k > n_obs:
        if verbose:
            imp_log.info('Too few observations, setting k to number of observations')
        k = n_obs
    # Get kNN indices
    knn_idx = [0] * (test_dat.shape[0])
    for i, vector in enumerate(test_dat):
        res = t.get_nns_by_vector(vector,
                                  k,
                                  search_k=-1,
                                  include_distances=False)
        knn_idx[i] = res
    knn_idx = np.array(knn_idx).astype(int)
    if verbose:
        t2 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t2)
        imp_log.info('Built kNN tree in {0:.2f} {1}'.format(time_run,
                                                            time_fmt))
    return knn_idx


def high_mem_gen_constrained_knn(dat_target,
                                 dat_source,
                                 n_neighbors=20,
                                 k_saturate=20,
                                 speed_factor=10,
                                 n_trees=10,
                                 seed=None,
                                 verbose=False):
    """
    Generates a kNN matrix for imputed

    Args:
        dat_target (df): Counts that will be replaced
        dat_source (df): Counts that will be used for imputation
        n_neighbors (int): Number of neighbors to make
        k_saturate (int): Number of neighbors before saturation
        speed_factor (int): Find speed_factor * n_neighbors
            Speeds up code at the expense of higher memory
        n_trees (int): Number of trees for generating kNN
            Larger value is more accurate but higher memory
        seed (int): Seed for randomization
        verbose (bool): print logging messages

    Returns:
        knn_impute: kNN for imputing data
    """
    # Start time
    if verbose:
        t0 = time.time()
        imp_log.info('Finding kNNs')
    # Check inputs
    if dat_source.shape[1] != dat_target.shape[1]:
        raise ValueError('Datasets have different numbers of genes')
    # Prepare for finding kNN
    accepted_knn = []
    accepted_cells = []
    rejected_cells = np.arange(dat_target.shape[0])
    # Record cells in x
    n_connects = np.zeros(dat_source.shape[0], dtype=int)
    unsaturated = (n_connects < k_saturate)
    unsaturated_cells = np.arange(dat_source.shape[0])[unsaturated]
    if seed is not None:
        np.random.seed(seed)
    # Loop over cells
    while rejected_cells.shape[0] != 0:
        # Logging message
        if verbose:
            pct_rem = 100 - rejected_cells.shape[0] / dat_target.shape[0] * 100
            pct_sat = unsaturated_cells.shape[0] / dat_source.shape[0] * 100
            imp_log.info('{0:.2f}% of target cells still need to make connections'.format(pct_rem))
            imp_log.info('{0:.2f}% of source cells can make connections'.format(pct_sat))
        np.random.shuffle(rejected_cells)
        # Find k value
        k_val = min(n_neighbors * speed_factor,
                    unsaturated_cells.shape[0])
        # Get kNN (looking for nearest neighbors in y for each cell in x)
        knn = high_mem_iteration_knn(train_dat=dat_source.values[unsaturated_cells, :],
                                     test_dat=dat_target.values[rejected_cells, :],
                                     k=k_val,
                                     n_trees=n_trees)
        # Transform kNN to global index
        knn = unsaturated_cells[knn]
        # Check each cell
        rejected_local_idx = []
        for local_idx, cell in enumerate(rejected_cells):
            # Get knn
            knn_in_source = knn[local_idx].copy()
            # Filter out saturated cells
            knn_in_source = knn_in_source[unsaturated[knn_in_source]]
            if knn_in_source.size < n_neighbors:
                # Too many, reject
                rejected_local_idx.append(local_idx)
            else:
                # Accept, not too many connections
                accepted_knn.append(knn_in_source[:n_neighbors])
                accepted_cells.append(cell)
                n_connects[knn_in_source[:n_neighbors]] += 1
                unsaturated = (n_connects < k_saturate)
        # Update values for next iteration
        unsaturated_cells = np.arange(dat_source.shape[0])[unsaturated]
        rejected_cells = rejected_cells[rejected_local_idx]
    # Get accepted knn
    accepted_knn = pd.DataFrame(np.vstack(accepted_knn),
                                index=accepted_cells)
    accepted_knn = accepted_knn.sort_index().values
    # Make into sparse format
    knn_impute = knn_to_sparse(knn=accepted_knn,
                               row_n=dat_target.shape[0],
                               col_n=dat_source.shape[0])
    # Normalize kNN (each cell in y, how many cells in x it connects to)
    degrees = np.ravel(knn_impute.sum(axis=1)) + 1e-7  # offset for zeros
    knn_impute = sparse.diags(1.0 / degrees).dot(knn_impute)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        imp_log.info('Found kNNs in {0:.2f} {1}'.format(time_run,
                                                        time_fmt))
    # Return to user
    return knn_impute


def normalize_dat(dat,
                  mod_sign=1):
    """
    Performs a z-score based normalization of a dataframe

    Args:
        dat (df): Contains counts to be normalized
        mod_sign (int): Negative/positive scale of normalized data for correlations

    Returns:
        norm (df): Z-scored and scaled data
    """
    norm = dat.rank(pct=True, axis=1).apply(func=stats.zscore,
                                            result_type='broadcast',
                                            axis=1,
                                            ddof=1) * mod_sign
    return norm


def high_mem_knn_impute(loom_source,
                        dat_source,
                        norm_source,
                        loom_target,
                        layer_target,
                        feat_target,
                        cell_target,
                        valid_ra_target,
                        valid_ca_target,
                        layer_impute,
                        correlation,
                        remove_version,
                        n_neighbors,
                        relaxation,
                        speed_factor,
                        n_trees,
                        seed,
                        verbose):
    """
    Imputes data in one direction for one pair of data

    Args:
        loom_source (str): Path to loom file containing source data
        dat_source (df): Contains counts for source data
        norm_source (df): Contains normalized source data counts
        loom_target (str): Path to loom file containing target data
        layer_target (str): Layer in loom_target with count data
        feat_target (str): Row attribute in loom_target with feature IDs
        cell_target (str): Column attribute in loom_target with cell IDs
        valid_ra_target (str): Row attribute with valid features specified
        valid_ca_target (str): Column attribute with valid cells specified
        layer_impute (str): Output layer in loom_target for imputed counts
            Valid_{layer_impute} will be added to rows and columns
        correlation (str): Expected correlation (negative or positive)
        remove_version (bool): If true, remove GENCODE version ID
        n_neighbors (int): Minimum number of neighbors for kNN
        relaxation (int): Factor to relax saturation by
        speed_factor (int): Factor to speed-up kNN search by
        n_trees (int): Number of trees for Annoy kNN
        seed (int): Seed for randomization
        verbose (bool): If true, print logging messages
    """
    # Start log
    if verbose:
        t0 = time.time()
        imp_log.info('Imputing from {0} to {1}'.format(loom_source,
                                                       loom_target))
    # Get correlation
    if correlation.lower() in ['neg', 'negative', '-']:
        mod_sign = -1
    elif correlation.lower() in ['pos', 'positive', '+']:
        mod_sign = 1
    else:
        raise ValueError('Unsupported correlation value ({})'.format(correlation))
    # Get target data
    dat_target = get_dat_df(loom_file=loom_target,
                            layer=layer_target,
                            feat_attr=feat_target,
                            cell_attr=cell_target,
                            valid_ra=valid_ra_target,
                            valid_ca=valid_ca_target,
                            remove_gene_version=remove_version)
    # Check dimensions
    if dat_target.shape[1] != dat_source.shape[1]:
        raise ValueError('Datasets have different number of genes')
    if np.array_equal(np.sort(np.unique(dat_source.columns.values)),
                      np.sort(np.unique(dat_target.columns.values))):
        pass
    else:
        raise ValueError('Datasets have different gene IDs')
    dat_target = dat_target.loc[:, dat_source.columns.values]
    # Z-score data
    norm_target = normalize_dat(dat_target,
                                mod_sign=mod_sign)
    # Determine the maximum number of kNNs that can be made
    n_target = dat_target.shape[0]
    n_source = dat_source.shape[0]
    k_saturate = int((n_target / n_source) * n_neighbors * relaxation) + 1
    # Impute data
    knn_impute = high_mem_gen_constrained_knn(dat_target=norm_target,
                                              dat_source=norm_source,
                                              n_neighbors=n_neighbors,
                                              k_saturate=k_saturate,
                                              speed_factor=speed_factor,
                                              n_trees=n_trees,
                                              seed=seed,
                                              verbose=verbose)
    # Impute data
    imputed = pd.DataFrame(knn_impute.dot(dat_source.values),
                           index=norm_target.index.values,
                           columns=norm_target.columns.values)
    # Get lookup
    with loompy.connect(loom_target) as ds:
        n_feat = ds.shape[0]
        n_cells = ds.shape[1]
        feat_lookup = pd.DataFrame({'idx': np.arange(n_feat)},
                                   index=ds.ra[feat_target])
        cell_lookup = pd.DataFrame({'idx': np.arange(n_cells)},
                                   index=ds.ca[cell_target])
    if remove_version:
        feat_lookup.index = utils.remove_gene_version(feat_lookup.index.values)
    # Sort lookup
    feat_lookup = feat_lookup.loc[imputed.columns.values]
    cell_lookup = cell_lookup.loc[imputed.index.values]
    # Transform imputed into a sparse matrix
    imputed = imputed.T
    imputed.columns = cell_lookup['idx'].values
    imputed['id'] = feat_lookup['idx'].values
    imputed = imputed.melt(id_vars='id')
    imputed = sparse.coo_matrix((imputed['value'].values,
                                 (imputed['id'].values,
                                  imputed['variable'].values)),
                                shape=(n_feat, n_cells))
    # Get valid imputed features / cells
    valid_cells = np.zeros(n_cells, dtype=int)
    valid_cells[cell_lookup['idx'].values] = 1
    valid_feat = np.zeros(n_feat, dtype=int)
    valid_feat[feat_lookup['idx'].values] = 1
    # Add imputed to loom file
    with loompy.connect(loom_target) as ds:
        ds.ca['Valid_{}'.format(layer_impute)] = valid_cells
        ds.ra['Valid_{}'.format(layer_impute)] = valid_feat
        ds.layers[layer_impute] = imputed
    # Log
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        imp_log.info('Imputed from {0} to {1} in {2:.2f} {3}'.format(loom_source,
                                                                     loom_target,
                                                                     time_run,
                                                                     time_fmt))


def high_mem_constrained_1d(loom_source,
                            loom_target,
                            layer_source='',
                            layer_target='',
                            layer_impute='imputed',
                            correlation='positive',
                            feat_source='Accession',
                            feat_target='Accession',
                            cell_source='CellID',
                            cell_target='CellID',
                            valid_ra_source=None,
                            valid_ra_target=None,
                            valid_ca_source=None,
                            valid_ca_target=None,
                            n_neighbors=20,
                            relaxation=10,
                            speed_factor=10,
                            n_trees=10,
                            remove_version=False,
                            seed=None,
                            verbose=False):
    """
    Imputes data from a given data modality (source) into another (target) quickly but at the cost of memory

    Args:
        loom_source (str): Path to loom file that will provide counts to others
        loom_target (str/list): Path(s) to loom files that will receive imputed counts
        layer_source (str): Layer in loom_source that will provide counts
        layer_target (str/list): Layer(s) in loom_target files specifying counts
            Used for finding correlations between loom files
        layer_impute (str/list): Output layer in loom_target
            A row attribute with the format Valid_{layer_out} will be added
            A col attribute with the format Valid_{layer_out} will be added
        correlation (str/list): Expected correlation between loom_source and loom_target
            positive/+ for RNA-seq and ATAC-seq
            negative/- for RNA-seq or ATAC-seq and snmC-seq
        feat_source (str): Row attribute specifying unique feature names in loom_source
        feat_target (str/list): Row attribute(s) specifying unique feature names in loom_target
        cell_source (str): Column attribute specifying unique cell IDs in loom_source
        cell_target (str/list): Column attribute specifying unique cell IDs in loom_target
        valid_ra_source (str): Row attribute specifying valid features in loom_source
            Should point to a boolean array
        valid_ra_target (str/list): Row attribute(s) specifying valid features in loom_target
            Should point to a boolean array
        valid_ca_source (str): Column attribute specifying valid cells in loom_source
            Should point to a boolean array
        valid_ca_target (str/list): Column attribute(s) specifying valid cells in loom_target
            Should point to a boolean array
        n_neighbors (int/list): Minimum amount of neighbors that can be made
        relaxation (int/list): Relax search for kNN by this factor
            Increases the number of neighbors that a source cell can make before saturation
        speed_factor (int/list): Speed up search of kNN by this factor
            Will increase memory but decrease running time
        n_trees (int): Number of trees for approximate kNN search
            See Annoy documentation
        remove_version (bool/list): If true remove version number
            Anything after the first period is dropped (useful for GENCODE IDs)
            If a list, will behave differently for each loom file
            If a boolean, will behave the same for each loom_file
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
                            n_neighbors,
                            relaxation,
                            speed_factor,
                            remove_version,
                            layer_impute]
        checked = utils.mimic_list(parameters=check_parameters,
                                   list_len=len(loom_target))
        feat_target = checked[0]
        cell_target = checked[1]
        valid_ra_target = checked[2]
        valid_ca_target = checked[3]
        n_neighbors = checked[4]
        relaxation = checked[5]
        speed_factor = checked[6]
        remove_version = checked[7]
        layer_impute = checked[8]
        is_a_list = True
    elif isinstance(loom_target, str):
        utils.all_same_type_size(parameters=[loom_target,
                                             layer_target,
                                             correlation,
                                             feat_target,
                                             cell_target],
                                 expected_type='str',
                                 confirm_size=False)
    if verbose:
        imp_log.info('Preparing for imputation')
    # Get source data (same throughout all iterations)
    dat_source = get_dat_df(loom_file=loom_source,
                            layer=layer_source,
                            feat_attr=feat_source,
                            cell_attr=cell_source,
                            valid_ra=valid_ra_source,
                            valid_ca=valid_ca_source,
                            remove_gene_version=remove_version)
    norm_source = normalize_dat(dat_source,
                                mod_sign=1)
    # Impute for each target
    if is_a_list:
        for i in np.arange(len(loom_target)):
            high_mem_knn_impute(loom_source=loom_source,
                                dat_source=dat_source,
                                norm_source=norm_source,
                                loom_target=loom_target[i],
                                layer_target=layer_target[i],
                                feat_target=feat_target[i],
                                cell_target=cell_target[i],
                                valid_ra_target=valid_ra_target[i],
                                valid_ca_target=valid_ca_target[i],
                                layer_impute=layer_impute[i],
                                correlation=correlation[i],
                                remove_version=remove_version[i],
                                n_neighbors=n_neighbors[i],
                                relaxation=relaxation[i],
                                speed_factor=speed_factor[i],
                                n_trees=n_trees,
                                seed=seed,
                                verbose=verbose)
    else:
        high_mem_knn_impute(loom_source=loom_source,
                            dat_source=dat_source,
                            norm_source=norm_source,
                            loom_target=loom_target,
                            layer_target=layer_target,
                            feat_target=feat_target,
                            cell_target=cell_target,
                            valid_ra_target=valid_ra_target,
                            valid_ca_target=valid_ca_target,
                            layer_impute=layer_impute,
                            correlation=correlation,
                            remove_version=remove_version,
                            n_neighbors=n_neighbors,
                            relaxation=relaxation,
                            speed_factor=speed_factor,
                            n_trees=n_trees,
                            seed=seed,
                            verbose=verbose)


def low_mem_knn_impute(loom_source,
                       layer_source,
                       zscore_source,
                       feat_source,
                       valid_ra_source,
                       valid_ca_source,
                       loom_target,
                       layer_target,
                       feat_target,
                       valid_ra_target,
                       valid_ca_target,
                       layer_impute,
                       correlation,
                       remove_version,
                       n_neighbors,
                       relaxation,
                       speed_factor,
                       n_trees,
                       batch_size,
                       seed,
                       tmp_dir,
                       verbose):
    """
    Imputes data in one direction for one pair of data in a slow, low memory fashion

    Args:
        loom_source (str): Path to loom file that will provide counts to others
        layer_source (str): Layer in loom_source that will provide counts
        zscore_source (str): Path to loom file containing z-scored data
        feat_source (str): Row attribute containing unique feature IDs
        valid_ra_source (str): Row attribute specifying features that can be used
        valid_ca_source (str): Column attribute specifying cells that can be used
        loom_target (str/list): Path(s) to loom files that will receive imputed counts
        layer_target (str): Layer in loom_target that contains counts for correlations
        feat_target (str): Row attribute containing unique feature IDs
        valid_ra_target (str): Row attribute specifying features that can be used
        valid_ca_target (str): Column attribute specifying cells that can be used
        layer_impute (str): Output layer for loom_target that will receive imputed counts
        correlation (str): Expected correlation (negative/-, positive/+)
        remove_version (bool): If true, remove GENCODE version ID from feat_source/feat_target
        n_neighbors (int): Minimum number of nearest neighbors to make
        relaxation (int): Factor for relaxing saturation limit
        speed_factor (int): Factor for speeding up kNN search
        n_trees (int): Number of trees for kNN search
        batch_size (int): Size of chunks for batch iterations
        seed (int): Seed for randomization
        tmp_dir (str): Path to output directory
        verbose (bool): If true, print logging messages

    """
    # Start log
    if verbose:
        t0 = time.time()
        imp_log.info('Imputing from {0} to {1}'.format(loom_source,
                                                       loom_target))
    # Get correlation
    if correlation.lower() in ['neg', 'negative', '-']:
        reverse_rank = True
    elif correlation.lower() in ['pos', 'positive', '+']:
        reverse_rank = False
    else:
        raise ValueError('Unsupported value for correlation ({})'.format(correlation))
    # Generate constrained kNN
    low_mem_constrained_knn(loom_target=loom_target,
                            knn_index='imputed_knn',  # hack for now, will be overwritten each time
                            layer_target=layer_target,
                            valid_ca_target=valid_ca_target,
                            valid_ra_target=valid_ra_target,
                            feature_target=feat_target,
                            loom_source=loom_source,
                            zscore_source=zscore_source,
                            valid_ca_source=valid_ca_source,
                            valid_ra_source=valid_ra_source,
                            feature_source=feat_source,
                            n_neighbors=n_neighbors,
                            reverse_rank=reverse_rank,
                            speed_factor=speed_factor,
                            relaxation=relaxation,
                            n_trees=n_trees,
                            batch_size=batch_size,
                            remove_version=remove_version,
                            seed=seed,
                            tmp_dir=tmp_dir,
                            verbose=verbose)
    # Impute data
    low_mem_impute_data(loom_source=loom_source,
                        layer_source=layer_source,
                        feat_source=feat_source,
                        valid_ca_source=valid_ca_source,
                        valid_ra_source=valid_ra_source,
                        loom_target=loom_target,
                        layer_impute=layer_impute,
                        feat_target=feat_target,
                        valid_ca_target=valid_ca_target,
                        valid_ra_target=valid_ra_target,
                        neighbor_index_target='imputed_knn',  # hack for now, will be overwritten each time
                        neighbor_index_source=None,
                        k_src_tar=None,
                        k_tar_src=n_neighbors,
                        k_rescue=None,
                        ka=None,
                        epsilon=None,
                        pca_attr=None,
                        neighbor_method='knn',
                        remove_version=remove_version,
                        offset=1e-5,
                        seed=seed,
                        batch_size=batch_size,
                        verbose=verbose)

    # Log
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        imp_log.info('Imputed from {0} to {1} in {2:.2f} {3}'.format(loom_source,
                                                                     loom_target,
                                                                     time_run,
                                                                     time_fmt))


def low_mem_constrained_1d(loom_source,
                           loom_target,
                           layer_source='',
                           layer_target='',
                           layer_impute='imputed',
                           correlation='positive',
                           feat_source='Accession',
                           feat_target='Accession',
                           cell_target='CellID',
                           valid_ra_source=None,
                           valid_ra_target=None,
                           valid_ca_source=None,
                           valid_ca_target=None,
                           n_neighbors=20,
                           relaxation=10,
                           speed_factor=10,
                           n_trees=10,
                           remove_version=False,
                           tmp_dir=None,
                           batch_size=5000,
                           seed=None,
                           verbose=False):
    """
    Imputes data from a given data modality (source) into another (target) with low memory but slowly

    Args:
        loom_source (str): Path to loom file that will provide counts to others
        loom_target (str/list): Path(s) to loom files that will receive imputed counts
        layer_source (str): Layer in loom_source that will provide counts
        layer_target (str/list): Layer(s) in loom_target files specifying counts
            Used for finding correlations between loom files
        layer_impute (str/list): Output layer in loom_target
            A row attribute with the format Valid_{layer_out} will be added
            A col attribute with the format Valid_{layer_out} will be added
        correlation (str/list): Expected correlation between loom_source and loom_target
            positive/+ for RNA-seq and ATAC-seq
            negative/- for RNA-seq or ATAC-seq and snmC-seq
        feat_source (str): Row attribute specifying unique feature names in loom_source
        feat_target (str/list): Row attribute(s) specifying unique feature names in loom_target
        cell_target (str/list): Column attribute specifying unique cell IDs in loom_target
        valid_ra_source (str): Row attribute specifying valid features in loom_source
            Should point to a boolean array
        valid_ra_target (str/list): Row attribute(s) specifying valid features in loom_target
            Should point to a boolean array
        valid_ca_source (str): Column attribute specifying valid cells in loom_source
            Should point to a boolean array
        valid_ca_target (str/list): Column attribute(s) specifying valid cells in loom_target
            Should point to a boolean array
        n_neighbors (int/list): Minimum amount of neighbors that can be made
        relaxation (int/list): Relax search for kNN by this factor
            Increases the number of neighbors that a source cell can make before saturation
        speed_factor (int/list): Speed up search of kNN by this factor
            Will increase memory but decrease running time
        n_trees (int): Number of trees for approximate kNN search
            See Annoy documentation
        remove_version (bool/list): If true remove version number
            Anything after the first period is dropped (useful for GENCODE IDs)
            If a list, will behave differently for each loom file
            If a boolean, will behave the same for each loom_file
        tmp_dir (str): Optional, path to output directory for temporary files
            If None, uses default temporary directory on your system
        batch_size (int): Number of elements per chunk when analyzing in batches
        seed (int): Initialization for random seed
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
                            valid_ra_target,
                            valid_ca_target,
                            n_neighbors,
                            relaxation,
                            speed_factor,
                            remove_version,
                            layer_impute]
        checked = utils.mimic_list(parameters=check_parameters,
                                   list_len=len(loom_target))
        feat_target = checked[0]
        valid_ra_target = checked[1]
        valid_ca_target = checked[2]
        n_neighbors = checked[3]
        relaxation = checked[4]
        speed_factor = checked[5]
        remove_version = checked[6]
        layer_impute = checked[7]
        is_a_list = True
    elif isinstance(loom_target, str):
        utils.all_same_type_size(parameters=[loom_target,
                                             layer_target,
                                             correlation,
                                             feat_target,
                                             cell_target],
                                 expected_type='str',
                                 confirm_size=False)
    if verbose:
        imp_log.info('Preparing for imputation')
    # Get source z-scored loom file
    zscore_source = temp_zscore_loom(loom_file=loom_source,
                                     raw_layer=layer_source,
                                     feat_attr=feat_source,
                                     valid_ca=valid_ca_source,
                                     valid_ra=valid_ra_source,
                                     batch_size=batch_size,
                                     tmp_dir=tmp_dir,
                                     verbose=verbose)
    # Impute for each target
    if is_a_list:
        for i in np.arange(len(loom_target)):
            low_mem_knn_impute(loom_source=loom_source,
                               layer_source=layer_source,
                               zscore_source=zscore_source,
                               feat_source=feat_source,
                               valid_ra_source=valid_ra_source,
                               valid_ca_source=valid_ca_source,
                               loom_target=loom_target[i],
                               layer_target=layer_target[i],
                               feat_target=feat_target[i],
                               valid_ra_target=valid_ra_target[i],
                               valid_ca_target=valid_ca_target[i],
                               layer_impute=layer_impute[i],
                               correlation=correlation[i],
                               remove_version=remove_version[i],
                               n_neighbors=n_neighbors[i],
                               relaxation=relaxation[i],
                               speed_factor=speed_factor[i],
                               n_trees=n_trees,
                               batch_size=batch_size,
                               seed=seed,
                               tmp_dir=tmp_dir,
                               verbose=verbose)
    else:
        low_mem_knn_impute(loom_source=loom_source,
                           layer_source=layer_source,
                           zscore_source=zscore_source,
                           feat_source=feat_source,
                           valid_ra_source=valid_ra_source,
                           valid_ca_source=valid_ca_source,
                           loom_target=loom_target,
                           layer_target=layer_target,
                           feat_target=feat_target,
                           valid_ra_target=valid_ra_target,
                           valid_ca_target=valid_ca_target,
                           layer_impute=layer_impute,
                           correlation=correlation,
                           remove_version=remove_version,
                           n_neighbors=n_neighbors,
                           relaxation=relaxation,
                           speed_factor=speed_factor,
                           n_trees=n_trees,
                           batch_size=batch_size,
                           seed=seed,
                           tmp_dir=tmp_dir,
                           verbose=verbose)
    # Clean-up files
    os.remove(zscore_source)


def low_mem_constrained_knn(loom_target,
                            knn_index,
                            layer_target,
                            valid_ca_target,
                            valid_ra_target,
                            feature_target,
                            loom_source,
                            zscore_source,
                            valid_ca_source,
                            valid_ra_source,
                            feature_source,
                            n_neighbors,
                            reverse_rank,
                            speed_factor,
                            relaxation,
                            n_trees,
                            batch_size,
                            remove_version,
                            tmp_dir,
                            seed,
                            verbose):
    # Start log
    if verbose:
        t0 = time.time()
        imp_log.info('Finding neighbors for {}'.format(loom_source))
    # Generate z-scored file
    zscore_target = temp_zscore_loom(loom_file=loom_target,
                                     raw_layer=layer_target,
                                     feat_attr=feature_target,
                                     valid_ca=valid_ca_target,
                                     valid_ra=valid_ra_target,
                                     batch_size=batch_size,
                                     tmp_dir=tmp_dir,
                                     verbose=verbose)

    # Prepare for kNN search
    col_target = utils.get_attr_index(loom_file=loom_target,
                                      attr=valid_ca_target,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_target = utils.get_attr_index(loom_file=loom_target,
                                      attr=valid_ra_target,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    col_source = utils.get_attr_index(loom_file=loom_source,
                                      attr=valid_ca_source,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_source = utils.get_attr_index(loom_file=loom_source,
                                      attr=valid_ra_source,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    with loompy.connect(loom_source) as ds:
        feat_select = ds.ra[feature_source][row_source]
    if remove_version:
        feat_select = utils.remove_gene_version(feat_select)
    n_target = np.sum(col_target)
    n_source = np.sum(col_source)
    accepted_idx = []
    accepted_cells = []
    rejected_cells = np.where(col_target)[0]
    k_saturate = int((n_target / n_source) * n_neighbors * relaxation) + 1
    with loompy.connect(loom_source) as ds:
        n_connects = np.repeat([k_saturate + 1], repeats=ds.shape[1])
    n_connects[col_source] = 0
    unsaturated = (n_connects < k_saturate)
    unsaturated_cells = np.where(unsaturated)[0]
    if seed is not None:
        np.random.seed(seed)
    # Perform search
    while rejected_cells.size != 0:
        # Get a random selection of cells
        np.random.shuffle(rejected_cells)
        if verbose:
            pct_rem = 100 - rejected_cells.shape[0] / n_target * 100
            pct_sat = unsaturated_cells.shape[0] / n_source * 100
            imp_log.info('{0:.2f}% of target cells have made connections'.format(pct_rem))
            imp_log.info('{0:.2f}% of source cells can make connections'.format(pct_sat))
        # Train the kNN
        t, index_file = low_mem_train_knn(loom_file=zscore_source,
                                          layer='',
                                          row_arr=row_source,
                                          col_arr=unsaturated_cells,
                                          feat_attr=feature_source,
                                          feat_select=feat_select,
                                          reverse_rank=False,
                                          remove_version=remove_version,
                                          tmp_dir=tmp_dir,
                                          seed=seed,
                                          batch_size=batch_size,
                                          verbose=verbose)
        # Build the kNN
        if verbose:
            imp_log.info('Building kNN')
        t.build(n_trees)
        # Query the kNN
        _, idx = low_mem_report_knn(loom_file=zscore_target,
                                    layer='',
                                    row_arr=row_target,
                                    col_arr=rejected_cells,
                                    feat_attr=feature_target,
                                    feat_select=feat_select,
                                    reverse_rank=reverse_rank,
                                    k=min(n_neighbors * speed_factor,
                                          unsaturated_cells.shape[0]),
                                    t=t,
                                    index_file=index_file,
                                    batch_size=batch_size,
                                    remove_version=remove_version,
                                    verbose=verbose)
        # Reindex values
        idx = unsaturated_cells[idx.astype(int)]
        rejected_local_idx = []
        for local_idx, cell in enumerate(rejected_cells):
            # Get all neighbors
            tmp_idx = idx[cell, :]
            # Remove saturated
            unsat_idx = unsaturated[tmp_idx]
            tmp_idx = tmp_idx[unsat_idx]
            # Determine if rejecting or accepting
            if tmp_idx.size < n_neighbors:
                rejected_local_idx.append(local_idx)
            else:
                accepted_idx.append(tmp_idx[:n_neighbors])
                accepted_cells.append(cell)
                n_connects[tmp_idx[:n_neighbors]] += 1
                unsaturated = (n_connects < k_saturate)  # unsaturated bool
        # Prep for next while loop
        unsaturated_cells = np.where(unsaturated)[0]
        rejected_cells = rejected_cells[rejected_local_idx]
    # Get final indices
    accepted_idx = pd.DataFrame(np.vstack(accepted_idx), index=accepted_cells)
    accepted_idx = accepted_idx.sort_index()
    with loompy.connect(loom_target) as ds:
        accepted_idx = accepted_idx.reindex(np.arange(ds.shape[1]))
        accepted_idx = accepted_idx.fillna(value=0)
        ds.ca[knn_index] = accepted_idx.values.astype(int)
    os.remove(zscore_target)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        imp_log.info(
            'Found neighbors for {0} in {1:.2f} {2}'.format(loom_source,
                                                            time_run,
                                                            time_fmt))


def low_mem_impute_data(loom_source,
                        layer_source,
                        feat_source,
                        valid_ca_source,
                        valid_ra_source,
                        loom_target,
                        layer_impute,
                        feat_target,
                        valid_ca_target,
                        valid_ra_target,
                        neighbor_index_target,
                        neighbor_index_source,
                        k_src_tar,
                        k_tar_src,
                        k_rescue,
                        ka,
                        epsilon,
                        pca_attr,
                        neighbor_method='knn',
                        remove_version=False,
                        offset=1e-5,
                        seed=None,
                        batch_size=5000,
                        verbose=False):
    """
    Performs imputation over a list (if provided) of layers imputing values in
    the source modality for the target data.

    Args:
        loom_source (str): Name of loom file that contains observed count data
        layer_source (str/list): Layer(s) containing observed count data
        feat_source (str): Row attribute specifying unique feature IDs
        valid_ca_source (str/None): Column attribute specifying columns to include
        valid_ra_source (str/None): Row attribute specifying rows to include
        loom_target (str): Name of loom file that will receive imputed counts
        layer_impute (str/list): Layer(s) that will contain imputed count data
        feat_target (str): Row attribute specifying unique feature IDs
        valid_ca_target (str/None): Column attribute specifying columns to include
        valid_ra_target (str/None): Row attribute specifying rows to include
        neighbor_index_target (str): Attribute containing indices for MNNs
        neighbor_index_source (str/None): Attribute containing indices for MNNs
        k_src_tar (int/None): Number of nearest neighbors for MNNs
        k_tar_src (int/None): Number of nearest neighbors for MNNs
        k_rescue (int/None): Number of nearest neighbors for rescue
        ka (int/None): If rescue, neighbor to normalize by
        epsilon (float/None): If MNN methods, epsilon value for Gaussian kernel
        pca_attr (str/None): If MNN methods, attribute containing PCs
        neighbor_method (str): How cells are chosen for imputation
            mnn_direct - include cells that did not make MNNs
            mnn_rescue - only include cells that made MNNs
            knn - use a restricted knn search to find neighbors
        remove_version (bool): Remove GENCODE version numbers from IDs
        offset (float): Offset for Markov normalization
        seed (int): Seed for Annoy
        batch_size (int): Size of batches
        verbose (bool): Print logging messages
    """
    if verbose:
        imp_log.info('Generating imputed data')
        t0 = time.time()
    # Get indices feature information
    fidx_tar = utils.get_attr_index(loom_file=loom_target,
                                    attr=valid_ra_target,
                                    columns=False,
                                    as_bool=True,
                                    inverse=False)
    fidx_src = utils.get_attr_index(loom_file=loom_source,
                                    attr=valid_ra_source,
                                    columns=False,
                                    as_bool=True,
                                    inverse=False)
    # Get relevant data from files
    with loompy.connect(filename=loom_source, mode='r') as ds:
        feat_src = ds.ra[feat_source]
    with loompy.connect(filename=loom_target, mode='r') as ds:
        num_feat = ds.shape[0]
        feat_tar = ds.ra[feat_target]
    # Determine features to include
    if remove_version:
        feat_tar = utils.remove_gene_version(gene_ids=feat_tar)
        feat_src = utils.remove_gene_version(gene_ids=feat_src)
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
    if neighbor_method == 'mnn_rescue':
        w_use = all_markov_self(loom_target=loom_target,
                                valid_target=valid_ca_target,
                                loom_source=loom_source,
                                valid_source=valid_ca_source,
                                neighbor_target=neighbor_index_target,
                                neighbor_source=neighbor_index_source,
                                k_src_tar=k_src_tar,
                                k_tar_src=k_tar_src,
                                k_rescue=k_rescue,
                                ka=ka,
                                epsilon=epsilon,
                                pca_attr=pca_attr,
                                offset=offset,
                                seed=seed,
                                verbose=verbose)
    elif neighbor_method == 'mnn_direct':
        w_use = get_markov_impute(loom_target=loom_target,
                                  loom_source=loom_source,
                                  valid_target=valid_ca_target,
                                  valid_source=valid_ca_source,
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
                               valid_target=valid_ca_target,
                               valid_source=valid_ca_source,
                               k=k_tar_src,
                               verbose=verbose)
    else:
        raise ValueError('Unsupported neighbor method')

    with loompy.connect(filename=loom_target) as ds_tar:
        # Make empty data
        ds_tar.layers[layer_impute] = sparse.coo_matrix(ds_tar.shape,
                                                        dtype=float)
        # Get index for batches
        valid_idx = np.unique(w_use.nonzero()[0])
        batches = np.array_split(valid_idx,
                                 np.ceil(valid_idx.shape[0] / batch_size))
        for batch in batches:
            tmp_w = w_use[batch, :]
            use_feat = np.unique(tmp_w.nonzero()[1])
            with loompy.connect(filename=loom_source, mode='r') as ds_src:
                tmp_dat = ds_src.layers[layer_source][:, use_feat][
                          feat_df['src'].values, :]
                tmp_dat = sparse.csr_matrix(tmp_dat.T)
            imputed = tmp_w[:, use_feat].dot(tmp_dat)
            imputed = utils.expand_sparse(mtx=imputed,
                                          col_index=feat_df[
                                              'tar'].values,
                                          col_n=num_feat)
            imputed = imputed.transpose()
            ds_tar.layers[layer_impute][:, batch] = imputed.toarray()
        valid_feat = np.zeros((ds_tar.shape[0],), dtype=int)
        valid_feat[feat_df['tar'].values] = 1
        ds_tar.ra['Valid_{}'.format(layer_impute)] = valid_feat
        valid_cells = np.zeros((ds_tar.shape[1],), dtype=int)
        valid_cells[valid_idx] = 1
        ds_tar.ca['Valid_{}'.format(layer_impute)] = valid_cells
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        imp_log.info('Imputed data in {0:.2f} {1}'.format(time_run, time_fmt))


def auto_find_k(loom_file,
                valid_ca,
                fraction,
                min_num,
                verbose):
    """
    Automatically finds an appropriate k value

    Args:
        loom_file (str): Path to loom file
        valid_ca (str): Row attribute specifying valid cells
        fraction (float): Find this fraction of cells
        min_num (int): Minimum k size
        verbose (bool): If true, print logging messages
    """
    # Get cell number
    cell_num = np.sum(utils.get_attr_index(loom_file=loom_file,
                                           attr=valid_ca,
                                           columns=True,
                                           as_bool=True,
                                           inverse=False))
    # Get fraction
    k = np.ceil(fraction * cell_num)
    # Round to nearest 10
    k = utils.round_nit(x=k,
                        units=10,
                        method='nearest')
    # Check minimum
    k = np.min([min_num, k])
    # Log
    if verbose:
        imp_log.info('{0} mutual k = {1}'.format(loom_file,
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
        imp_log.warning('ka is too large, resetting')
        ka = np.ceil(0.5 * k)
        if ka == k:
            raise ValueError('k value is too low')
        imp_log.warning('New ka is {}'.format(ka))
    else:
        ka = ka
    return ka


def low_mem_get_mnn(loom_target,
                    layer_target,
                    neighbor_distance_target,
                    neighbor_index_target,
                    max_k_target,
                    loom_source,
                    zscore_source,
                    neighbor_distance_source,
                    neighbor_index_source,
                    max_k_source,
                    correlation,
                    feature_id_target,
                    feature_id_source,
                    valid_ca_target=None,
                    valid_ra_target=None,
                    valid_ca_source=None,
                    valid_ra_source=None,
                    n_trees=10,
                    seed=None,
                    batch_size=5000,
                    remove_version=False,
                    tmp_dir=None,
                    verbose=False):
    """
    Gets kNN distances and indices by iterating over a loom file

    Args:
        loom_target (str): Path to loom file
        layer_target (str): Layer containing data for loom_target
        neighbor_distance_target (str): Output attribute for distances
        neighbor_index_target (str): Output attribute for indices
        max_k_target (int): Maximum number of nearest neighbors for target
        loom_source (str): Path to loom file
        zscore_source (str): Path to temporary loom file containing z-scored data
        neighbor_distance_source (str): Output attribute for distances
        neighbor_index_source (str): Output attribute for indices
        max_k_source  (int): Maximum number of nearest neighbors for source
        correlation (str): Expected direction of relationship
            positive or +
            negative or -
        feature_id_target (str): Row attribute containing unique feature IDs
        feature_id_source (str): Row attribute containing unique feature IDs
        valid_ca_target (str): Column attribute specifying valid cells
        valid_ra_target (str): Row attribute specifying valid features
        valid_ca_source (str): Column attribute specifying valid cells
        valid_ra_source (str): Row attribute specifying valid features
        n_trees (int): Number of trees to use for kNN
            more trees = more precision
        seed (int): Seed for Annoy
        batch_size (int): Size of chunks for iterating
        remove_version (bool): Remove GENCODE version IDs
        tmp_dir (str/None): Path to output directory for temporary files
            If None, writes to system default
        verbose (bool): Print logging messages
    """
    # Prep for function
    if verbose:
        imp_log.info('Finding MNN distances and indices')
        t0 = time.time()
    # Prep for kNN
    col_target = utils.get_attr_index(loom_file=loom_target,
                                      attr=valid_ca_target,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_target = utils.get_attr_index(loom_file=loom_target,
                                      attr=valid_ra_target,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    col_source = utils.get_attr_index(loom_file=loom_source,
                                      attr=valid_ca_source,
                                      columns=True,
                                      as_bool=True,
                                      inverse=False)
    row_source = utils.get_attr_index(loom_file=loom_source,
                                      attr=valid_ra_source,
                                      columns=False,
                                      as_bool=True,
                                      inverse=False)
    # Make lookup
    lookup_target = pd.Series(np.where(col_target)[0],
                              index=np.arange(np.sum(col_target)))
    lookup_source = pd.Series(np.where(col_source)[0],
                              index=np.arange(np.sum(col_source)))

    # Get features
    with loompy.connect(filename=loom_target) as ds:
        target_feat = ds.ra[feature_id_target][row_target]
    with loompy.connect(filename=loom_source) as ds:
        source_feat = ds.ra[feature_id_source][row_source]
    if remove_version:
        target_feat = utils.remove_gene_version(target_feat)
        source_feat = utils.remove_gene_version(source_feat)
    if np.any(np.sort(target_feat) != np.sort(source_feat)):
        raise ValueError('Feature mismatch!')
    if correlation.lower() in ['neg', 'negative', '-']:
        reverse_rank = True
    elif correlation.lower() in ['pos', 'positive', '+']:
        reverse_rank = False
    else:
        raise ValueError('Unsupported correlation value ({})'.format(correlation))
    # Make temporary files holding zscores
    zscore_target = temp_zscore_loom(loom_file=loom_target,
                                     raw_layer=layer_target,
                                     feat_attr=feature_id_target,
                                     valid_ca=valid_ca_target,
                                     valid_ra=valid_ra_target,
                                     batch_size=batch_size,
                                     tmp_dir=tmp_dir,
                                     verbose=verbose)
    # Train kNN
    t_s2t, s_idx = low_mem_train_knn(loom_file=zscore_target,
                                     layer='',
                                     row_arr=row_target,
                                     col_arr=col_target,
                                     feat_attr=feature_id_target,
                                     feat_select=target_feat,
                                     reverse_rank=reverse_rank,
                                     remove_version=remove_version,
                                     tmp_dir=tmp_dir,
                                     seed=seed,
                                     batch_size=batch_size,
                                     verbose=verbose)
    t_t2s, t_idx = low_mem_train_knn(loom_file=zscore_source,
                                     layer='',
                                     row_arr=row_source,
                                     col_arr=col_source,
                                     feat_attr=feature_id_source,
                                     feat_select=source_feat,
                                     reverse_rank=False,
                                     remove_version=remove_version,
                                     tmp_dir=tmp_dir,
                                     seed=seed,
                                     batch_size=batch_size,
                                     verbose=verbose)
    # Build trees
    if verbose:
        imp_log.info('Building kNN')
    t_t2s.build(n_trees)
    t_s2t.build(n_trees)
    # Get distances and indices
    dist_target, idx_target = low_mem_report_knn(loom_file=zscore_target,
                                                 layer='',
                                                 row_arr=row_target,
                                                 col_arr=col_target,
                                                 feat_attr=feature_id_target,
                                                 feat_select=target_feat,
                                                 reverse_rank=False,
                                                 k=max_k_target,
                                                 t=t_t2s,
                                                 index_file=t_idx,
                                                 batch_size=batch_size,
                                                 remove_version=remove_version,
                                                 verbose=verbose)
    dist_source, idx_source = low_mem_report_knn(loom_file=zscore_source,
                                                 layer='',
                                                 row_arr=row_source,
                                                 col_arr=col_source,
                                                 feat_attr=feature_id_source,
                                                 feat_select=source_feat,
                                                 reverse_rank=reverse_rank,
                                                 k=max_k_source,
                                                 t=t_s2t,
                                                 index_file=s_idx,
                                                 batch_size=batch_size,
                                                 remove_version=remove_version,
                                                 verbose=verbose)
    # Get correct indices (import if restricted to valid cells)
    correct_idx_target = np.reshape(lookup_source.loc[np.ravel(idx_target).astype(int)].values,
                                    idx_target.shape)
    correct_idx_source = np.reshape(lookup_target.loc[np.ravel(idx_source).astype(int)].values,
                                    idx_source.shape)
    # Add data to files
    with loompy.connect(filename=loom_target) as ds:
        ds.ca[neighbor_distance_target] = dist_target
        ds.ca[neighbor_index_target] = correct_idx_target
    with loompy.connect(filename=loom_source) as ds:
        ds.ca[neighbor_distance_source] = dist_source
        ds.ca[neighbor_index_source] = correct_idx_source
    # Remove temporary files
    os.remove(zscore_target)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        imp_log.info(
            'Found neighbors in {0:.2f} {1}'.format(time_run, time_fmt))


def low_mem_mnn_impute(loom_source,
                       layer_source,
                       zscore_source,
                       feat_source,
                       valid_ra_source,
                       valid_ca_source,
                       loom_target,
                       layer_target,
                       feat_target,
                       valid_ra_target,
                       valid_ca_target,
                       layer_impute,
                       correlation,
                       pca_target,
                       neighbor_method,
                       remove_version,
                       n_trees,
                       batch_size,
                       seed,
                       tmp_dir,
                       verbose):
    """
    Imputes data in one direction for one pair of data in a slow, low memory fashion

    Args:
        loom_source (str): Path to loom file that will provide counts to others
        layer_source (str): Layer in loom_source that will provide counts
        zscore_source (str): Path to loom file containing z-scored data
        feat_source (str): Row attribute containing unique feature IDs
        valid_ra_source (str): Row attribute specifying features that can be used
        valid_ca_source (str): Column attribute specifying cells that can be used
        loom_target (str/list): Path(s) to loom files that will receive imputed counts
        layer_target (str): Layer in loom_target that contains counts for correlations
        feat_target (str): Row attribute containing unique feature IDs
        valid_ra_target (str): Row attribute specifying features that can be used
        valid_ca_target (str): Column attribute specifying cells that can be used
        layer_impute (str): Output layer for loom_target that will receive imputed counts
        correlation (str): Expected correlation (negative/-, positive/+)
        pca_target (str/None): If mnn_rescue, attribute containing PCs in loom_target
        neighbor_method (str): How cells are chosen for imputation
            mnn_direct - include cells that did not make MNNs
            mnn_rescue - only include cells that made MNNs
            knn - use a restricted knn search to find neighbors
        remove_version (bool): If true, remove GENCODE version ID from feat_source/feat_target
        n_trees (int): Number of trees for kNN search
        batch_size (int): Size of chunks for batch iterations
        seed (int): Seed for randomization
        tmp_dir (str): Path to output directory
        verbose (bool): If true, print logging messages

    """
    # Start log
    if verbose:
        t0 = time.time()
        imp_log.info('Imputing from {0} to {1}'.format(loom_source,
                                                       loom_target))
    # Determine k-values (hack, in future should let user chose)
    mutual_k_target = auto_find_k(loom_file=loom_target,
                                  valid_ca=valid_ca_target,
                                  fraction=0.01,
                                  min_num=200,
                                  verbose=verbose)
    mutual_k_source = auto_find_k(loom_file=loom_source,
                                  valid_ca=valid_ca_source,
                                  fraction=0.01,
                                  min_num=200,
                                  verbose=verbose)
    mutual_k_max = np.max([mutual_k_target, mutual_k_source])
    rescue_k_target = auto_find_k(loom_file=loom_target,
                                  valid_ca=valid_ca_target,
                                  fraction=0.001,
                                  min_num=50,
                                  verbose=verbose)
    ka_target = check_ka(k=rescue_k_target,
                         ka=5)
    # Get MNNs
    low_mem_get_mnn(loom_target=loom_target,
                    layer_target=layer_target,
                    neighbor_distance_target='mnn_dist',  # hack, need to update
                    neighbor_index_target='mnn_index',  # hack, need to update
                    max_k_target=mutual_k_max,
                    loom_source=loom_source,
                    zscore_source=zscore_source,
                    neighbor_distance_source='mnn_dist',  # hack, need to update
                    neighbor_index_source='mnn_index',  # hack, need to update
                    max_k_source=mutual_k_max,
                    correlation=correlation,
                    feature_id_target=feat_target,
                    feature_id_source=feat_source,
                    valid_ca_target=valid_ca_target,
                    valid_ra_target=valid_ra_target,
                    valid_ca_source=valid_ca_source,
                    valid_ra_source=valid_ra_source,
                    n_trees=n_trees,
                    seed=seed,
                    batch_size=batch_size,
                    remove_version=remove_version,
                    tmp_dir=tmp_dir,
                    verbose=verbose)
    # Impute data
    low_mem_impute_data(loom_source=loom_source,
                        layer_source=layer_source,
                        feat_source=feat_source,
                        valid_ca_source=valid_ca_source,
                        valid_ra_source=valid_ra_source,
                        loom_target=loom_target,
                        layer_impute=layer_impute,
                        feat_target=feat_target,
                        valid_ca_target=valid_ca_target,
                        valid_ra_target=valid_ra_target,
                        neighbor_index_target='mnn_dist',  # hack, need to update
                        neighbor_index_source='mnn_index',  # hack, need to update
                        k_src_tar=mutual_k_source,
                        k_tar_src=mutual_k_target,
                        k_rescue=rescue_k_target,
                        ka=ka_target,
                        epsilon=1,  # hack, need to update
                        pca_attr=pca_target,
                        neighbor_method=neighbor_method,
                        remove_version=remove_version,
                        offset=1e-5,
                        seed=seed,
                        batch_size=batch_size,
                        verbose=verbose)

    # Log
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        imp_log.info('Imputed from {0} to {1} in {2:.2f} {3}'.format(loom_source,
                                                                     loom_target,
                                                                     time_run,
                                                                     time_fmt))


def low_mem_mnn_1d(loom_source,
                   loom_target,
                   layer_source='',
                   layer_target='',
                   layer_impute='imputed',
                   correlation='positive',
                   feat_source='Accession',
                   feat_target='Accession',
                   cell_target='CellID',
                   pca_target=None,
                   valid_ra_source=None,
                   valid_ra_target=None,
                   valid_ca_source=None,
                   valid_ca_target=None,
                   neighbor_method='mnn_rescue',
                   n_trees=10,
                   remove_version=False,
                   tmp_dir=None,
                   batch_size=5000,
                   seed=None,
                   verbose=False):
    """
    Imputes data from a given data modality (source) into another (target) with low memory but slowly

    Args:
        loom_source (str): Path to loom file that will provide counts to others
        loom_target (str/list): Path(s) to loom files that will receive imputed counts
        layer_source (str): Layer in loom_source that will provide counts
        layer_target (str/list): Layer(s) in loom_target files specifying counts
            Used for finding correlations between loom files
        layer_impute (str/list): Output layer in loom_target
            A row attribute with the format Valid_{layer_out} will be added
            A col attribute with the format Valid_{layer_out} will be added
        correlation (str/list): Expected correlation between loom_source and loom_target
            positive/+ for RNA-seq and ATAC-seq
            negative/- for RNA-seq or ATAC-seq and snmC-seq
        feat_source (str): Row attribute specifying unique feature names in loom_source
        feat_target (str/list): Row attribute(s) specifying unique feature names in loom_target
        cell_target (str/list): Column attribute specifying unique cell IDs in loom_target
        pca_target (str/list/None): Column attribute containing PCs in loom_target
            Used if neighbor_method == mnn_rescue
        valid_ra_source (str): Row attribute specifying valid features in loom_source
            Should point to a boolean array
        valid_ra_target (str/list): Row attribute(s) specifying valid features in loom_target
            Should point to a boolean array
        valid_ca_source (str): Column attribute specifying valid cells in loom_source
            Should point to a boolean array
        valid_ca_target (str/list): Column attribute(s) specifying valid cells in loom_target
            Should point to a boolean array
        neighbor_method (str): How cells are chosen for imputation
            mnn_direct - include cells that did not make MNNs
            mnn_rescue - only include cells that made MNNs
        n_trees (int): Number of trees for approximate kNN search
            See Annoy documentation
        remove_version (bool/list): If true remove version number
            Anything after the first period is dropped (useful for GENCODE IDs)
            If a list, will behave differently for each loom file
            If a boolean, will behave the same for each loom_file
        tmp_dir (str): Optional, path to output directory for temporary files
            If None, uses default temporary directory on your system
        batch_size (int): Number of elements per chunk when analyzing in batches
        seed (int): Initialization for random seed
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
                            valid_ra_target,
                            valid_ca_target,
                            remove_version,
                            layer_impute,
                            pca_target]
        checked = utils.mimic_list(parameters=check_parameters,
                                   list_len=len(loom_target))
        feat_target = checked[0]
        valid_ra_target = checked[1]
        valid_ca_target = checked[2]
        remove_version = checked[3]
        layer_impute = checked[4]
        pca_target = checked[5]
        is_a_list = True
    elif isinstance(loom_target, str):
        utils.all_same_type_size(parameters=[loom_target,
                                             layer_target,
                                             correlation,
                                             feat_target,
                                             cell_target],
                                 expected_type='str',
                                 confirm_size=False)
    if verbose:
        imp_log.info('Preparing for imputation')
    # Get source z-scored loom file
    zscore_source = temp_zscore_loom(loom_file=loom_source,
                                     raw_layer=layer_source,
                                     feat_attr=feat_source,
                                     valid_ca=valid_ca_source,
                                     valid_ra=valid_ra_source,
                                     batch_size=batch_size,
                                     tmp_dir=tmp_dir,
                                     verbose=verbose)
    # Impute for each target
    if is_a_list:
        for i in np.arange(len(loom_target)):
            low_mem_mnn_impute(loom_source=loom_source,
                               layer_source=layer_source,
                               zscore_source=zscore_source,
                               feat_source=feat_source,
                               valid_ra_source=valid_ra_source,
                               valid_ca_source=valid_ca_source,
                               loom_target=loom_target[i],
                               layer_target=layer_target[i],
                               feat_target=feat_target[i],
                               valid_ra_target=valid_ra_target[i],
                               valid_ca_target=valid_ca_target[i],
                               layer_impute=layer_impute[i],
                               correlation=correlation[i],
                               neighbor_method=neighbor_method,
                               pca_target=pca_target[i],
                               remove_version=remove_version[i],
                               n_trees=n_trees,
                               batch_size=batch_size,
                               seed=seed,
                               tmp_dir=tmp_dir,
                               verbose=verbose)
    else:
        low_mem_mnn_impute(loom_source=loom_source,
                           layer_source=layer_source,
                           zscore_source=zscore_source,
                           feat_source=feat_source,
                           valid_ra_source=valid_ra_source,
                           valid_ca_source=valid_ca_source,
                           loom_target=loom_target,
                           layer_target=layer_target,
                           feat_target=feat_target,
                           valid_ra_target=valid_ra_target,
                           valid_ca_target=valid_ca_target,
                           layer_impute=layer_impute,
                           correlation=correlation,
                           neighbor_method=neighbor_method,
                           pca_target=pca_target,
                           remove_version=remove_version,
                           n_trees=n_trees,
                           batch_size=batch_size,
                           seed=seed,
                           tmp_dir=tmp_dir,
                           verbose=verbose)
    # Clean-up files
    os.remove(zscore_source)


def perform_imputation(loom_source,
                       loom_target,
                       method='knn',
                       layer_source='',
                       layer_target='',
                       layer_impute='imputed',
                       correlation='positive',
                       feat_source='Accession',
                       feat_target='Accession',
                       cell_source='CellID',
                       cell_target='CellID',
                       pca_target=None,
                       valid_ra_source=None,
                       valid_ra_target=None,
                       valid_ca_source=None,
                       valid_ca_target=None,
                       n_neighbors=20,
                       relaxation=10,
                       speed_factor=10,
                       n_trees=10,
                       remove_version=False,
                       low_mem=False,
                       batch_size=5000,
                       tmp_dir=None,
                       seed=None,
                       verbose=False):
    """
        Imputes data from a given data modality (source) into another (target)

        Args:
            loom_source (str): Path to loom file that will provide counts to others
            loom_target (str/list): Path(s) to loom files that will receive imputed counts
            method (str): Method for performing imputation
                knn: constrained knn
                mnn_direct: mutual nearest neighbors with rescue of cells (can only be done if low_mem = False)
                mnn_rescue: mutual nearest neighbors with cells that made direct neighbors (only if low_mem = False)
            layer_source (str): Layer in loom_source that will provide counts
            layer_target (str/list): Layer(s) in loom_target files specifying counts
                Used for finding correlations between loom files
            layer_impute (str/list): Output layer in loom_target
                A row attribute with the format Valid_{layer_out} will be added
                A col attribute with the format Valid_{layer_out} will be added
            correlation (str/list): Expected correlation between loom_source and loom_target
                positive/+ for RNA-seq and ATAC-seq
                negative/- for RNA-seq or ATAC-seq and snmC-seq
            feat_source (str): Row attribute specifying unique feature names in loom_source
            feat_target (str/list): Row attribute(s) specifying unique feature names in loom_target
            cell_source (str): Column attribute specifying unique cell IDs in loom_source
            cell_target (str/list): Column attribute specifying unique cell IDs in loom_target
            pca_target (str/list/None): Attribute containing PCs in loom_target
                Used if method is mnn_rescue
            valid_ra_source (str): Row attribute specifying valid features in loom_source
                Should point to a boolean array
            valid_ra_target (str/list): Row attribute(s) specifying valid features in loom_target
                Should point to a boolean array
            valid_ca_source (str): Column attribute specifying valid cells in loom_source
                Should point to a boolean array
            valid_ca_target (str/list): Column attribute(s) specifying valid cells in loom_target
                Should point to a boolean array
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
    if method == 'knn':
        if low_mem:
            low_mem_constrained_1d(loom_source=loom_source,
                                   loom_target=loom_target,
                                   layer_source=layer_source,
                                   layer_target=layer_target,
                                   layer_impute=layer_impute,
                                   correlation=correlation,
                                   feat_source=feat_source,
                                   feat_target=feat_target,
                                   cell_target=cell_target,
                                   valid_ra_source=valid_ra_source,
                                   valid_ra_target=valid_ra_target,
                                   valid_ca_source=valid_ca_source,
                                   valid_ca_target=valid_ca_target,
                                   n_neighbors=n_neighbors,
                                   relaxation=relaxation,
                                   speed_factor=speed_factor,
                                   n_trees=n_trees,
                                   remove_version=remove_version,
                                   tmp_dir=tmp_dir,
                                   batch_size=batch_size,
                                   seed=seed,
                                   verbose=verbose)
        else:
            high_mem_constrained_1d(loom_source=loom_source,
                                    loom_target=loom_target,
                                    layer_source=layer_source,
                                    layer_target=layer_target,
                                    layer_impute=layer_impute,
                                    correlation=correlation,
                                    feat_source=feat_source,
                                    feat_target=feat_target,
                                    cell_source=cell_source,
                                    cell_target=cell_target,
                                    valid_ra_source=valid_ra_source,
                                    valid_ra_target=valid_ra_target,
                                    valid_ca_source=valid_ca_source,
                                    valid_ca_target=valid_ca_target,
                                    n_neighbors=n_neighbors,
                                    relaxation=relaxation,
                                    speed_factor=speed_factor,
                                    n_trees=n_trees,
                                    remove_version=remove_version,
                                    seed=seed,
                                    verbose=verbose)
    elif method == 'mnn_rescue':
        if low_mem:
            low_mem_mnn_1d(loom_source=loom_source,
                           loom_target=loom_target,
                           layer_source=layer_source,
                           layer_target=layer_target,
                           layer_impute=layer_impute,
                           correlation=correlation,
                           feat_source=feat_source,
                           feat_target=feat_target,
                           cell_target=cell_target,
                           pca_target=pca_target,
                           valid_ra_source=valid_ra_source,
                           valid_ra_target=valid_ra_target,
                           valid_ca_source=valid_ca_source,
                           valid_ca_target=valid_ca_target,
                           neighbor_method=method,
                           n_trees=n_trees,
                           remove_version=remove_version,
                           tmp_dir=tmp_dir,
                           batch_size=batch_size,
                           seed=seed,
                           verbose=verbose)
        else:
            # TO DO: ADD LOW MEMORY VERSION
            raise ValueError('mnn_rescue can only be performed if low_mem is true')
    elif method == 'mnn_direct':
        if low_mem:
            low_mem_mnn_1d(loom_source=loom_source,
                           loom_target=loom_target,
                           layer_source=layer_source,
                           layer_target=layer_target,
                           layer_impute=layer_impute,
                           correlation=correlation,
                           feat_source=feat_source,
                           feat_target=feat_target,
                           cell_target=cell_target,
                           pca_target=pca_target,
                           valid_ra_source=valid_ra_source,
                           valid_ra_target=valid_ra_target,
                           valid_ca_source=valid_ca_source,
                           valid_ca_target=valid_ca_target,
                           neighbor_method=method,
                           n_trees=n_trees,
                           remove_version=remove_version,
                           tmp_dir=tmp_dir,
                           batch_size=batch_size,
                           seed=seed,
                           verbose=verbose)
        else:
            # TO DO: ADD LOW MEMORY VERSION
            raise ValueError('mnn_direct can only be performed if low_mem is true')
    else:
        raise ValueError('method must be knn, mnn_direct, or mnn_rescue not {}'.format(method))
