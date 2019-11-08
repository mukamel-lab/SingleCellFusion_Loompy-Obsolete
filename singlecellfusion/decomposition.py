"""
Functions used to perform dimensionality reduction on loom files

Written by Wayne Doyle

(C) 2018 Mukamel Lab GPLv2
"""
import loompy
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
import fbpca
import logging
import time
from . import utils

# Start log
decomp_log = logging.getLogger(__name__)


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
            decomp_log.error(err_msg)
        raise ValueError(decomp_log)
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
        decomp_log.info('Adjusted batch size to {0} for PCA'.format(batch_size))
    # Return value
    return batch_size


def prep_pca(view,
             layer,
             row_idx,
             scale_attr=None):
    """
    Performs data processing for PCA on a given layer

    Args:
        view (loompy object): Slice of loom file
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


def batch_pca(loom_file,
              layer,
              out_attr='PCA',
              valid_ca=None,
              valid_ra=None,
              scale_attr=None,
              n_pca=50,
              drop_first=False,
              batch_size=512,
              verbose=False):
    """
    Performs incremental PCA on a loom file

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer containing data for PCA
        out_attr (str): Name of PCA attribute
            Valid_{out_attr} will also be added to indicate used cells
        valid_ca (str): Optional, only use cells specified by valid_ca
        valid_ra (str): Optional, only use features specified by valid_ra
        scale_attr (str): Optional, attribute specifying cell scaling factor
        n_pca (int): Number of components for PCA
        drop_first (bool): Drops first PC
            Useful if the first PC correlates with a technical feature
            If true, a total of n_pca is still generated and added to loom_file
            If true, the first principal component will be lost
        batch_size (int): Number of elements per chunk
        verbose (bool): If true, print logging messages

    Returns:
        Adds components to ds.ca.{out_attr}
        Adds quality control to ds.ca.Valid_{out_attr}
    """
    if verbose:
        decomp_log.info('Fitting PCA')
        t0 = time.time()
    if drop_first:
        n_tmp = n_pca + 1
    else:
        n_tmp = n_pca
    # Check user's batch size
    batch_size = check_pca_batches(loom_file=loom_file,
                                   n_pca=n_pca,
                                   batch_size=batch_size,
                                   verbose=verbose)
    # Perform PCA
    pca = IncrementalPCA(n_components=n_tmp)
    with loompy.connect(loom_file) as ds:
        ds.ca[out_attr] = np.zeros((ds.shape[1], n_pca), dtype=float)
        n = ds.ca[out_attr].shape[0]
        # Get column and row indices
        col_idx = utils.get_attr_index(loom_file=loom_file,
                                       attr=valid_ca,
                                       columns=True,
                                       inverse=False)
        row_idx = utils.get_attr_index(loom_file=loom_file,
                                       attr=valid_ra,
                                       columns=False,
                                       inverse=False)
        # Fit model
        layers = utils.make_layer_list(layers=layer)
        for (_, _, view) in ds.scan(items=col_idx,
                                    layers=layers,
                                    axis=1,
                                    batch_size=batch_size):
            dat = prep_pca(view=view,
                           layer=layer,
                           row_idx=row_idx,
                           scale_attr=scale_attr)
            pca.partial_fit(dat)
        if verbose:
            t1 = time.time()
            time_run, time_fmt = utils.format_run_time(t0, t1)
            decomp_log.info('Fit PCA in {0:.2f} {1}'.format(time_run, time_fmt))
        # Transform
        for (_, selection, view) in ds.scan(items=col_idx,
                                            layers=layers,
                                            axis=1,
                                            batch_size=batch_size):
            dat = prep_pca(view=view,
                           layer=layer,
                           row_idx=row_idx,
                           scale_attr=scale_attr)
            dat = pca.transform(dat)
            if drop_first:
                dat = dat[:, 1:]
            mask = selection == np.arange(n)[:, None]
            ds.ca[out_attr] += mask.dot(dat)
        # Add to file
        if valid_ca:
            ds.ca['Valid_{}'.format(out_attr)] = ds.ca[valid_ca]
        else:
            ds.ca['Valid_{}'.format(out_attr)] = np.ones((ds.shape[1],),
                                                         dtype=int)
        # Log
        if verbose:
            t2 = time.time()
            time_run, time_fmt = utils.format_run_time(t1, t2)
            decomp_log.info(
                'Reduced dimensions in {0:.2f} {1}'.format(time_run, time_fmt))


def high_mem_pca(loom_file,
                 layer='',
                 n_pca=50,
                 cell_attr='CellID',
                 valid_ca=None,
                 valid_ra=None,
                 verbose=False):
    """
    Gets a data frame of PCs for a given set of data from a loom file

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer in loom_file containing data
        n_pca (int): Number of PCs to obtain
        cell_attr (str): Column attribute containing unique cell IDs
        valid_ca (str/None): Column attribute specifying cells to include
        valid_ra (str/None): Row attribute specifying features to include
        verbose (bool): If true, print logging messages
    """
    if verbose:
        decomp_log.info('Running PCA on {}'.format(loom_file))
        t0 = time.time()
    # Get indices
    row_idx = utils.get_attr_index(loom_file=loom_file,
                                   attr=valid_ra,
                                   as_bool=False,
                                   inverse=False)
    col_idx = utils.get_attr_index(loom_file=loom_file,
                                   attr=valid_ca,
                                   as_bool=False,
                                   inverse=False)
    # Get data
    with loompy.connect(loom_file) as ds:
        dat = ds.layers[layer].sparse(row_idx, col_idx).todense().T
        cell_ids = ds.ca[cell_attr][col_idx]
    # Perform PCA
    u, s, v = fbpca.pca(dat, n_pca)
    pcs = u.dot(np.diag(s))
    sigma = np.sqrt(np.sum(s * s) / (pcs.shape[0] * pcs.shape[1]))
    pcs = pcs / sigma
    # Make into dataframe
    pcs = pd.DataFrame(pcs,
                       index=cell_ids,
                       columns=np.arange(n_pca))
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        decomp_log.info('Reduced dimensions in {0:.2f} {1}'.format(time_run, time_fmt))
    return pcs
