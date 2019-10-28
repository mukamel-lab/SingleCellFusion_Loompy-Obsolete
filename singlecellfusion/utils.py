"""
Utility functions that can be run either as sub-routines or independently

Written by Wayne Doyle unless otherwise noted

(C) 2019 Mukamel Lab GPLv2
"""
import numpy as np
import loompy
import os
import re
from scipy import sparse
from scipy import stats
from sklearn.utils import sparsefuncs
import time
import logging

# Start log
util_log = logging.getLogger(__name__)

bin_dir = os.path.dirname(os.path.realpath(__file__))


# General utilities
def round_unit(x,
               units=10,
               method='ceil'):
    """
    Rounds a number to the nearest unit

    Args:
        x (int/float): A number
        units (int): Nearest base to round to
        method (str): Method for rounding
            ceil: Round up
            floor: Round down
            nearest: Round up or down, whichever is closest
                If equal, performs ceil

    Returns:
        y (int): x to the nearest unit

    Based off of Parker's answer on StackOverflow:
    https://stackoverflow.com/questions/26454649/...
    python-round-up-to-the-nearest-ten
    """
    if method == 'ceil':
        y = int(np.ceil(x / units)) * units
    elif method == 'floor':
        y = int(np.floor(x / units)) * units
    elif method == 'nearest':
        highest = int(np.ceil(x / units)) * units
        lowest = int(np.floor(x / units)) * units
        high_diff = np.abs(highest - x)
        low_diff = np.abs(lowest - x)
        if lowest == 0 or high_diff < low_diff:
            y = highest
        else:
            y = lowest
    else:
        util_log.error('Improper value for method')
        raise ValueError
    return y


def alphanum_key(item):
    """
    Key function for nat_sort

    Args:
        item (str): Value to sort

    Based on Mark Byer's post on StackOverflow:
    https://stackoverflow.com/questions/...
    4836710/does-python-have-a-built-in-function-for-string-natural-sort

    """
    keys = []
    item = str(item)
    for i in re.split('([0-9]+)', item):
        if i.isdigit():
            i = int(i)
        else:
            i = i.lower()
        keys.append(i)
    return keys


def nat_sort(items):
    """
    Takes a list of items and sorts them in a natural order

    Args:
        items (list): List of items

    Based on Mark Byer's post on StackOverflow:
    https://stackoverflow.com/questions/...
    4836710/does-python-have-a-built-in-function-for-string-natural-sort
    """
    return sorted(items, key=alphanum_key)


def format_run_time(t0, t1):
    """
    Formats the time between two points into human-friendly format
    
    Args:
        t0 (float): Output of time.time()
        t1 (float): Output of time.time()
    
    Returns:
        time_run (float): Elapsed time in human-friendly format
        time_fmt (str): Unit of time_run
    """
    time_run = t1 - t0
    if time_run > 86400:
        time_run = time_run / 86400
        time_fmt = 'days'
    elif time_run > 3600:
        time_run = time_run / 3600
        time_fmt = 'hours'
    elif time_run > 60:
        time_run = time_run / 60
        time_fmt = 'minutes'
    else:
        time_fmt = 'seconds'
    return [time_run, time_fmt]


def get_mouse_chroms(prefix=False,
                     include_y=False):
    """
    Returns a dictionary of chromosomes and their sizes (in bases)
    
    Args:
        prefix (bool): If true, include chr prefix
        include_y (bool): If true include Y chromosome
    
    Returns:
        chrom_dict (dict): keys are chromosomes, values are lengths
    """
    chrom_dict = {'1': 195471971,
                  '2': 182113224,
                  '3': 160039680,
                  '4': 156508116,
                  '5': 151834684,
                  '6': 149736546,
                  '7': 145441459,
                  '8': 129401213,
                  '9': 124595110,
                  '10': 130694993,
                  '11': 122082543,
                  '12': 120129022,
                  '13': 120421639,
                  '14': 124902244,
                  '15': 104043685,
                  '16': 98207768,
                  '17': 94987271,
                  '18': 90702639,
                  '19': 61431566,
                  'X': 171031299,
                  }
    if include_y:
        chrom_dict['Y'] = 91744698
    if prefix:
        mod = dict()
        for key in chrom_dict.keys():
            new_key = 'chr' + key
            mod[new_key] = chrom_dict[key]
        chrom_dict = mod
    return chrom_dict


def expand_sparse(mtx,
                  col_index=None,
                  row_index=None,
                  col_n=None,
                  row_n=None,
                  dtype=float):
    """
    Expands a sparse matrix
    
    Args:
        mtx (sparse 2D array): Matrix from a subset of loom file
        col_index (1D array): Numerical indices of columns included in mtx
        row_index (1D array): Numerical indices of rows included in mtx
        col_n (int): Number of loom file columns
        row_n (int): Number of loom file rows
        dtype (str): Type of data in output matrix
    
    Returns:
        mtx (sparse 2D array): mtx with missing values included as zeros

    Warning:
        Do not use on transposed matrices
    """
    mtx = mtx.tocoo()
    if col_index is None and row_index is None:
        pass
    elif col_index is not None and col_n is None:
        raise ValueError('Must provide both col_index and col_n')
    elif row_index is not None and row_n is None:
        raise ValueError('Must provide both row_index and row_n')
    elif col_index is None:
        mtx = sparse.coo_matrix((mtx.data,
                                 (row_index[mtx.nonzero()[0]],
                                  mtx.nonzero()[1])),
                                shape=(row_n, mtx.shape[1]),
                                dtype=dtype)
    elif row_index is None:
        mtx = sparse.coo_matrix((mtx.data,
                                 (mtx.nonzero()[0],
                                  col_index[mtx.nonzero()[1]])),
                                shape=(mtx.shape[0], col_n),
                                dtype=dtype)
    else:
        mtx = sparse.coo_matrix((mtx.data,
                                 (row_index[mtx.nonzero()[0]],
                                  col_index[mtx.nonzero()[1]])),
                                shape=(row_n, col_n),
                                dtype=dtype)
    return mtx


def remove_gene_version(gene_ids):
    """
    Goes through an array of gene IDs and removes version numbers
        Useful for GENCODE
        
    Args:
        gene_ids (1D array): gene IDs
    
    Returns:
        gene_ids (1D array): Gene IDs with version numbers removed
    
    Assumptions:
        The only period in the gene ID is directly before the gene version
    """
    gene_ids = np.array(list(map(lambda x: re.sub(r'\..*$', '', x), gene_ids)))
    return gene_ids


def make_nan_array(num_rows, num_cols):
    """
    Makes an array of NaN values

    Args:
        num_rows (int): Number of rows for output array
        num_cols (int): Number of columns for output array

    Returns:
        nan_array (ndarray): Array of NaN values
    """
    nan_array = np.empty((num_rows, num_cols))
    nan_array.fill(np.nan)
    return nan_array


# Loom utilities
def get_pct(loom_file,
            num_val,
            axis=0):
    """
    Calculates the percentage of a given number over a given loom axis

    Args:
        loom_file (str): Path to loom file
        num_val (int): Number to calculate percentage with
        axis (int): Axis to calculate percentage with
            0: rows
            1: columns

    Returns:
        pct (float): Percentage of num_val/axis * 100
    """
    if axis == 0 or axis == 1:
        with loompy.connect(filename=loom_file, mode='r') as ds:
            pct = num_val / ds.shape[axis] * 100
    else:
        raise ValueError('Axis must be 0 or 1')
    return pct


def get_attr_index(loom_file,
                   attr=None,
                   columns=False,
                   as_bool=True,
                   inverse=False):
    """
    Gets index for desired attributes in a loom file

    Args:
        loom_file (str): Path to loom file
        attr (str): Optional, attribute used to restrict index
            If None, all elements are included
        columns (boolean): Specifies if pulling rows or columns
            True: column attributes
            False: row attributes
        as_bool (bool): Return as boolean (true) or numerical (false) array
        inverse (bool): If true, returns inverse of index
            All trues are false, all falses are true

    Returns:
        idx (1D array): Index of attributes to use
            boolean if as_bool, numerical if not as_bool

    Assumptions:
        attr specifies a boolean array attribute in loom_file
    """

    with loompy.connect(filename=loom_file, mode='r') as ds:
        if columns:
            if attr:
                idx = ds.ca[attr].astype(bool)
            else:
                idx = np.ones((ds.shape[1],), dtype=bool)
        else:
            if attr:
                idx = ds.ra[attr].astype(bool)
            else:
                idx = np.ones((ds.shape[0],), dtype=bool)
    if inverse:
        idx = np.logical_not(idx)
    if as_bool:
        pass
    else:  # ASSUMPTION: 1D array input
        idx = np.where(idx)[0]
    return idx


def make_layer_list(layers):
    """
    Makes a list of layers to include when looping over a loom file

    Args:
        layers (str/list): Layer(s) in loom file to include

    Returns:
        out (list): Layer(s) in loom file to include
            Transformed to list and '' is added if not included
    """
    if isinstance(layers, str):
        if layers == '':
            out = ['']
        else:
            out = ['', layers]
    elif isinstance(layers, list):
        layers = set(layers)
        if '' in layers:
            out = list(layers)
        else:
            layers.add('')
            out = list(layers)
    else:
        raise ValueError('Unsupported type for layers')
    return out


def high_mem_mean_and_std(loom_file,
                          layer,
                          axis=None,
                          valid_ca=None,
                          valid_ra=None):
    """
    Calculates mean and standard deviation in a high memory fashion

    Args:
        loom_file (str): Path to loom file containing mC/C counts
        layer (str): Layer containing mC/C counts
        axis (int): Axis to calculate mean and standard deviation
            None: values are for entire layer
            0: Statistics are for cells
            1: Statistics are for features
        valid_ca (str): Optional, only use cells specified by valid_ca
        valid_ra (str): Optional, only use features specified by valid_ra
    """
    # Get valid indices
    row_idx = get_attr_index(loom_file=loom_file,
                             attr=valid_ra,
                             columns=False,
                             as_bool=False,
                             inverse=False)
    col_idx = get_attr_index(loom_file=loom_file,
                             attr=valid_ca,
                             columns=True,
                             as_bool=False,
                             inverse=False)
    # Get data
    with loompy.connect(loom_file, mode='r') as ds:
        dat = ds.layers[layer].sparse(row_idx, col_idx)
    # Get mean and variance
    if axis == 0:
        my_mean, my_var = sparsefuncs.mean_variance_axis(dat.tocsc(), axis=0)
        my_std = np.sqrt(my_var)
    elif axis == 1:
        my_mean, my_var = sparsefuncs.mean_variance_axis(dat.tocsr(), axis=1)
        my_std = np.sqrt(my_var)
    elif axis is None:
        my_mean = dat.tocsr().mean(axis=None)
        sqrd = dat.copy()
        sqrd.data **= 2
        my_var = sqrd.sum(axis=None) / (sqrd.shape[0] * sqrd.shape[1]) - my_mean ** 2
        my_std = np.sqrt(my_var)
    else:
        raise ValueError('Unsupported axis value ({})'.format(axis))
    return my_mean, my_std


def low_mem_mean_and_std(loom_file,
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
        util_log.info('Calculating statistics for {}'.format(loom_file))
        t0 = time.time()
    # Get indices
    layers = make_layer_list(layers=layer)
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
    col_idx = get_attr_index(loom_file=loom_file,
                             attr=valid_ca,
                             columns=True,
                             inverse=False)
    row_idx = get_attr_index(loom_file=loom_file,
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
        time_run, time_fmt = format_run_time(t0, t1)
        util_log.info(
            'Calculated statistics in {0:.2f} {1}'.format(time_run, time_fmt))
    return [my_mean, my_std]


def all_same_type_size(parameters,
                       expected_type,
                       confirm_size=False):
    """
    Checks if a list of parameters are all the same instance and length

    Args:
        parameters (list): List of items to check
        expected_type (str): Expected instance (list,str)
        confirm_size (bool): If true, make sure all the same length
    """
    sizes = None
    if isinstance(parameters, list):
        for item in parameters:
            # Make sure it is expected type
            if expected_type == 'str':
                if not isinstance(item, str):
                    raise ValueError('Item type is not a string')
            elif expected_type == 'list':
                if not isinstance(item, list):
                    raise ValueError('Item type is not a list')
            else:
                raise ValueError('Unsupported expected_type ({})'.format(expected_type))
            # Make sure the size is correct
            if confirm_size and sizes is None:
                sizes = len(expected_type)
            elif confirm_size and sizes != len(expected_type):
                raise ValueError('parameters are diferent sizes')
            else:
                pass
    else:
        raise ValueError('parameters must be a list')


def mimic_list(parameters,
               list_len):
    """
    Transforms a passed string or None parameter into a list

    Args:
        parameters (list): Parameters to make into a list
        list_len (int): Length of output list

    Returns:
        out_param (list): Input parameters as a list
    """
    out_param = []
    if isinstance(parameters, list):
        for item in parameters:
            if isinstance(item, list):
                if len(item) == list_len:
                    out_param.append(item)
                else:
                    raise ValueError('One of the parameters was a wrong-sized list')
            else:
                new_item = [item] * list_len
                out_param.append(new_item)
    else:
        raise ValueError('parameters must be a list')
    if len(out_param) == len(parameters):
        return out_param
    else:
        raise ValueError('logic mistake in conversion, wrong number of items')


def batch_add_sparse(loom_file,
                     layers,
                     row_attrs,
                     col_attrs,
                     append=False,
                     empty_base=False,
                     batch_size=512):
    """
    Batch adds sparse matrices to a loom file

    Args:
        loom_file (str): Path to output loom file
        layers (dict): Keys are names of layers, values are matrices to include
            Matrices should be features by observations
        row_attrs (dict): Attributes for rows in loom file
        col_attrs (dict): Attributes for columns in loom file
        append (bool): If true, append new cells. If false, overwrite file
        empty_base (bool): If true, add an empty array to the base layer
        batch_size (int): Size of batches of cells to add
    """
    # Check layers
    feats = set([])
    obs = set([])
    for key in layers:
        if not sparse.issparse(layers[key]):
            raise ValueError('Expects sparse matrix input')
        feats.add(layers[key].shape[0])
        obs.add(layers[key].shape[1])
    if len(feats) != 1 or len(obs) != 1:
        raise ValueError('Matrix dimension mismatch')
    # Get size of batches
    obs_size = list(obs)[0]
    feat_size = list(feats)[0]
    batches = np.array_split(np.arange(start=0,
                                       stop=obs_size,
                                       step=1),
                             np.ceil(obs_size / batch_size))
    for batch in batches:
        batch_layer = dict()
        if empty_base:
            batch_layer[''] = np.zeros((feat_size, batch.shape[0]), dtype=int)
        for key in layers:
            batch_layer[key] = layers[key].tocsc()[:, batch].toarray()
        batch_col = dict()
        for key in col_attrs:
            batch_col[key] = col_attrs[key][batch]
        if append:
            with loompy.connect(filename=loom_file) as ds:
                ds.add_columns(layers=batch_layer,
                               row_attrs=row_attrs,
                               col_attrs=batch_col)
        else:
            loompy.create(filename=loom_file,
                          layers=batch_layer,
                          row_attrs=row_attrs,
                          col_attrs=batch_col)
            append = True


def kruskal(*args):
    """
    Adapted version of Kruskal-Wallis function from Scipy
    Original Scipy function is copyrighted by Gary Strangman and The Scipy Developers

    Args:
        sample1, sample2, ... (array-like): Two or more arrays with the sample measurements can be given as arguments.

    Returns:
        statistic (float): The Kruskal-Wallis H statistic, corrected for ties
        pvalue (float): The p-value for the test using the assumption that H has a chi square distribution
    """
    # Check inputs
    args = list(map(np.asarray, args))
    num_groups = len(args)
    if num_groups < 2:
        raise ValueError("Need at least two groups")
    for arg in args:
        if arg.size == 0:
            return [np.nan, np.nan]
    n = np.asarray(list(map(len, args)))
    # Process data
    alldata = np.concatenate(args)
    ranked = stats.rankdata(alldata)
    ties = stats.tiecorrect(ranked)
    if ties == 0:
        return [np.nan, np.nan]
    else:
        # Compute sum^2/n for each group and sum
        j = np.insert(np.cumsum(n), 0, 0)
        ssbn = 0
        for i in range(num_groups):
            a = np.ravel(ranked[j[i]:j[i + 1]])
            s = np.sum(a, 0)
            if not np.isscalar(s):
                s = s.astype(float) * s
            else:
                s = float(s) * s
            ssbn += s / n[i]
        # Get totals
        totaln = np.sum(n)
        h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
        df = num_groups - 1
        h /= ties
        chi_result = stats.distributions.chi2.sf(h, df)
        return [h, chi_result]
