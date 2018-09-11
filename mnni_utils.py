"""
General utitilies for performing MNNI imputation

Written by Wayne Doyle unless noted
"""

# Packages
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.preprocessing import scale
from scipy import sparse

def interpret_ax(ax):
    """
    Converts string for an axis into an integer representation
    
    Args:
        ax (int/str): 0 or rows for rows, 1 or columns for columns
    
    Returns:
        int_ax (int): 0 for rows, 1 for columns
    """
    if ax == 0 or ax == 'rows':
        int_ax = 0
    elif ax == 1 or ax == 'columns':
        int_ax = 1
    else:
        raise ValueError('{} is not supported for ax'.format(ax))
    return int_ax

def rotate_obs_by_feat(dat,
                       feat_ax):
    """
    Rotates a dataframe to be observations by features
    
    Args:
        dat (dataframe): Dataframe of interest
        feat_ax (int/str): Feature axis. 0/rows: rows, 1/columns:columns
    
    Returns:
        dat (dataframe): Observations by features dataframe
    """
    feat_ax = interpret_ax(feat_ax)
    if feat_ax == 0:
        dat = dat.T
    return dat

def transpose_ax(ax):
    """
    Changes the representation of an axis to its transposition
    
    Args:
        ax (int): Integer of 0 or 1
    
    Returns
        new_ax (int): Transposed ax
    """
    if ax == 0:
        new_ax = 1
    elif ax == 1:
        new_ax = 0
    else:
        raise ValueError('{} is not supported for ax'.format(ax))
    return new_ax

def index2list(index):
    """
    Converts an index to a list

    Args:
        index (list): List of ids for cells/samples

    Returns:
        index (list): List of ids for cells/samples
    """
    if type(index) == pd.core.indexes.base.Index:
        index = index.tolist()
    elif type(index) != list:
        raise ValueError('index should be a list')
    return index

def find_num_index(full,
                   desired):
    """
    Finds the numerical indices of desired elements in full
    
    Args:
        full (array): 1D array of elements
        desired (array): 1D array of elements that are present in full
    
    Returns:
        num_idx (list): List of numerical indices for desired elements in full
    """
    num_idx = np.ravel(np.where(np.isin(full,desired)))
    return num_idx

def restrictfeature_length(counts,
                           features,
                           lengths, 
                           nmin = None, 
                           nmax = None,
                           sparse=False):
    """
    Restricts counts to only features of a specified size

    Args:
        counts (matrix): Numpy array or sparse matrix of count data
        features (list): List of features in counts
        lengths (list): List of feature lengths
        nmin (int): If passed, mininum length of a feature
        nmax (int): If passed, maximum length of a feature
        sparse (boolean): If true, counts is a sparse matrix

    Returns:
        counts (matrix): Numpy array or sparse matrix of count data
        features (list): List of features in counts
        lengths (list): List of feature lengths

    Assumptions:
        counts is in the format of observations by features
    """
    lengths = np.asarray(lengths)
    if nmin and nmax:
        min_idx = lengths > nmin
        max_idx = lengths < nmax
        lgt_idx = min_idx & max_idx
    elif nmax:
        lgt_idx = lengths < nmax
    elif nmin:
        lgt_idx = lengths < nmax
    lgt_idx = list(np.where(lgt_idx))
    if sparse:
        counts = counts.tocsc()[:,lgt_idx]
    else:
        counts = counts[:,lgt_idx]
    lengths = list(np.asarray(lengths)[lgt_idx])
    features = list(np.asarray(features)[lgt_idx])
    return [counts,features,lengths]

def restrictfeature_ncells(counts,
                           features,
                           ncells,
                           sparse=False):
    """
    Restricts features to only those that are nonzero in n cells

    Args:
        counts (matrix): Numpy array or sparse matrix of count data
        features (list): List of features in counts
        ncells (int): Mininum number of non-zero cells per feature
        sparse (boolean): If true, counts is a sparse matrix

    Returns:
        counts (matrix): Numpy array or sparse matrix of count data
        features (list): List of features in counts

    Assumptions:
        counts is in the format of observatiosn by features
    """
    nz_vals = counts.nonzero()[1]
    nz_counts = counter(nz_vals)
    nz_idx = [x for x in nz_counts if nz_counts[x] > ncells]
    if sparse:
        counts = counts.tocsc()[:,nz_idx]
    else:
        counts = counts[:,nz_idx]
    features = list(np.asarray(features)[nz_idx])
    return [counts,
            features]

def center_scale(array,
                 feat_ax,
                 centered=True,
                 scaled=True):
    """
    Centers and scales an array
    
    Args:
        array (array): 2D array of values to be centered and scaled
        feat_ax (int): Axis where features are located. 0 for rows, 1 for columns
        centered (boolean): If true, mean centers values (removes sparseness!)
        scaled (boolean): If true, scales values by standard deviation
        is_sparse (boolean): If true, 
    """
    skax = transpose_ax(feat_ax)
    if centered or scaled:
        if sparse.issparse(array):
            array = array.todense()
        centered = scale(X = array,
                axis=skax,
                with_mean = centered,
                with_std = scaled)
    else:
        centered = array
    return centered

def center_scale_reduce(array,
                        feat_ax,
                        n_comp,
                        centered=True,
                        scaled=True,
                        svd = False,
                        seed = 23):
    """
    Centers, scales, and reduces an array
    
    Args:
        array (array): 2D array of values to be centered and scaled
        feat_ax (int): Axis where features are located. 0 for rows, 1 for columns
        n_comp (int): Number of components to reduce to
        centered (boolean): If true, mean centers values
        scaled (boolean): If true, scales values by standard deviation
        svd (boolean): If true, uses SVD instead of PCA
        seed (int): Random seed        
    """
    centered = center_scale(array,
                            feat_ax,
                            centered=centered,
                            scaled=scaled)
    if svd:
        svd = TruncatedSVD(n_components = n_comp, 
                           algorithm = 'randomized', 
                           random_state = seed)
        components = svd.fit_transform(centered)
    else:
        pca = PCA(n_components = n_comp, 
                  svd_solver = 'randomized',
                  random_state = seed)
        components = pca.fit_transform(centered)
    return components


