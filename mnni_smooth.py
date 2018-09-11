"""
Adaptation of MAGIC for working with sparse, high-dimensional epigenomic data

This code originates from https://github.com/KrishnaswamyLab/MAGIC

The publication describing MAGIC is 'MAGIC: A diffusion-based imputation method reveals gene-gene interactions in single-cell RNA-sequencing data'

The publication was authored by: David van Dijk, Juozas Nainys, Roshan Sharma, Pooja Kathail, Ambrose J Carr, Kevin R Moon, Linas Mazutis, Guy Wolf, Smita Krishnaswamy, Dana Pe'er

The DOI is https://doi.org/10.1101/111591 

Edited by Wayne Doyle
"""

# Packages 
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
import mnni_utils
import mnni_clustering

def compute_markov(data, 
                   k=10, 
                   epsilon=1, 
                   ka=0,
                   p = 0.9,
                   distance_metric='euclidean', 
                   provide_knn = False, 
                   distances = None, 
                   indices = None):
    """
    Calculates the normalized Markov matrix for imputing values

    Args:
        data (matrix): Count data (typically after dimensionality reduction)
        k (int): Number of nearest neighbors
        epsilon (int): Variance parameter
        distance_metric (string): Metric for nearest neighbors
        ka (int): kNN autotune parameter
        p (float): Proportion of smoothing from own cell
        provide_knn (boolean): If true, user provides distances/indices
        distances (array): Array of distances for KNN
        indices (array): Array of indices for KNN

    Returns:
        T (matrix): Normalized Markov matrix
    """
    N = data.shape[0]
    # Nearest neighbors
    if provide_knn:
        if indices is None or distances is None:
            raise ValueError('If provide_knn, provide the KNN')
        if indices.shape[0] != N or distances.shape[0] != N:
            raise ValueError('Dimensions of KNN do not match expectations')
    else:
        print('Computing distances')
        nbrs = NearestNeighbors(n_neighbors=k, 
                metric=distance_metric).fit(data)
        distances, indices = nbrs.kneighbors(data)
    # Remove self from distances/indices
    if distances.shape[1] == k:
        distances = distances[:,1:]
        indices = indices[:,1:]
    # Normalize by ka's distance
    print('Normalizing distance to {}'.format(ka))
    if ka > 0:
        distances = distances / (distances[:,ka].reshape(-1,1))
    # Calculate Gaussian kernel
    print('Computing kernel with epsilon of {}'.format(epsilon))
    adjs = np.exp(-((distances**2)/(epsilon**2)))
    # Construct a sparse matrix
    cols = np.ravel(indices)
    rows = np.repeat(np.arange(N), k-1) #k-1 to remove self
    vals = np.ravel(adjs)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    # Symmetrize A
    W = W + W.T
    # Normalize A
    W = sparse.csr_matrix(W/W.sum(axis=1))
    # Include cell's own self
    eye = sparse.identity(N)
    if p:
        W = p*eye + (1-p)*W
    return W

def impute_fast(data, 
                W, 
                t, 
                W_t=None, 
                tprev=None):
    """
    Imputes values into count matrix

    Args:
        data (matrix): Sparse matrix of count data
        W (matrix): Normalized Markov matrix
        t (int): Number of times W will be multipled by itself
        W_t (matrix): Normalized matrix that has alredy been raised to a power
        tprev (int): Power used on W_t parameter

    Returns:
        new_data (sparse matrix): Data with imputed values
    """
    #L^t
    print('MAGIC: W_t = W^t')
    if W_t == None:
        W_t = W.tocsr().power(t)
    else:
        W_t = W_t.tocsr().dot(W.tocsr().power(t-tprev))
    print('MAGIC: data_new = W_t * data')
    if t > 0:
        data_new = W_t.tocsr().dot(data.tocsr())
    return data_new, W_t

def smooth(data, 
           n_comp=100, 
           scaled = False, 
           provide_knn = False, 
           distances = None, 
           indices = None,
           t=None, 
           k=12, 
           ka=4, 
           epsilon=1,
           p = 0.9,
           seed=23):
    """
    Imputes smoothed data values on sparse/noisy data

    Args:
        data (matrix): Observations by features sparse matrix
        n_comp (int): Number of components for SVD
        scaled (boolean): Scales data before dimensionality reduction
        provide_knn (boolean): If true, user provides distances/indices
        distances (array): Array of distances for KNN
        indices (array): Array of indices for KNN
        t (int): Number of times for diffusion
        k (int): Number of nearest neighbors for imputation
        ka (int): kNN autotune parameter (used for normalized distances)
        p (float): Proportion of smoothing from own cell
        epsilon (int): Variance for Gaussian kernel
        seed (int): Seed for randomization
    """
    if provide_knn:
        # Get markov matrix
        W=compute_markov(data, k=k, epsilon=epsilon, 
                distance_metric='euclidean', ka=ka,
                provide_knn = True, distances = distances, 
                indices = indices)
    else:
        # Reduce dimensions
        if n_comp != None:
            print('Reducing dimensions')
            reduced = mnni_utils.center_scale_reduce(array = data,
                                                     feat_ax = 1,
                                                     n_comp = n_comp,
                                                     centered=False,                            
                                                     scaled=scaled,
                                                     svd = True,
                                                     seed = seed)
        else:
            reduced = data
        # Get markov matrix
        W = compute_markov(data = reduced, 
                           k=k, 
                           epsilon=epsilon, 
                           distance_metric='euclidean', 
                           ka=ka,
                           provide_knn = False, 
                           distances = None, 
                           indices = None)
    # Get new data
    new_data, W_t = impute_fast(data = data, 
                                W = W, 
                                t = t)
    return new_data

def smooth_raw_counts(raw,
                      normalized,
                      n_comp=100, 
                      scaled = False,
                      t=None, 
                      k=12, 
                      ka=4, 
                      epsilon=1,
                      p = 0.9,
                      seed=23,):
    """
    Performs smoothing of raw (not normalized to library depth) counts
    
    Args:
        raw (sparse matrix): Raw counts (observations by features)
        normalized (sparse matrix): Library normalized counts (observations by features)
        n_comp (int): Number of components for dimensionality reduction
        scaled (boolean): Scales data before dimensionality reduction
        t (int): Number of times for diffusion
        k (int): Number of nearest neighbors for imputation
        ka (int): kNN autotune parameter (used for normalized distances)
        epsilon (int): Variance for Gaussian kernel
        p (float): Proportion of smoothing from own cell
        seed (int): Seed for randomization
    """
    # Check inputs
    if raw.shape[0] != normalized.shape[0] or raw.shape[1] != normalized.shape[1]:
        raise ValueError('raw and normalizes must have same dimensions!')
    # Reduce dimensions of library normalized
    print('Reducing dimensions')
    reduced = mnni_utils.center_scale_reduce(array = normalized,
                                             feat_ax = 1,
                                             n_comp = n_comp,
                                             centered=False,
                                             scaled=scaled,
                                             svd = True,
                                             seed = seed)
    # Generate KNN
    distances, indices = mnni_clustering.gen_knn_dist_index(X = reduced,
                                                            k = k,
                                                            labels = None,
                                                            metric = 'euclidean',
                                                            reduced = True)
    # Smooth data
    smoothed = smooth(data = raw, 
                      provide_knn = True, 
                      distances = distances, 
                      indices = indices,
                      t=t, 
                      k=k, 
                      ka=ka, 
                      epsilon=epsilon,
                      p = p)
    return smoothed