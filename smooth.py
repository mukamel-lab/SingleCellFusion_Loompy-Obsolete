"""
Adapatation of MAGIC for working with loom files and epigenomic data

This code originates from https://github.com/KrishnaswamyLab/MAGIC

The publication describing MAGIC is 'MAGIC: A diffusion-based imputation method reveals gene-gene interactions in single-cell RNA-sequencing data'

The publication was authored by: David van Dijk, Juozas Nainys, Roshan Sharma, Pooja Kathail, Ambrose J Carr, Kevin R Moon, Linas Mazutis, Guy Wolf, Smita Krishnaswamy, Dana Pe'er

The DOI is https://doi.org/10.1101/111591
"""

import pandas as pd
import numpy as np
from scipy import sparse
import time
import logging
import loompy
from . import loom_utils
from . import general_utils
from . import graphs
from . import decomposition

# Start log
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)   


def compute_markov(loom_file,
                   neighbor_attr,
                   distance_attr,
                   out_graph,
                   valid_attr = None,
                   k = 30,
                   ka = 4,
                   epsilon = 1,
                   p = 0.9,
                   batch_size = 512,
                   verbose = False):
    """
    Calculates Markov matrix for smoothing
    
    Args:
        loom_file (str): Path to loom file
        neighbor_attr (str): Name of attribute containing kNN indices
        distance_attr (str): Name of attribute containing kNN distances
        out_graph (str): Name of output graph containing Markov matrix for smoothing
        valid_attr (str): Name of attribute specifying valid cells
        k (int): Number of nearest neighbors
        ka (int): Normalize by this distance neighbor
        epsilon (int): Variance parameter
        p (float): Contribution to smoothing from a cell's own self (0-1)
    """
    if verbose:
        t0 = time.time()
        logger.info('Computing Markov matrix for smoothing')
        param_msg = 'Parameters: k = {0}, ka = {1}, epsilon = {2}, p = {3}'
        logger.info(param_msg.format(k,ka,epsilon,p))
    valid_idx = loom_utils.get_attr_index(loom_file = loom_file,
                                      attr = valid_attr,
                                      columns=True,
                                      as_bool = True,
                                      inverse = False)
    # Generate Markov in batches
    comb_adj = []
    comb_idx = []
    comb_dist = []
    with loompy.connect(loom_file) as ds:
        res_N = np.sum(valid_idx)
        tot_N = ds.shape[1]
        distances = ds.ca[distance_attr][valid_idx]
        indices = ds.ca[neighbor_attr][valid_idx]
        # Remove self
        if distances.shape[1] == k:
            distances = distances[:,1:]
            indices = indices[:,1:]
        elif distances.shape[1] != k - 1:
            raise ValueError('Size of kNN is unexpected')
        # Normalize by ka's distance
        if ka > 0:
            distances = distances / (distances[:,ka].reshape(-1,1))
        # Calculate gaussian kernel
        adjs = np.exp(-((distances**2)/(epsilon**2)))
    # Construct W
    rows = np.repeat(np.where(valid_idx),k-1) #k-1 to remove self
    cols = np.ravel(indices)
    vals = np.ravel(adjs)
    W = sparse.csr_matrix((vals,(rows,cols)), shape = (tot_N,tot_N))
    # Symmetrize W
    W = W + W.T
    # Normalize W
    divisor = np.ravel(np.repeat(W.sum(axis=1), W.getnnz(axis=1)))
    W.data /= divisor
    # Include self
    eye = sparse.identity(W.shape[0])
    if p:
        W = p*eye + (1-p)*W
    # Add to file
    with loompy.connect(loom_file) as ds:
        ds.col_graphs[out_graph] = W
    # Report if user wants
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0,t1)
        logger.info('Generated Markov matrix in {0:.2f} {1}'.format(time_run, time_fmt))
                   
def perform_smoothing(loom_file,
                      in_layer,
                      out_layer,
                      w_graph,
                      verbose = False):
    """
    Performs actual act of smoothing on cells in a loom file
    
    Args:
        loom_file (str): Path to loom file
        in_layer (str): Layer containing observed counts
        out_layer (str): Name of output layer
        w_graph (str): Name of col_graph containing markov matrix
        verbose (bool): If true, prints logging messages
    """
    if verbose:
        t0 = time.time()
        logger.info('Performing smoothing')
    with loompy.connect(loom_file) as ds:
        w = ds.col_graphs[w_graph].tocsr()
        c = ds.layers[in_layer].sparse().T.tocsr() #Transpose so smoothing on cells
        s = w.dot(c).T
        ds.layers[out_layer] = s.tocoo()
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0,t1)
        logger.info('Smoothed in {0:.2f} {1}'.format(time_run,time_fmt))

def smooth_counts(loom_file,
                  valid_attr = None,
                  gen_pca = False,
                  pca_attr = None,
                  pca_layer = '',
                  n_comp = 50,
                  row_attr = None,
                  scale_attr = None,
                  gen_knn = False,
                  neighbor_attr = None,
                  distance_attr = None,
                  k = 30,
                  num_trees = 50,
                  metric = 'euclidean',
                  gen_w = False,
                  w_graph = 'W',
                  observed_layer = 'counts',
                  smoothed_layer = 'smoothed_counts',
                  ka = 4,
                  epsilon = 1,
                  p = 0.9,
                  batch_size = 512,
                  seed = 23,
                  verbose = False):
    """
    Performs Louvain-Jaccard clustering on a loom file
    
    Args:
        loom_file (str): Path to loom file
        valid_attr (str): Attribute specifying cells to include
        gen_pca (bool): If true, perform dimensionality reduction
        pca_attr (str): Name of attribute containing PCs
            If gen_pca, this is the name of the output attribute
                Defaults to PCA
        layer (str): Layer in loom file containing data for PCA
        n_comp (int): Number of components for PCA (if pca_attr not provided)
        row_attr (str): Attribute specifying features to include
            Only used if performing PCA 
        scale_attr (str): Optional, attribute specifying cell scaling factor
            Only used if performing PCA
        gen_knn (bool): If true, generate kNN indices/distances
        neighbor_attr (str): Attribute specifying neighbor indices
            If gen_knn, this is the name of the output attribute
            Defaults to k{k}_neighbors
        distance_attr (str): Attribute specifying distances
            If gen_knn, this is the name of the output attribute
            Defaults to k{k}_distances
        k (int): Number of nearest neighbors
            Only used if generating kNN
        num_trees (int): Number of trees for approximate kNN
            Only used if generating kNN
            Increased number leads to greater precision
        metric (str): Metric for measuring distance (defaults from annoy)
            Only used if generating kNN
            angular, euclidean, manhattan, hamming, dot
        gen_w (bool): If true, generate Markov matrix for smoothing
        w_graph (str): col_graph containing Markov matrix for smoothing
            If gen_w, this is the name of the output graph
            Default is W
        observed_layer (str): Layer containing observed counts
        smoothed_layer (str): Output layer of smoothed counts
        ka (int): Normalize by this distance neighbor
        epsilon (int): Variance parameter
        p (float): Contribution to smoothing from a cell's own self (0-1)
        batch_size (int): Number of elements per chunk (for PCA)
        seed (int): Random seed for clustering
        verbose (str): If true, print logging statements
    """
    with loompy.connect(loom_file) as ds:
        # Perform PCA
        if gen_pca:
            if pca_attr is None:
                pca_attr = 'PCA'
            decomposition.batch_pca(loom_file = loom_file,
                                         layer = pca_layer,
                                         out_attr = pca_attr,
                                         col_attr = valid_attr,
                                         row_attr = row_attr,
                                         scale_attr = scale_attr,
                                         n_comp = n_comp,
                                         batch_size = batch_size,
                                         verbose = verbose)
        # Generate kNN
        if gen_knn:
            if neighbor_attr is None:
                neighbor_attr = 'k{}_neighbors'.format(k)
            if distance_attr is None:
                distance_attr = 'k{}_distances'.format(k)
            graphs.generate_knn(loom_file = loom_file,
                                     dat_attr = pca_attr,
                                     valid_attr = valid_attr,
                                     neighbor_attr = neighbor_attr,
                                     distance_attr = distance_attr,
                                     k = k,
                                     num_trees = num_trees,
                                     metric = metric,
                                     batch_size = batch_size,
                                     verbose = verbose)
            
        # Generate Markov matrix
        if gen_w:
            if w_graph is None:
                w_graph = 'W'
            compute_markov(loom_file = loom_file,
                           neighbor_attr = neighbor_attr,
                           distance_attr = distance_attr,
                           out_graph = w_graph,
                           valid_attr = valid_attr,
                           k = k,
                           ka = ka,
                           epsilon = epsilon,
                           p = p,
                           batch_size = batch_size,
                           verbose = verbose)
        # Smooth counts
        perform_smoothing(loom_file = loom_file,
                          in_layer = observed_layer,
                          out_layer = smoothed_layer,
                          w_graph = w_graph,
                          verbose = verbose)        


                      

