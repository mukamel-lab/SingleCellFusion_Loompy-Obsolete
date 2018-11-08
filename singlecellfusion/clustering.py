"""
Collection of functions used to perform clustering on a loom file

Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import pandas as pd
import numpy as np
from scipy import sparse
import time
import logging
import igraph as ig
import louvain
from fitsne import FItSNE
from . import graphs
from . import loom_utils
from . import general_utils
from . import decomposition

# Start log
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)   

def louvain_clustering(loom_file,
                       graph_attr,
                       clust_attr = 'ClusterID',
                       cell_attr = 'CellID',
                       valid_attr = None,
                       directed = True,
                       batch_size = 1000,
                       seed = 23,
                       verbose = False):
    """
    Performs Louvain clustering on a given weighted adjacency matrix
    
    Args:
        loom_file (str): Path to loom file
        graph_attr (str): Name of col_graphs object in loom_file containing kNN
        col_attr (str): Name of attribute specifying columns (cells) to include
        clust_attr (str): Name of attribute specifying clusters
        cell_attr (str): Name of attribute containing cell identifiers
        valid_attr (str): Name of attribute specifying cells to use
        directed (bool): If true, graph should be directed
        seed (int): Seed for random processes
        verbose (bool): If true, print logging messages
    
    Returns:
        clusts (1D array): Cluster identities for cells in adj_mtx
    
    Adapted from code written by Fangming Xie
    """
    col_idx = loom_utils.get_attr_index(loom_file = loom_file,
            attr = valid_attr,
            columns=True,
            as_bool = False,
            inverse = False)
    with loompy.connect(loom_file) as ds:
        adj_mtx = ds.col_graphs[graph_attr]
    adj_mtx = adj_mtx.tocsr()[col_idx,:][:,col_idx]
    if adj_mtx.shape[0] != adj_mtx.shape[1]:
        raise ValueError('Adjacency matrix must be symmetrical!')
    # Generate graph
    if verbose:
        t0 = time.time()
        logger.info('Converting to igraph')
    G = graphs.adjacency_to_igraph(adj_mtx = adj_mtx, 
                                        directed=directed)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0,t1)
        logger.info('Converted to igraph in {0:.2f} {1}'.format(time_run,time_fmt))
    # Cluster with Louvain
    if verbose:
        logger.info('Performing clustering with Louvain')
    louvain.set_rng_seed(seed)
    partition1 = louvain.find_partition(G,
                                        louvain.ModularityVertexPartition,
                                        weights = G.es['weight'])
    # Get cluster IDs
    clusts = np.empty((adj_mtx.shape[0],),dtype=int)
    clusts[:] = np.nan
    for i,cluster in enumerate(partition1):
        for element in cluster:
            clusts[element] = i+1
    # Add labels to loom_file
    with loompy.connect(loom_file) as ds:
        labels = pd.DataFrame(np.repeat('Fake',ds.shape[1]),
                              index = ds.ca[cell_attr],
                              columns = ['Orig'])
        if valid_attr:
            valid_idx = ds.ca[valid_attr].astype(bool)
        else:
            valid_idx = np.ones((ds.shape[1],),dtype = bool)
        clusts = pd.DataFrame(clusts,
                              index = ds.ca[cell_attr][valid_idx],
                              columns = ['Mod'])
        labels = pd.merge(labels,
                          clusts,
                          left_index=True,
                          right_index=True,
                          how = 'left')
        labels = labels.fillna(value = 'Noise')
        labels = labels['Mod'].values.astype(str)
        ds.ca[clust_attr] = labels
    if verbose:
        t2= time.time()
        time_run, time_fmt = general_utils.format_run_time(t1,t2)
        logger.info('Clustered cells in {0:.2f} {1}'.format(time_run,time_fmt))
        
def louvain_jaccard(loom_file,
                    clust_attr = 'ClusterID',
                    cell_attr = 'CellID',
                    valid_attr = None,
                    gen_pca = False,
                    pca_attr = None,
                    layer = '',
                    n_comp = 50,
                    row_attr = None,
                    scale_attr = None,
                    gen_knn = False,
                    neighbor_attr = None,
                    distance_attr = None,
                    k = 30,
                    num_trees = 50,
                    metric = 'euclidean',
                    gen_jaccard = False,
                    jaccard_graph = None,
                    batch_size = 512,
                    seed = 23,
                    verbose = False):
    """
    Performs Louvain-Jaccard clustering on a loom file
    
    Args:
        loom_file (str): Path to loom file
        clust_attr (str): Output attribute containing clusters
            Convention is ClusterID
        cell_attr (str): Attribute specifying cell IDs
            Convention is CellID
        valid_attr (str): Attribute specifying cells to include
        row_attr (str): Attribute specifying features to include in PCA
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
        gen_jaccard (bool): If true, generate Jaccard weighted adjacency matrix
        jaccard_graph (str): Name of col_graphs containing adjacency matrix
            If gen_jaccard, this is the name of the output graph
            Default is Jaccard
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
                                         layer = layer,
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
            
        # Generate Jaccard-weighted adjacency
        if gen_jaccard:
            if jaccard_graph is None:
                jaccard_graph = 'Jaccard'
            graphs.loom_adjacency(loom_file = loom_file,
                                       neighbor_attr = neighbor_attr,
                                       graph_attr = jaccard_graph,
                                       weight = True,
                                       normalize = False,
                                       normalize_axis = None,
                                       offset = None,
                                       valid_attr = valid_attr,
                                       batch_size = batch_size)
        if clust_attr is None:
            clust_attr = 'ClusterID'
        louvain_clustering(loom_file = loom_file,
                           graph_attr = jaccard_graph,
                           clust_attr = clust_attr,
                           cell_attr = cell_attr,
                           valid_attr = valid_attr,
                           directed = True,
                           batch_size = batch_size,
                           seed = seed,
                           verbose = verbose)
