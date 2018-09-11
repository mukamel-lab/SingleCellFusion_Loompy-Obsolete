"""
Collection of functions used for clustering and projecting multi-dimensional data into low-dimensional space

Written by Fangming Xie with some edits by Wayne Doyle
"""
# Import packages
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import louvain
import igraph as ig
from fitsne import FItSNE
from collections import OrderedDict
import mnni_utils

def compute_jaccard_weights(X, 
                            k, 
                            as_sparse = False):
    """
    Weights a kNN by the Jaccard index between two nodes
    
    Args: 
        X (array): unweighted kNN ajacency matrix (each row Xi* gives the kNNs of cell i) 
        k (int): number of nearest neighbors
        as_sparse (boolean): If true, return weighted kNN as a sparse matrix
        
    Returns:
        Y (array): Weighted kNN adjacency matrix
    
    Assumptions:
        Values of X are floats between 0 and 1
    """
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    ni, nj = X.shape
    assert ni == nj
    tmp = X.dot(X.T)
    Y = X.multiply(tmp/(2*k - tmp.todense()))   
    if as_sparse:
        return Y
    else:
        return Y.todense()

def gen_knn_dist_index(X,
                       k,
                       feat_ax,
                       labels = [],
                       metric = 'euclidean',
                       reduced = False,
                       n_comp = 50,
                       centered = False,
                       scaled = False,
                       svd = False,
                       seed = 23):
    """
    Generates distances and indices from a KNN for a data matrix
    
    Args:
        X (array): Array in the format of observations by features
        k (int): Number of nearest neighbors
        feat_ax (int/str): Axis where features are located
            0 or rows for rows
            1 or columns for columns
        labels (list): Labels for distances and indices
        metric (str): metric for sklearn NearestNeighbors distance
        reduced (boolean): If data has already been reduced
        n_comp (int): Number of components for data reduction
        centered (boolean): Mean centers data
        scaled (boolean): Scales data
        svd (boolean): Use SVD instead of PCA
        seed (int): Random seed
    
    Returns:
        distances (dataframe): Distance to cell specified by indices
        indices (dataframe): Indices for k nearest neighbors
    """
    feat_ax = mnni_utils.interpret_ax(feat_ax)
    # Reduce dimensions
    if reduced:
        red_arr = X
    else:
        red_arr = mnni_utils.center_scale_reduce(array = X,
                                                 feat_ax = feat_ax,
                                                 n_comp = n_comp,
                                                 centered = centered,
                                                 scaled = scaled,
                                                 svd = svd,
                                                 seed = seed)
    # Generate KNN
    nbrs = NearestNeighbors(n_neighbors = k,
                            metric='euclidean').fit(red_arr)
    distances, indices = nbrs.kneighbors(red_arr)
    if len(labels) == distances.shape[0]:
        distances = pd.DataFrame(distances, 
                                 index = labels)
        indices = pd.DataFrame(indices, 
                               index = labels)
    else:
        distances = pd.DataFrame(distances)
        indices = pd.DataFrame(indices)
    return [distances, indices]
    

def gen_knn_graph(X,
                  k):
    """
    Generates a k-nearest neighbors graph from data matrix, weighted by Jaccard index between 2 nodes

    Args:
        X (array): Array in format of n_obs by n_features
        k (int): Number of nearest neighbors

    Returns:
        gw_knn (array): Weighted kNN graph
    """
    # Get 0-1 adjacency matrix
    knn = NearestNeighbors(n_neighbors=k, 
                           metric = 'euclidean').fit(X)
    g_knn = knn.kneighbors_graph(X, 
                                 mode = 'connectivity')
    g_knn = g_knn.toarray()
    # Get weights
    gw_knn = compute_jaccard_weights(X=g_knn, 
                                     k = k)
    return gw_knn

def louvain_clustering(adj_mtx,
                       index,
                       option='DIRECTED',
                       sample_suffix=None, 
                       seed = 23):
    """
    Performs Louvain clustering given a weightd adjacency matrix

    Args:
        adj_mtx (array): Weighted adjacency matrix
        index (list): List of ids for cells/samples 
        option (str): options for ig.Graph.Weighted_Adjacency
            DIRECTED OR UNDIRECTED
        sample_suffix (str): Optional, suffix to be removed from index
        seed (int): Seed for random processes
    
    Returns:
        clusts (dataframe): Dataframe containing cluster identities (cluster_ID) for each observation
    """
    # Handle index
    index = mnni_utils.index2list(index)
    # Cluster with Louvain
    G = ig.Graph.Weighted_Adjacency(adj_mtx.tolist(), mode=option)
    louvain.set_rng_seed(seed)
    partition1 = louvain.find_partition(G,
            louvain.ModularityVertexPartition,
            weights = G.es['weight'])
    # Get labels
    labels = [0]*len(index)
    for i,cluster in enumerate(partition1):
        for element in cluster:
            labels[element] = 'cluster_' + str(i+1)
    # Make dataframe
    if sample_suffix:
        clusts = pd.DataFrame(index=[idx[:-len(sample_suffix)] for idx in index])
    else:
        clusts=pd.DataFrame(index=index)
    clusts['cluster_ID'] = labels
    clusts.rename_axis('sample',inplace=True)
    return clusts

def louvain_jaccard(mtx,
                    feat_ax,
                    index,
                    sparse = False,
                    n_comp=50,
                    k=30,
                    output_file=None,
                    sample_suffix=None,
                    reduced=False,
                    centered = False, 
                    scaled = False, 
                    svd = False, 
                    seed = 23):
    """
    Performs Louvain-Jaccard clustering from a feature matrix
    
    Args:
        mtx (sparse matrix or dataframe): observation by feature matrix
        feat_ax (int/str): Axis where features are located. 0/rows for rows, 1/columns for columns
        index (list): List of ids for observations
        sparse (boolean): True if sparse matrix, False if dense dataframe
        n_comp (int): Number of components for dimensionality reduction
        k (int): Number of nearest neighbors
        output_file (str): Path to output file
        sample_suffix (str): Optional, suffix for samples that should be removed
        reduced (boolean): If true, mtx is a dimensionality reduced matrix
        centered (boolean): If true, centers features before dimensionality reduction (removes sparseness!)
        scaled (boolean): If true, scales features before dimensionality reduction
        svd (boolean): If true use SVD for dimensionality reduction. If false use PCA
        seed (int): Seed for random processes
    
    Returns:
        clusts (dataframe): Cluster assignments for cells
        summary (dictionary): Cluster summary results
    """
    # Handle index
    index = mnni_utils.index2list(index)
    # Perform dimensionality reduction
    if reduced:
        components = mtx
    else:
        components = mnni_utils.center_scale_reduce(array = mtx,
                                                    feat_ax=feat_ax,
                                                    n_comp=n_comp,
                                                    centered=centered,
                                                    scaled=scaled,
                                                    svd=svd,
                                                    seed = 23)
    # Build a Jaccard weighted kNN
    adj_mtx = gen_knn_graph(X = components, k = k)
    # Run Louvain-Jaccard 
    clusts = louvain_clustering(adj_mtx = adj_mtx, 
                                index = index, 
                                option = 'DIRECTED', 
                                sample_suffix = sample_suffix,
                                seed=seed)
    # Get number of clusters
    nclust = np.unique(clusts['cluster_ID'].values).shape[0]
    # Summary
    summary = OrderedDict({'n_cluster': nclust, 
                           'n_cells': mtx.shape[0], 
                           'n_components':n_comp, 
                           'k':k})
    # Handle output
    if output_file:
        clusts.to_csv(output_file,sep='\t',
                      na_rep='NA',
                      header=True,
                      index=True)
        keys = list(summary.keys())
        values = [str(value) for value in list(summary.values())]
        with open(output_file+'.summary','w') as fout:
            fout.write('\t'.join(keys)+'\n')
            fout.write('\t'.join(values)+'\n')
    return clusts,summary

def run_tsne(mtx,
             feat_ax,
             index,
             sparse = False,
             perp=30,
             n_comp=50,
             n_tsne=2,
             centered=False,
             scaled=False,
             svd=False, 
             fft=False,
             reduced = False, 
             output_file=None,
             sample_suffix = None, 
             seed = 23, 
             nproc = 1, 
             n_iter = 1000, 
             **kwargs):
    """
    Generates tSNE coordinates for a given feature matrix

    Args:
        mtx (sparse matrix or dataframe): observation by feature matrix
        feat_ax (int/str): Axis where features are located
            0/rows for rows
            1/columns for columns
        index (list): List of ids for observations
        sparse (boolean): True if sparse matrix, False if dense dataframe
        perp (int): Perplexity
        n_comp (int): Number of components for dimensionality reduction
        n_tsne (int): Number of components for tSNE
        centered (boolean): If true, centers features before dimensionality reduction (removes sparseness!)
        scaled (boolean): If true, scales features before dimensionality reduction
        svd (boolean): If true, uses SVD for dimensionality reduction. If false, PCA
        fft (boolean): If true, uses FFT-accelerated tSNE
        reduced (boolean): If true, mtx is a dimensionality reduced matrix
        output_file (str): Path to output file
        sample_suffix (int): Optional, suffic for samples that should be removed
        seed (int): Seed for random processes
        nproc (int): Optional, number of processors to use if fft
        n_iter (int): Number of iterations for tSNE
        kwargs (various): Keywords for sklearn TSNE

    Returns:
        df_tsne (dataframe): tSNE coordinates for each cell
    """
    if sample_suffix:
        index=[idx[:-len(sample_suffix)] for idx in index]
    # Perform dimensionality reduction
    if reduced:
        components = mtx
    else:
        components = center_scale_reduce(array = mtx,
                                         feat_ax=feat_ax,
                                         n_comp=n_comp,
                                         centered=centered,
                                         scaled=scaled,
                                         svd = False,
                                         seed = 23)
    # Run tSNE
    if fft:
        components = components.copy(order = 'C')
        ts = FItSNE(components, 
                    no_dims = n_tsne, 
                    perplexity = perp,
                    rand_seed = seed, 
                    nthreads = nproc, 
                    max_iter = n_iter)
    else:
        tsne = TSNE(n_components = n_tsne, 
                    init = 'pca', 
                    random_state = seed,
                perplexity = perp, 
                    n_iter = n_iter, 
                    **kwargs)
        ts = tsne.fit_transform(components)
    # Make dataframe
    if n_tsne == 2:
        df_tsne = pd.DataFrame(ts,
                               columns=['tsne_x','tsne_y'])
    elif n_tsne == 3:
        df_tsne = pd.DataFrame(ts,
                               columns=['tsne_x','tsne_y','tsne_z'])
    else:
        raise ValueError('n_tsne must be 2 or 3')
    # Add sample information
    if sample_suffix:
        df_tsne['sample'] = [idx[:-len(sample_suffix)] for idx in index]
    else:
        df_tsne['sample'] = index
    df_tsne.set_index('sample',
                      inplace=True)
    # Handle output files
    if output_file:
        df_tsne.to_csv(output_file,
                       sep='\t',
                       na_rep='NA',
                       header=True,
                       index=True)
    return df_tsne



