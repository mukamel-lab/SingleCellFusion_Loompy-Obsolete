"""
Collection of functions used to perform imputation across datasets
    
Written/developed by Fangming Xie and Wayne Doyle

"""

import loompy
import numpy as np
import pandas as pd
import time
from scipy import sparse
import functools
import re
import logging
from . import general_utils
from . import loom_utils
from . import graphs

# Start log
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)   

def get_n_variable_features(loom_file,
                            layer,
                            out_attr = None,
                            id_attr = 'Accession',
                            n_feat = 4000,
                            measure = 'vmr',
                            row_attr = None,
                            col_attr = None,
                            batch_size = 512,
                            verbose = False):
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
            cv: coeffecient of variation
        row_attr (str): Optional, attribute to restrict features by
        col_attr (str): Optional, attribute to restrict cells by
        batch_size (int): Size of chunks
            Will generate a dense array of batch_size by cells
        verbose (bool): Print logging messages
    """
    # Get valid indices
    col_idx = loom_utils.get_attr_index(loom_file = loom_file,
                                        attr = col_attr,
                                        columns = True,
                                        as_bool = True,
                                        inverse = False)
    row_idx = loom_utils.get_attr_index(loom_file = loom_file,
                                        attr = row_attr,
                                        columns = False,
                                        as_bool = True,
                                        inverse = False)
    layers = loom_utils.make_layer_list(layers = layer)
    if verbose:
        logger.info('Finding {} variable features for {}'.format(n_feat, loom_file))
        t0 = time.time()
    # Determine variability
    with loompy.connect(loom_file) as ds:
        var_df = pd.DataFrame({'var':np.zeros((ds.shape[0],),dtype=int),
                               'idx':np.zeros((ds.shape[0],),dtype=int)},
                           index = ds.ra[id_attr])
        for (_,selection,view) in ds.scan(items = row_idx,
                                          axis = 0,
                                          layers = layers,
                                          batch_size = batch_size):
            dat = view.layers[layer][:,col_idx]
            if measure.lower() == 'sd' or measure.lower() == 'std':
                var_df['var'].iloc[selection] = np.std(dat,axis=1)
            elif measure.lower() == 'vmr':
                var_df['var'].iloc[selection] = np.var(dat,axis=1)/np.mean(dat,axis=1)
            elif measure.lower() == 'cv':
                var_df['var'].iloc[selection] = np.std(dat,axis=1) / np.mean(dat,axis=1)
            else:
                raise ValueError('Unsupported measure value ({})'.format(measure))
        # Get top n variable features
        n_feat = min(n_feat,var_df.shape[0])
        hvf = var_df['var'].sort_values(ascending=False).head(n_feat).index.values
        var_df.loc[hvf, 'idx'] = 1
        if out_attr is None:
            out_attr = 'hvf_{}'.format(n_feat)
        ds.ra[out_attr] = var_df['idx'].values.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0,t1)
        logger.info('Found variable features in {0:.2f} {1}'.format(time_run,time_fmt))

def prep_for_common(loom_file,
                     id_attr = 'Accession',
                     valid_attr = None,
                     remove_version = False):
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
    valid_idx = loom_utils.get_attr_index(loom_file = loom_file,
                                          attr = valid_attr,
                                          columns = False,
                                          as_bool = True,
                                          inverse = False)
    with loompy.connect(loom_file) as ds:
        features = ds.ra[id_attr][valid_idx]
        if remove_version:
            features = general_utils.remove_gene_version(gene_ids = features)
    return features

def add_common_features(loom_file,
                        id_attr,
                        common_features,
                        out_attr,
                        remove_version = False):
    """
    Adds index of common features to loom file (run with find_common_features)
    
    Args:
        loom_file (str): Path to loom file
        id_attr (str): Name of attribute specifying unique feature IDs
        common_features (1D array): Array of common features
        out_attr (str): Name of output attribute specifying common features
        
        remove_version (bool): If true remove versioning
            Anything after the first period is dropped
            Useful for GENCODE gene IDs
    """
    # Make logical index of desired features
    feat_ids = prep_for_common(loom_file = loom_file,
                               id_attr = id_attr,
                               remove_version = remove_version,
                               valid_attr = None)
    with loompy.connect(loom_file) as ds:
        logical_idx = pd.Series(data = np.zeros((ds.shape[0],),
                                                dtype = int),
                                index = feat_ids,
                                dtype = int)
        logical_idx.loc[common_features] = 1         
        ds.ra[out_attr] = logical_idx.values

def find_common_features(loom_x,
                         loom_y,
                         out_attr,
                         id_x = 'Accession',
                         id_y = 'Accession',
                         valid_x = None,
                         valid_y = None,
                         remove_version = False,
                         verbose = False):
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
        logger.info('Finding common features')
    # Get features
    feat_x = prep_for_common(loom_file = loom_x,
                                id_attr = id_x,
                                valid_attr = valid_x,
                                remove_version = remove_version)
    feat_y = prep_for_common(loom_file = loom_y,
                                id_attr = id_y,
                                valid_attr = valid_y,
                                remove_version = remove_version)
    # Find common features
    feats = [feat_x,feat_y]
    common_feat = functools.reduce(np.intersect1d,feats)
    if common_feat.shape[0] == 0:
        raise ValueError('Could not identify any common features')
    # Add indices
    add_common_features(loom_file = loom_x,
                        id_attr = id_x,
                        common_features = common_feat,
                        out_attr = out_attr,
                        remove_version = True)
    add_common_features(loom_file = loom_y,
                        id_attr = id_y,
                        common_features = common_feat,
                        out_attr = out_attr,
                        remove_version = True)
    if verbose:
        with loompy.connect(loom_x) as ds:
            common_x = np.sum(ds.ra[out_attr])
        logger.info('Found {} features in common'.format(common_x))
        
def view_corrcoef(ds_x,
                  view_x,
                  layer_x,
                  corr_x,
                  col_x,
                  row_x,
                  sel_x,
                  ds_y,
                  view_y,
                  layer_y,
                  corr_y,
                  col_y,
                  row_y,
                  sel_y,
                  direction):
    """
    Calculates correlation coefficients between two loompy view objects
    
    Args:
        ds_x (loompy object): Handle of loompy file
        view_x (loompy object): View from a loompy file
        layer_x (str): Layer of desired counts for correlation in x
        corr_x (str): Name of output correlation attribute
        col_x (array): Boolean array of columns to include in x
        row_x (array): Boolean array of rows to include in x
        sel_x (array): Columns to include from col_x
        ds_y (loompy object): Handle of loompy file
        view_y (loompy object): View from a loompy file
        layer_y (str): Layer of desired counts for correlation in y
        corr_y (str): Name of output correlation attribute
        col_y (array): Boolean array of columns to include in y
        row_y (array): Boolean array of rows to include in y
        sel_y (array): Columns to include from col_y
        direction (str): Direction of expected correlation
            negative/- or positive/+
    
    Uses code written by dbliss on StackOverflow
        https://stackoverflow.com/questions/30143417/
    
    Update with batch add
    """
    # Get number of cells
    num_x = ds_x.shape[1]
    num_y = ds_y.shape[1]
    # Handle columns
    col_x = col_x[sel_x]
    col_y = col_y[sel_y]
    # Get data
    dat_x = view_x.layers[layer_x][:,col_x][row_x,:].T
    dat_y = view_y.layers[layer_y][:,col_y][row_y,:].T
    # Get ranks
    dat_x = pd.DataFrame(dat_x).rank(pct = True, axis = 1).values
    dat_y = pd.DataFrame(dat_y).rank(pct = True, axis = 1).values
    if direction == '+' or direction == 'positive':
        pass
    elif direction == '-' or direction == 'negative':
        dat_x = 1 - dat_x
    else:
        raise ValueError('Unsupported direction value ({})'.format(direction))
    # Get number of features
    if dat_x.shape[1] == dat_y.shape[1]:
        n = dat_x.shape[1]
    else:
        raise ValueError('dimension mismatch')
    # Calculate coefficients
    mean_x = dat_x.mean(axis = 1)
    mean_y = dat_y.mean(axis = 1)
    std_x = dat_x.std(axis = 1,
                  ddof = n-1)
    std_y = dat_y.std(axis = 1,
                  ddof = n-1)
    cov = np.dot(dat_x,dat_y.T) - n * np.dot(mean_x[:, np.newaxis],
                                             mean_y[np.newaxis, :])
    coeff =  sparse.csr_matrix(cov / np.dot(std_x[:, np.newaxis], 
                                            std_y[np.newaxis, :]))
    dat_x = None
    dat_y = None
    cov = None
    coeff = general_utils.expand_sparse(mtx = coeff,
                                        col_index = sel_y,
                                        row_index = sel_x,
                                        col_N = num_y,
                                        row_N = num_x)
    ds_x.ca[corr_x] += coeff.todense()
    ds_y.ca[corr_y] += coeff.T.todense()
    coeff = None

def generate_correlations(loom_x,
                          layer_x,
                          corr_x,
                          loom_y,
                          layer_y,
                          corr_y,
                          direction,
                          ca_x = None,
                          ra_x = None,
                          ca_y = None,
                          ra_y = None,
                          batch_x = 512,
                          batch_y = 512,
                          verbose = False):
    """
    Adds correlation matrices between two modalites to loom files
    
    Args:
        loom_x (str): Path to loom file
        layer_x (str): Name of layer containing counts
        corr_x (str): Name of output correlation attribute
        loom_y (str): Path to loom file
        layer_y (str): Name of layer containing counts
        corr_y (str): Name of output correlation attribute
        direction (str): Direction of expected correlation
            negative/- or positive/+
        ca_x (str): Name of column attribute to restrict counts by
        ra_x (str): Name of row attribute to restrict counts by
        ca_y (str): Name of column attribute to restrict counts by
        ra_y (str): Name of row attribute to restrict counts by
        batch_x (int): Chunk size for batches
        batch_y (int): Chunk size for batches
        verbose (bool): Print logging messages
    
    Notes
        Not very efficient, could generate dat_x and pass into subfunction
        To reduce memory could perform garbage collection after subfunction
    """
    if verbose:
        logger.info('Generating correlation matrix')
        t0 = time.time()
    # Get relevant attributes
    layers_x = loom_utils.make_layer_list(layer_x)
    col_x = loom_utils.get_attr_index(loom_file = loom_x,
                                      attr = ca_x,
                                      columns=True,
                                      as_bool = True,
                                      inverse = False)
    row_x = loom_utils.get_attr_index(loom_file = loom_x,
                                      attr = ra_x,
                                      columns=False,
                                      as_bool = True,
                                      inverse = False)
    layers_y = loom_utils.make_layer_list(layer_y)
    col_y = loom_utils.get_attr_index(loom_file = loom_y,
                                      attr = ca_y,
                                      columns=True,
                                      as_bool = True,
                                      inverse = False)
    row_y = loom_utils.get_attr_index(loom_file = loom_y,
                                      attr = ra_y,
                                      columns=False,
                                      as_bool = True,
                                      inverse = False)
    # Prepare for correlation matrix
    with loompy.connect(loom_x) as ds_x:
        with loompy.connect(loom_y) as ds_y:
            num_x = ds_x.shape[1]
            num_y = ds_y.shape[1]
            ds_x.ca[corr_x] = np.zeros((num_x,num_y),dtype = float)
            ds_y.ca[corr_y] = np.zeros((num_y,num_x),dtype = float)
    # Generate correlation matrix
    with loompy.connect(loom_x) as ds_x:   
        for (_,sel_x,view_x) in ds_x.scan(axis = 1,
                                          items = col_x,
                                          layers = layers_x,
                                          batch_size = batch_x):
            with loompy.connect(loom_y) as ds_y:
                for (_,sel_y,view_y) in ds_y.scan(axis=1,
                                                  items = col_y,
                                                  layers = layers_y,
                                                  batch_size = batch_y):
                    view_corrcoef(ds_x = ds_x,
                                  view_x = view_x,
                                  layer_x = layer_x,
                                  corr_x = corr_x,
                                  col_x = col_x,
                                  row_x = row_x,
                                  sel_x = sel_x,
                                  ds_y = ds_y,
                                  view_y = view_y,
                                  layer_y = layer_y,
                                  corr_y = corr_y,
                                  col_y = col_y,
                                  row_y = row_y,
                                  sel_y = sel_y,
                                  direction = direction)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0,t1)
        logger.info('Generated correlations in {0:.2f} {1}'.format(time_run,time_fmt))

def gen_knn_from_corr(loom_file,
                      corr,
                      neighbor_attr,
                      distance_attr,
                      k,
                      self_idx,
                      other_idx, 
                      batch_size = 512):
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
    with loompy.connect(loom_file) as ds:
        neighbors = np.zeros((ds.shape[1],k),dtype = int)
        distances = np.zeros((ds.shape[1],k),dtype = float)
        for(_,selection,view) in ds.scan(axis = 1,
                                         layers = [''],
                                         items = self_idx,
                                         batch_size = batch_size):
            tmp = pd.DataFrame(view.ca[corr][:,other_idx])
            knn = ((-tmp).rank(axis=1) <= k).values.astype(bool)
            if np.unique(np.sum(knn,axis = 1)).shape[0] != 1:
                raise ValueError('k is inappropriate for data')
            tmp_neighbor = np.reshape(np.where(knn)[1], 
                                      (selection.shape[0],-1))
            tmp_distance = np.reshape(tmp.values[knn], 
                                      (selection.shape[0],-1))
            neighbors[selection,:] = tmp_neighbor
            distances[selection,:] = tmp_distance
        ds.ca[neighbor_attr] = neighbors
        ds.ca[distance_attr] = distances
        
def multimodal_adjacency(distances,
                         neighbors,
                         num_col,
                         new_k = None):
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
    knn = ((-tmp).rank(axis=1,method='first') <= new_k).values.astype(bool)
    if np.unique(np.sum(knn,axis=1)).shape[0] != 1:
        raise ValueError('k is inappropriate for data')
    A = sparse.csr_matrix((np.ones((neighbors.shape[0]*new_k,),dtype=int),
                           (np.where(knn)[0],neighbors[knn])),
                          (neighbors.shape[0],num_col))
    return A

def gen_impute_adj(loom_file,
                   neighbor_attr,
                   distance_attr,
                   k_1,
                   k_2,
                   self_idx,
                   other_idx,
                   batch_size):
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
    
    Returns
        A_1 (sparse matrix): Adjacency matrix for k_1
        A_2 (sparse_matrix): Adjacency matrix for k_2
    """
    A_1 = []
    A_2 = []
    num_other = np.sum(other_idx)
    num_self = np.sum(self_idx)
    with loompy.connect(loom_file) as ds:
        if self_idx.shape[0] != ds.shape[1]:
            raise ValueError('Index does not match dimensions')
        for (_,selection,view) in ds.scan(axis = 1,
                                          layers = [''],
                                          items = self_idx,
                                          batch_size = batch_size):
            A_1.append(multimodal_adjacency(distances = view.ca[distance_attr],
                                            neighbors = view.ca[neighbor_attr],
                                            num_col = num_other,
                                            new_k = k_1))
            A_2.append(multimodal_adjacency(distances = view.ca[distance_attr],
                                           neighbors = view.ca[neighbor_attr],
                                           num_col = num_other,
                                           new_k = k_2))
    # Make matrices
    A_1 = sparse.vstack(A_1)
    A_2 = sparse.vstack(A_2)
    A_1 = general_utils.expand_sparse(mtx = A_1,
                                      col_index = np.where(other_idx)[0],
                                      row_index = np.where(self_idx)[0],
                                      col_N = other_idx.shape[0],
                                      row_N = self_idx.shape[0])
    A_2 = general_utils.expand_sparse(mtx = A_2,
                                      col_index = np.where(other_idx)[0],
                                      row_index = np.where(self_idx)[0],
                                      col_N = other_idx.shape[0],
                                      row_N = self_idx.shape[0])
    return A_1, A_2

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
    Ax_xy, Ay_xy = gen_impute_adj(loom_file = loom_x,
                                  neighbor_attr = neighbor_x,
                                  distance_attr = distance_x,
                                  k_1 = kx_xy,
                                  k_2 = ky_xy,
                                  self_idx = col_x,
                                  other_idx = col_y,
                                  batch_size = batch_x)
    Ax_yx, Ay_yx = gen_impute_adj(loom_file = loom_y,
                                  neighbor_attr = neighbor_y,
                                  distance_attr = distance_y,
                                  k_1 = kx_yx,
                                  k_2 = ky_yx,
                                  self_idx = col_y,
                                  other_idx = col_x,
                                  batch_size = batch_y)
    # Generate mutual neighbors adjacency
    A_xy = (Ax_xy.multiply(Ax_yx.T))
    A_yx = (Ay_yx.multiply(Ay_xy.T))   
    return A_xy, A_yx

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
                   verbose = False):
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
    
    To Do:
        Have non-normalized adjacency be added to loom to reduce memory
    """
    if verbose:
        logger.info('Generating mutual adjacency matrix')
        t0 = time.time()
    with loompy.connect(loom_x) as ds:
        num_x = ds.shape[1]
    with loompy.connect(loom_y) as ds:
        num_y = ds.shape[1]
    # Get x and y
    k_xy = np.arange(step_x_to_y,
                     max_x_to_y,
                     step_x_to_y)
    k_yx = np.arange(step_y_to_x,
                     max_y_to_x,
                     step_y_to_x)
    # Loop over k values
    for idx, (kx_xy, kx_yx, ky_xy, ky_yx) in enumerate(zip(k_xy,
                                                           mutual_scale_x_to_y*k_xy,
                                                           mutual_scale_y_to_x*k_yx,
                                                           k_yx)):

        # Make adjacency matrix
        A_xy, A_yx = gen_k_adj(loom_x = loom_x,
                               neighbor_x = neighbor_x,
                               distance_x = distance_x,
                               kx_xy = kx_xy,
                               kx_yx = kx_yx,
                               col_x = col_x,
                               batch_x = batch_x,
                               loom_y = loom_y,
                               neighbor_y = neighbor_y,
                               distance_y = distance_y,
                               ky_yx = ky_yx,
                               ky_xy = ky_xy,
                               col_y = col_y,
                               batch_y = batch_y)
        # Get cells
        c_x = np.sort(np.unique(A_xy.nonzero()[0]))
        c_y = np.sort(np.unique(A_yx.nonzero()[0]))
        # Update mutual adjacency
        if idx == 0:
            gA_xy = A_xy.copy().tolil()
            gc_x = c_x.copy()
            gA_yx = A_yx.copy().tolil()
            gc_y = c_y.copy()
        else:
            for j in c_x:
                if j not in gc_x:
                    gA_xy[j,:] = A_xy[j,:]
                    gc_x = np.append(gc_x,j)
            for j in c_y:
                if j not in gc_y:
                    gA_yx[j,:] = A_yx[j,:]
                    gc_y = np.append(gc_y,j)
        if verbose:
            basic_msg = '{0}: {1} cells with {2} k to other modality and {3} k back'
            logger.info(basic_msg.format(loom_x, len(gc_x),kx_xy, kx_yx))
            logger.info(basic_msg.format(loom_y, len(gc_y),ky_yx, ky_xy))
    return gA_xy, gA_yx, gc_x, gc_y

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
                      offset = 1e-5,
                      verbose = False):
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
        logger.info('Generating mutual Markov')
        t0 = time.time()
    # Get adjacency matrices
    gA_xy, gA_yx, gc_x, gc_y = gen_mutual_adj(loom_x = loom_x,
                                  neighbor_x = neighbor_x,
                                  distance_x = distance_x,
                                  max_x_to_y = max_x_to_y,
                                  step_x_to_y = step_x_to_y,
                                  mutual_scale_x_to_y = mutual_scale_x_to_y,
                                  col_x = col_x,
                                  batch_x = batch_x,
                                  loom_y = loom_y,
                                  neighbor_y = neighbor_y,
                                  distance_y = distance_y,
                                  max_y_to_x = max_y_to_x,
                                  step_y_to_x = step_y_to_x,
                                  mutual_scale_y_to_x = mutual_scale_y_to_x,
                                  col_y = col_y,
                                  batch_y = batch_y,
                                  verbose = verbose)
    # Normalize adjacency matrices
    gA_xy = graphs.normalize_adj(adj_mtx = gA_xy,
                                      axis = 1,
                                      offset = offset)
    gA_yx = graphs.normalize_adj(adj_mtx = gA_yx,
                                      axis = 1,
                                      offset = offset)
    with loompy.connect(loom_x) as ds:
        ds.ca[mutual_x] = gA_xy.toarray()
        used_idx = np.zeros((ds.shape[1],),dtype = int)
        used_idx[gc_x] = 1
        ds.ca[used_x] = used_idx
    with loompy.connect(loom_y) as ds:
        ds.ca[mutual_y] = gA_yx.toarray()
        used_idx = np.zeros((ds.shape[1],),dtype = int)
        used_idx[gc_y] = 1
        ds.ca[used_y] = used_idx
    if verbose:
        basic_msg = 'From {0} obtained {1} cells'
        logger.info(basic_msg.format(loom_x,len(gc_x)))
        logger.info(basic_msg.format(loom_y,len(gc_y)))
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0,t1)
        logger.info('Generated Markov in {0:.2f} {1}'.format(time_run,time_fmt))
        
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
                offset = 1e-5,
                remove_version = False,
                batch_size = 512,
                verbose = False):
    """
    Performs imputation over a list (if provided) of layers
    
    Args:
        loom_source (str): Name of loom file that contains observed count data
        layer_source (str/list): Layer(s) containing observed count data
        id_source (str): Row attribute specifying unique feature IDs
        cell_source (str): Column attribute specifying columns to include
        feat_source (str): Row attribute specifying rows to include
        loom_target (str): Name of loom file that will receive imputed count data
        layer_target (str/list): Layer(s) that will contain imputed count data
        id_target (str): Row attribute specifying unique feature IDs
        cell_target (str): Column attribute specifying columns to include
        feat_target (str): Row atttribute specifying rows to include
        markov_mnn (str): Column attribute specifying MNN Markov in target
        markov_self (str): Optional, col_graph attribute specifying target's Markov
        offset (float): Offset for normalizing adjacency matrices
        remove_verison (bool): Remove GENCODE version numbers from IDs
        batch_size (int): Chunk size
        verbose (bool): Print logging messages
    
    To Do:
        Possibly allow additional restriction of features during imputation
        Batch impute to reduce memory
    """
    if verbose:
        logger.info('Generating imputed {}'.format(layer_target))
        t0 = time.time()
    # Get indices feature information
    cidx_tar = loom_utils.get_attr_index(loom_file = loom_target,
                                        attr = cell_target,
                                        columns=True,
                                        as_bool = True,
                                        inverse = False)
    fidx_tar = loom_utils.get_attr_index(loom_file = loom_target,
                                        attr = feat_target,
                                        columns=False,
                                        as_bool = True,
                                        inverse = False)
    out_idx = np.where(cidx_tar)[0]
    cidx_src = loom_utils.get_attr_index(loom_file = loom_source,
                                        attr = cell_source,
                                        columns = True,
                                        as_bool = True,
                                        inverse = False)
    fidx_src = loom_utils.get_attr_index(loom_file = loom_source,
                                        attr = feat_source,
                                        columns = False,
                                        as_bool = True,
                                        inverse = False)
    # Get relevant data from files
    with loompy.connect(loom_target) as ds:
        num_feat = ds.shape[0]
        feat_tar = ds.ra[id_target]
        W_impute = sparse.csr_matrix(ds.ca[markov_mnn][cidx_tar,:][:,cidx_src])
        if markov_self is not None:
            W_self = ds.col_graphs[markov_self].tolil()[cidx_tar,:][:,cidx_tar]
    with loompy.connect(filename = loom_source, mode = 'r') as ds:
        feat_src = ds.ra[id_source]
    # Determine features to include
    if remove_version:
        feat_tar = general_utils.remove_gene_version(gene_ids = feat_tar)
        feat_src = general_utils.remove_gene_version(gene_ids = feat_src)
    feat_tar = pd.DataFrame(np.arange(0,feat_tar.shape[0]),
                            index = feat_tar,
                            columns = ['tar'])
    feat_src = pd.DataFrame(np.arange(0,feat_src.shape[0]),
                            index = feat_src,
                            columns = ['src'])
    feat_tar = feat_tar.iloc[fidx_tar]
    feat_src = feat_src.iloc[fidx_src]
    feat_df = pd.merge(feat_tar,
                       feat_src,
                       left_index = True,
                       right_index = True,
                       how = 'inner')
    # Update self Markov
    if markov_self is not None:
        if W_impute.shape[0] != W_self.shape[0]:
            raise ValueError('Dimensions of Markov must match!')
        gci = np.unique(W_impute.nonzero()[0])
        for i in range(W_self.shape[0]):
            if i in gci:
                W_self[i,:] = 0
                W_self[i,i] = 1
            else:
                W_self[:,i] = 0
        W_self = graphs.normalize_adj(adj_mtx = W_self,
                                        axis = 1,
                                        offset = offset)
        W_use = W_self.dot(W_impute)
    else:
        W_use = W_impute
    with loompy.connect(loom_target) as ds_tar:
        # Make empty data
        ds_tar.layers[layer_target] = sparse.coo_matrix(ds_tar.shape,
                                                        dtype = float)
        # Get index for batches
        valid_idx = np.unique(W_use.nonzero()[0])
        batches = np.array_split(valid_idx,np.ceil(valid_idx.shape[0]/batch_size))
        for batch in batches:
            tmp_use = W_use[batch,:]
            use_idx = np.unique(tmp_use.nonzero()[1])
            with loompy.connect(filename = loom_source, mode = 'r') as ds_src:
                    tmp_dat = ds_src.layers[layer_source][:,use_idx][feat_df['src'].values,:]
                    tmp_dat = sparse.csr_matrix(tmp_dat).T
            imputed = tmp_use[:,use_idx].dot(tmp_dat)
            imputed = general_utils.expand_sparse(mtx = imputed,
                                                  col_index = feat_df['tar'].values,
                                                  col_N = num_feat)
            imputed = imputed.transpose()
            loc_idx = out_idx[batch]
            ds_tar.layers[layer_target][:,loc_idx] = imputed.toarray()
        valid_feat = np.zeros((ds_tar.shape[0],),dtype = int)
        valid_feat[feat_df['tar'].values] = 1
        ds_tar.ra['Valid_{}'.format(layer_target)] = valid_feat
        valid_cells = np.zeros((ds_tar.shape[1],),dtype=int)
        valid_cells[out_idx[valid_idx]] = 1
        ds_tar.ca['Valid_{}'.format(layer_target)] = valid_cells
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0,t1)
        logger.info('Imputed data in {0:.2f} {1}'.format(time_run,time_fmt))      

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
                     offset = 1e-5,
                     remove_version = False,
                     batch_size = 512,
                     verbose = False):
    """
    Performs imputation over a list (if provided) of layers
    
    Args:
        loom_source (str): Name of loom file that contains observed count data
        layer_source (str/list): Layer(s) containing observed count data
        id_source (str): Row attribute specifying unique feature IDs
        cell_source (str): Column attribute specifying columns to include
        feat_source (str): Row attribute specifying rows to include
        loom_target (str): Name of loom file that will receive imputed count data
        layer_target (str/list): Layer(s) that will contain imputed count data
        id_target (str): Row attribute specifying unique feature IDs
        cell_target (str): Column attribute specifying columns to include
        feat_target (str): Row attribute specifying rows to include
        markov_mnn (str): Column attribute specifying MNN Markov in target
        markov_self (str): col_graph attribute specifying target's Markov
        offset (float): Offset for normalizing adjacency matrices
        remove_verison (bool): Remove GENCODE version numbers from IDs
        batch_size (int): Size of chunks
        verbose (bool): Print logging messages
    """
    if isinstance(layer_source,list) and isinstance(layer_target,list):
        if len(layer_source) != len(layer_target):
            raise ValueError('layer_source and layer_target should have same length')
        for i in range(0,len(layer_source)):
            impute_data(loom_source = loom_source,
                        layer_source = layer_source[i],
                        id_source = id_source,
                        cell_source = cell_source,
                        feat_source = feat_source,
                        loom_target = loom_target,
                        layer_target = layer_target[i],
                        id_target = id_target,
                        cell_target = cell_target,
                        feat_target = feat_target,
                        markov_mnn = markov_mnn,
                        markov_self = markov_self,
                        offset = offset,
                        remove_version = remove_version,
                        batch_size = batch_size,
                        verbose = verbose)
    elif isinstance(layer_source,str) and isinstance(layer_target,str):
        impute_data(loom_source = loom_source,
                    layer_source = layer_source,
                    id_source = id_source,
                    cell_source = cell_source,
                    feat_source = feat_source,
                    loom_target = loom_target,
                    layer_target = layer_target,
                    id_target = id_target,
                    cell_target = cell_target,
                    feat_target = feat_target,
                    markov_mnn = markov_mnn,
                    markov_self = markov_self,
                    offset = offset,
                    remove_version = remove_version,
                    batch_size = batch_size,
                    verbose = verbose)
    else:
        raise ValueError('layer_source and layer_target should be consistent shapes')
        
def prep_for_imputation(loom_x,
                        corr_x,
                        neighbor_x,
                        distance_x,
                        mutual_x,
                        used_x,
                        loom_y,
                        corr_y,
                        neighbor_y,
                        distance_y,
                        mutual_y,
                        used_y,
                        ca_x = None,
                        ra_x = None,
                        id_x = 'Accession',
                        max_x_to_y = 100,
                        step_x_to_y = 10,
                        mutual_scale_x_to_y = 2,
                        batch_x = 512,
                        ca_y = None,
                        ra_y = None,
                        id_y = 'Accession',
                        max_y_to_x = 100,
                        step_y_to_x = 10,
                        mutual_scale_y_to_x = 2,
                        batch_y = 512,
                        offset = 1e-5,
                        remove_version = False,
                        verbose = False):
    """
    Generates mutual kNN and Markov for imputation
    
    Args:
        loom_x (str): Path to loom file containing one dataset
        corr_x (str): Name of column attribute containing correlations
        neighbor_x (str): Name of kNN neighbors
            k = max_x_to_y * scale_x_to_y
        distance_x (str): Name of kNN distances
        self_x (str): Name of col_graph specifying single dataset Markov
        mutual_x (str): Name of column attribute containing mutual Markov
        used_x (str): Name of column attribute containing cells that made MNNs
        layer_x (str/list): Layer(s) containing counts used for imputation
        out_x (str/list): Output layer(s) for imputed data
        loom_y (str): Path to loom file containing one dataset
        corr_y (str): Name of column attribute containing correlations
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
    # Start log
    if verbose:
        logger.info('Preparing to impute between {0} and {1}'.format(loom_x,
                                                                     loom_y))
        t0 = time.time() 
    # Get columns
    col_x = loom_utils.get_attr_index(loom_file = loom_x,
                                      attr = ca_x,
                                      columns=True,
                                      as_bool = True,
                                      inverse = False)
    col_y = loom_utils.get_attr_index(loom_file = loom_y,
                                      attr = ca_y,
                                      columns=True,
                                      as_bool = True,
                                      inverse = False)
    # Generate kNN for max value
    gen_knn_from_corr(loom_file = loom_x,
                      corr = corr_x,
                      neighbor_attr = neighbor_x,
                      distance_attr = distance_x,
                      k = max_x_to_y*mutual_scale_x_to_y,
                      self_idx = col_x,
                      other_idx = col_y,
                      batch_size = batch_x)
    gen_knn_from_corr(loom_file = loom_y,
                      corr = corr_y,
                      neighbor_attr = neighbor_y,
                      distance_attr = distance_y,
                      k = max_y_to_x*mutual_scale_y_to_x,
                      self_idx = col_y,
                      other_idx = col_x,
                      batch_size = batch_y)
    # Generate mutual nearest neighbors Markov matrix
    gen_mutual_markov(loom_x = loom_x,
                      neighbor_x = neighbor_x,
                      distance_x = distance_x,
                      mutual_x = mutual_x,
                      used_x = used_x,
                      max_x_to_y = max_x_to_y,
                      step_x_to_y = step_x_to_y,
                      mutual_scale_x_to_y = mutual_scale_x_to_y,
                      col_x = col_x,
                      batch_x = batch_x,
                      loom_y = loom_y,
                      neighbor_y = neighbor_y,
                      distance_y = distance_y,
                      mutual_y = mutual_y,
                      used_y = used_y,
                      max_y_to_x = max_y_to_x,
                      step_y_to_x = step_y_to_x,
                      mutual_scale_y_to_x = mutual_scale_y_to_x,
                      col_y = col_y,
                      batch_y = batch_y,
                      offset = offset,
                      verbose = verbose)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0,t1)
        logger.info('Prepared for imputation in {0:.2f} {1}'.format(time_run,time_fmt))
        
def impute_between_datasets(loom_x,
                            corr_x,
                            neighbor_x,
                            distance_x,
                            self_x,
                            mutual_x,
                            used_x,
                            layer_x,
                            out_x,
                            loom_y,
                            corr_y,
                            neighbor_y,
                            distance_y,
                            self_y,
                            mutual_y,
                            used_y,
                            layer_y,
                            out_y,
                            ca_x = None,
                            ra_x = None,
                            id_x = 'Accession',
                            max_x_to_y = 100,
                            step_x_to_y = 10,
                            mutual_scale_x_to_y = 2,
                            batch_x = 512,
                            ca_y = None,
                            ra_y = None,
                            id_y = 'Accession',
                            max_y_to_x = 100,
                            step_y_to_x = 10,
                            mutual_scale_y_to_x = 2,
                            batch_y = 512,
                            offset = 1e-5,
                            remove_version = False,
                            verbose = False):
    """
    Imputes data between datasets
    
    Args:
        loom_x (str): Path to loom file containing one dataset
        corr_x (str): Name of column attribute containing correlations
        neighbor_x (str): Name of kNN neighbors
            k = max_x_to_y * scale_x_to_y
        distance_x (str): Name of kNN distances
        self_x (str): Name of col_graph specifying single dataset Markov
        mutual_x (str): Name of column attribute containing mutual Markov
        used_x (str): Name of column attribute containing cells that made MNNs
        layer_x (str/list): Layer(s) containing counts used for imputation
        out_x (str/list): Output layer(s) for imputed data
        loom_y (str): Path to loom file containing one dataset
        corr_y (str): Name of column attribute containing correlations
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
    # Prepare for imputation
    prep_for_imputation(loom_x = loom_x,
                        corr_x = corr_x,
                        neighbor_x = neighbor_x,
                        distance_x = distance_x,
                        self_x = self_x,
                        mutual_x = mutual_x,
                        used_x = used_x,
                        layer_x = layer_x,
                        out_x = out_x,
                        loom_y = loom_y,
                        corr_y = corr_y,
                        neighbor_y = neighbor_y,
                        distance_y = distance_y,
                        self_y = self_y,
                        mutual_y = mutual_y,
                        used_y = used_y,
                        layer_y = layer_y,
                        out_y = out_y,
                        ca_x = ca_x,
                        ra_x = ra_x,
                        id_x = id_x,
                        max_x_to_y = max_x_to_y,
                        step_x_to_y = step_x_to_y,
                        mutual_scale_x_to_y = mutual_scale_x_to_y,
                        batch_x = batch_x,
                        ca_y = ca_y,
                        ra_y = ra_y,
                        id_y = id_y,
                        max_y_to_x = max_y_to_x,
                        step_y_to_x = step_y_to_x,
                        mutual_scale_y_to_x = mutual_scale_y_to_x,
                        batch_y = batch_y,
                        offset = offset,
                        remove_version = remove_version,
                        verbose = verbose)
    # Impute data for loom_x
    loop_impute_data(loom_source = loom_y,
                     layer_source = layer_y,
                     id_source = id_y,
                     cell_source = ca_y,
                     feat_source = ra_y,
                     loom_target = loom_x,
                     layer_target = out_x,
                     id_target = id_x,
                     cell_target = ca_x,
                     feat_target = ra_x,
                     markov_mnn = mutual_x,
                     markov_self = self_x,
                     offset = offset,
                     remove_version = remove_version,
                     batch_size = batch_x,
                     verbose = verbose)
    # Impute data for loom_y
    loop_impute_data(loom_source = loom_x,
                     layer_source = layer_x,
                     id_source = id_x,
                     cell_source = ca_x,
                     feat_source = ra_y,
                     loom_target = loom_y,
                     layer_target = out_y,
                     id_target = id_y,
                     cell_target = ca_y,
                     feat_target = ra_y,
                     markov_mnn = mutual_y,
                     markov_self = self_y,
                     offset = offset,
                     remove_version = remove_version,
                     batch_size = batch_y,
                     verbose = verbose)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0,t1)
        logger.info('Completed imputation in {0:.2f} {1}'.format(time_run,time_fmt))