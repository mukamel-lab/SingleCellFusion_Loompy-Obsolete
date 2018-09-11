"""
Functions used to impute data across modalities

Written by Wayne Doyle using code/ideas from Fangming Xie
"""

# Packages
import pandas as pd
import numpy as np
import functools
from scipy import sparse
import mnni_utils
from sklearn.neighbors import NearestNeighbors
from mnni_clustering import compute_jaccard_weights
import collections

def load_sparse(matrix_fn,
                feat_fn,
                obs_fn,
                feat_ax,
                tuple_name='combined'):
    """
    Loads a sparse matrix in a format conducive to imputation across modalities
    
    Args:
        matrix_fn (str): Path to sparse matrix file
        feat_fn (str): Path to features file
        obs_fn (str): Path to observations file
        feat_ax (int/str): Features axis in matrix. 0/rows for rows, 1/columns for columns
        out_name (str): Name of named tuple output
    
    Returns:
        comb_dat (named tuple): Named tuple containing:
            features (array): Feature identities
            observations (array): Observation identities
            data (sparse matrix): Matrix data
    """
    dat = sparse.load_npz(matrix_fn)
    features = np.loadtxt(feat_fn,
                          dtype=str)
    observations = np.loadtxt(obs_fn,
                              dtype=str)
    dat = mnni_utils.rotate_obs_by_feat(dat = dat,
                                        feat_ax = feat_ax)
    if dat.shape != (observations.shape[0], features.shape[0]):
        raise ValueError('Size of observations/features does not match size of data')
    comb_dat = collections.namedtuple(tuple_name,
                                      ['features',
                                       'observations',
                                       'data'])
    return comb_dat(features,
                    observations,
                    dat)

def read_and_convert_to_sparse(filename, 
                               feat_ax,
                               fsep='\t',
                               header=0,
                               index_col=0, 
                               tuple_name='combined'):
    dat = pd.read_table(filename,
                        sep=fsep,
                        header=header,
                        index_col=index_col)
    feat_ax = mnni_utils.interpret_ax(feat_ax)
    dat = mnni_utils.rotate_obs_by_feat(dat = dat,
                                        feat_ax = feat_ax)
    features = dat.columns.values
    observations = dat.index.values
    dat = sparse.csr_matrix(dat.values)
    comb_dat = collections.namedtuple(tuple_name,
                                      ['features',
                                       'observations',
                                       'data'])
    return comb_dat(features,
                    observations,
                    dat)

def restrict_covered_features(dat,
                              feat_ax,
                              fraction_covered=0):
    """
    Restricts dataset to contain only cells with a certain level of coverage at given features
    
    Args:
        dat (named tuple): Tuple of counts (data), features, and observations for a dataset
        feat_ax (int/str): Axis where features are located. 0/rows: rows, 1/columns:columns
        fraction_covered (float between 0 and 1): Fraction of cells needed for a kept feature
    Returns:
        dat (named tuple): Tuple of counts (data), features, and observations for a dataset
    """
    # Check fraction_covered
    if 0 <= fraction_covered <= 1:
        pass
    else:
        raise ValueError('fraction_covered must be between 0 and 1, not {}'.format(fraction_covered))
    # Get axis information
    obs_ax = mnni_utils.transpose_ax(feat_ax)
    # Get the non-zero features
    nz_feat = dat.data.nonzero()[feat_ax]
    num_cells = dat.data.shape[obs_ax] * fraction_covered
    # Count non-zero features
    nz_counts = collections.Counter(nz_feat)
    # Restrict to features that are non-zero
    keep_idx = []
    for key in nz_counts:
        if nz_counts[key] > num_cells:
            keep_idx.append(key)
    # Restrict data
    if sparse.issparse(dat.data):
        if feat_ax == 0:
            res_dat = dat.data.tocsr()[keep_idx,:]
        else:
            res_dat = dat.data.tocsc()[:,keep_idx]
    else:
        if feat_ax == 0:
            res_dat = dat.data[keep_idx,:]
        else:
            res_dat = dat.data[:,keep_idx]
    res_tuple = collections.namedtuple(type(dat).__name__,
                                       ['features',
                                        'observations',
                                        'data'])
    return res_tuple(dat.features[keep_idx],
                     dat.observations,
                     res_dat)

def get_n_var_feat(dat,
                   feat_ax,
                   which_features=[],
                   n_feat=8000,
                   var_measure='vmr'):
    """
    Gets n highly variable features from a dataframe
    
    Args:
        dat (named tuple): Tuple of counts (data), features, and observations
        feat_ax (int/str): Axis where features are located. 0/rows: rows, 1/columns: columns
        which_features (array/list): Features to be considered. If None, uses all.
        n_feat (int): Number of features to consider
        var_measure (str): Metric to be used to determine highly variable genes
            sd for standard deviation
            vmr for variance mean ratio
            cv for coefficient of variation
    
    Returns:
        res_var (list): List of highly variable features
    """
    feat_ax = mnni_utils.interpret_ax(feat_ax)
    var_ax = mnni_utils.transpose_ax(feat_ax) #For numpy/scipy convention
    # Get index of features in common
    if len(which_features) > 0:
        num_idx = mnni_utils.find_num_index(dat.features,
                                            which_features)
    else:
        num_idx = list(range(0,dat.data.shape[feat_ax]))
    # Restrict to common features and calculate variance
    if feat_ax == 0:
        res_dat = dat.data.tocsr()[num_idx,:]
    else:
        res_dat = dat.data.tocsc()[:,num_idx]
    tmp_dat = res_dat.copy()
    tmp_dat.data **= 2
    variance = np.asarray(tmp_dat.mean(axis = var_ax)) - np.asarray(res_dat.mean(axis = var_ax))**2
    variance = variance.flatten()
    means = res_dat.mean(axis = var_ax)
    means[means == 0] = np.nan #Handle divide by zero warnings
    tmp_dat = None
    res_feat = dat.features[num_idx]
    # Handle var_measure
    if var_measure == 'sd':
        all_var = np.sqrt(variance)
    elif var_measure == 'vmr':
        all_var = variance / means
    elif var_measure == 'cv':
        all_var = np.sqrt(variance) / means
    if feat_ax == 1:
        all_var = all_var.T
    all_var = pd.DataFrame(all_var,columns = ['var'],index=res_feat)
    res_var = all_var['var'].sort_values(ascending=False).head(n_feat).index.values
    return res_var

def get_common_feat(dat_one,
                    dat_two,
                    feat_ax_one,
                    feat_ax_two,
                    n_feat,
                    var_one = 'vmr',
                    var_two = 'vmr'):
    """
    Obtains a list of common highly variable features between two datasets
    
    Args:
        dat_one (named tuple): Tuple of counts (data), features, and observations for dataset one
        dat_two (named tuple): Tuple of counts (data), features, and observations for dataset two
        feat_ax_one (int/str): Axis where dataset one's features are located. 0/rows: rows, 1/columns:columns
        feat_ax_two (int/str): Axis where dataset two's features are located. 0/rows: rows, 1/columns:columns
        n_feat (int): Number of features to consider
        var_one (str): Measure of variance for dataset one
        var_two (str): Measure of variance for dataset two
    
    Returns:
        share_var (list): List of common, highly variable features
    """
    # Get features
    feat_ax_one = mnni_utils.interpret_ax(feat_ax_one)
    feat_ax_two = mnni_utils.interpret_ax(feat_ax_two)
    feats = [dat_one.features,
             dat_two.features]
    # Get common features
    common_feat = functools.reduce(np.intersect1d,feats)
    print('There are {} features in common'.format(len(common_feat)))
    # Get n highly variable features
    hvg_mess = 'Getting {0} highly variable genes ({1}) in dataset {2}'
    print(hvg_mess.format(n_feat,
                          var_one,
                          'one'))
    feat_one = get_n_var_feat(dat = dat_one, 
                              which_features = common_feat,
                              feat_ax=feat_ax_one, 
                              n_feat=n_feat,
                              var_measure=var_one)
    print(hvg_mess.format(n_feat,
                          var_two,
                          'two'))
    feat_two = get_n_var_feat(dat = dat_two, 
                              which_features = common_feat,
                              feat_ax=feat_ax_two, 
                              n_feat=n_feat,
                              var_measure=var_two)
    # Get shared variable genes
    feat_list = [feat_one,feat_two]
    share_var = list(functools.reduce(np.intersect1d,feat_list))
    print('There are {} variable features in common'.format(len(share_var)))
    return share_var

def restrict_to_features(dat,
                         feat_ax,
                         which_features=[]):
    """
    Restricts dataset to to only specific features
    
    Args:
        dat (named tuple): Tuple of counts (data), features, and observations for a dataset
        feat_ax (int/str): Axis where features are located. 0/rows: rows, 1/columns:columns
        which_features (list/array): Optional, list of features to be considered
    Returns:
        dat (named tuple): Tuple of counts (data), features, and observations for a dataset
    """
    # Get index of features in common
    feat_ax = mnni_utils.interpret_ax(feat_ax)
    if len(which_features) > 0:
        num_idx = mnni_utils.find_num_index(dat.features,
                                            which_features)
    else:
        num_idx = list(range(0,dat.data.shape[feat_ax]))
    # Get the index of non-zero cells at features
    if feat_ax == 0:
        if sparse.issparse(dat.data):
            res_dat = dat.data.tocsr()[num_idx,:]
        else:
            res_dat = dat.data[num_idx,:]
    else:
        if sparse.issparse(dat.data):
            res_dat = dat.data.tocsc()[:,num_idx]
        else:
            res_dat = dat.data.tocsc()[:,num_idx]
    res_tuple = collections.namedtuple(type(dat).__name__,
                                       ['features',
                                        'observations',
                                        'data'])
    return res_tuple(dat.features[num_idx],
                     dat.observations,
                     res_dat)

def restrict_cells_with_features(dat,
                                 feat_ax,
                                 which_features=[]):
    """
    Restricts dataset to contain only cells with coverage at given features
    
    Args:
        dat (named tuple): Tuple of counts (data), features, and observations for a dataset
        feat_ax (int/str): Axis where features are located. 0/rows: rows, 1/columns:columns
        which_features (list/array): Optional, list of features to be considered
    Returns:
        dat (named tuple): Tuple of counts (data), features, and observations for a dataset
    """
    # Get index of features in common
    feat_ax = mnni_utils.interpret_ax(feat_ax)
    if len(which_features) > 0:
        num_idx = mnni_utils.find_num_index(dat.features,
                                            which_features)
    else:
        num_idx = list(range(0,dat.data.shape[feat_ax]))
    # Get the index of non-zero cells at features
    if feat_ax == 0:
        if sparse.issparse(dat.data):
            nz_cells = np.unique(dat.data.tocsr()[num_idx,:].nonzero()[1])
            res_dat = dat.data.tocsc()[:,nz_cells]
        else:
            nz_cells = np.unique(dat.data[num_idx,:].nonzero()[1])
            res_dat = dat.data[:,nz_cells]
    else:
        if sparse.issparse(dat.data):
            nz_cells = np.unique(dat.data.tocsc()[:,num_idx].nonzero()[0])
            res_dat = dat.data.tocsr()[nz_cells,:]
        else:
            nz_cells = np.unique(dat.data[:,num_idx].nonzero()[0])
            res_dat = dat.data[nz_cells,:]
    res_tuple = collections.namedtuple(type(dat).__name__,
                                       ['features',
                                        'observations',
                                        'data'])
    return res_tuple(dat.features,
                     dat.observations[nz_cells],
                     res_dat)

def get_rank_percentile(dat,
                        feat_ax):
    """
    Gets percentile rank of features per observation
    
    Args:
        dat (named tuple): Tuple of counts (data), features, and observations
        feat_ax (int/str): Axis where dataset one's features are located. 0/rows: rows, 1/columns:columns
    
    Returns:
        df_rank (dataframe): Percentile ranking of data from counts
    """
    if sparse.issparse(dat.data):
        df_rank = pd.DataFrame(dat.data.todense()).rank(pct=True,
                                                        axis = feat_ax) 
    else:
        df_rank = pd.DataFrame(dat.data).rank(pct=True,
                                              axis = feat_ax) 
    return df_rank

def reorder_tuple_feat(dat,
                       feat_ax,
                       idx):
    """
    Reorders features in a namedtuple to an arbritary order
    
    Args:
        dat (named tuple): Tuple of counts (data), features, and observations
        feat_ax (int/str): Axis where dataset one's features are located. 0/rows:rows, 1/columns:columns
        idx (list/array): List of numerical indices for features
    
    Returns:
        dat (named tuple): Tuple of counts (data), features, and observations
    """
    feat_ax = mnni_utils.interpret_ax(feat_ax)
    if feat_ax == 0:
        if sparse.issparse(dat.data):
            res_dat = dat.data.tocsr()[idx,:]
        else:
            res_dat = dat.data[idx,:]
    elif feat_ax == 1:
        if sparse.issparse(dat.data):
            res_dat = dat.data.tocsc()[:,idx]
        else:
            res_dat = dat.data[:,idx]
    reorder_tuple = collections.namedtuple(type(dat).__name__,
                               ['features',
                                'observations',
                                'data'])
    return reorder_tuple(dat.features[idx],
                         dat.observations,
                         res_dat)
    
def match_feature_order(dat_one,
                        dat_two,
                        feat_ax_one,
                        feat_ax_two):
    """
    Reorders two named tuples so they have the same order of features
    
    Args:
        dat_one (named tuple): Tuple of counts (data), features, and observations for dataset one
        dat_two (named tuple): Tuple of counts (data), features, and observations for dataset two
        feat_ax_one (int/str): Axis where dataset one's features are located. 0/rows: rows, 1/columns:columns
        feat_ax_two (int/str): Axis where dataset two's features are located. 0/rows: rows, 1/columns:columns
    
    Returns:
        dat_one (named tuple): Tuple of counts (data), features, and observations for dataset one
        dat_two (named tuple): Tuple of counts (data), features, and observations for dataset two
    """
    # Handle axes
    feat_ax_one = mnni_utils.interpret_ax(feat_ax_one)
    feat_ax_two = mnni_utils.interpret_ax(feat_ax_two)
    # Get values to make sure it is ok to reorder
    len_one = len(dat_one.features)
    len_two = len(dat_one.features)
    uniq_one = np.unique(dat_one.features)
    uniq_two = np.unique(dat_two.features)
    # Reorder axes
    if np.all(np.equal(uniq_one,uniq_two)) and (len_one == len_two) and (len_one == len(uniq_one)):
        df_one = pd.DataFrame(dat_one.features,
                              columns = ['features']).reset_index().set_index(['features'])
        df_one.columns = ['idx']
        df_two = pd.DataFrame(dat_two.features,
                              columns = ['features']).reset_index().set_index(['features'])
        df_two.columns = ['idx']
        idx_one = df_one['idx'].tolist()
        idx_two = df_two.loc[df_one.index]['idx'].tolist()
        dat_two = reorder_tuple_feat(dat_two,
                                     feat_ax=feat_ax_two,
                                     idx=idx_two)
    else:
        return ValueError('Cannot match features')
    return dat_one,dat_two

def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def corr_two_mods(dat_one,
                  dat_two,
                  correlation,
                  feat_ax_one,
                  feat_ax_two,
                  features=[]):
    """
    Generates a correlation matrix between highly variable features of two datasets
    
    Args:
        dat_one (named tuple): Tuple of counts (data), features, and observations for dataset one
        dat_two (named tuple): Tuple of counts (data), features, and observations for dataset two
        correlation (str): Description of correlation between datasets one and two
            positive or + for a positive correlation, negative or - for a negative correlation
        feat_ax_one (int/str): Axis where dataset one's features are located. 0/rows: rows, 1/columns:columns
        feat_ax_two (int/str): Axis where dataset two's features are located. 0/rows: rows, 1/columns:columns
        features (list): List of features in common between the two data sets
    
    Returns:
        df_corr (dataframe): Dataframe of correlation values between datasets one and two (obs by obs)
    """
    if len(features) > 0:
        dat_one = restrict_to_features(dat = dat_one,
                                       feat_ax=feat_ax_one,
                                       which_features = features)
        dat_two = restrict_to_features(dat = dat_two,
                                       feat_ax=feat_ax_two,
                                       which_features = features)
    # Reorder data so features match
    dat_one, dat_two = match_feature_order(dat_one = dat_one, 
                                           dat_two = dat_two,
                                           feat_ax_one = feat_ax_one,
                                           feat_ax_two = feat_ax_two)
    # Get ranking of genes per cell
    rank_one = get_rank_percentile(dat = dat_one, 
                                   feat_ax = feat_ax_one)
    rank_two = get_rank_percentile(dat = dat_two, 
                                   feat_ax = feat_ax_two)
    # Transpose if needed
    rank_one = mnni_utils.rotate_obs_by_feat(rank_one,
                                             feat_ax_one)
    rank_two = mnni_utils.rotate_obs_by_feat(rank_two,
                                             feat_ax_two)
    # Handle correlations
    if correlation == 'negative' or correlation == '-':
        rank_two = 1 - rank_two
    elif correlation == 'positive' or correlation == '+':
        pass
    else:
        raise ValueError('Unsupported correlation value')
    # Perform correlation
    n_one = rank_one.shape[0]
    corr = generate_correlation_map(x = rank_one, y = rank_two)
    # Make dataframe of correlations
    df_corr = pd.DataFrame(corr, 
                           index = dat_one.observations, 
                           columns = dat_two.observations)
    # Return correlation matrices
    return df_corr

def normalize_adj(adj,
                  norm_ax,
                  offset=1e-5):
    """
    Normalizes an adjacency matrix
    
    Args:
        adj (sparse matrix): Adjacency matrix across datasets
        norm_ax (int/str): Axis to normalize along. 0/rows: rows, 1/columns:columns
        offset (int): Offset to avoid divide by zero errors (in case mean is zero)
    
    Returns:
        norm_adj (sparse matrix): Normalized adjacency matrix
    """
    # Handle axis
    norm_ax = mnni_utils.interpret_ax(norm_ax)
    norm_ax = mnni_utils.transpose_ax(norm_ax)
    # Get reciprocal for matrix multiplication
    diags = sparse.diags(1/(adj.sum(axis=norm_ax)+offset).A.ravel())
    norm_adj = diags.dot(adj)
    return norm_adj

def gen_markov_unimodal(dat, 
                        feat_ax,
                        k,
                        n_comp, 
                        centered = False, 
                        scaled = False, 
                        svd = False, 
                        seed = 23):
    """
    Generates Markov matrix for a single dataset
    
    Args:
        dat (named tuple): Tuple of counts (data), features, and observations
        feat_ax (int/str): Axis where features are located. 0/rows: rows, 1/columns:columns
        k (int): Number of nearest neighbors
        n_comp (int): Number of components for dimensionaly reduction
        centered (boolean): If true, center data before dimensionality reduction (removes sparseness!)
        scaled (boolean): If true, scales data before dimensionality reduction
        svd (boolean): If true, uses SVD for dimensionality reduction. If false, uses PCA
        seed (int): Random seed
    
    Returns:
        markov (array): Markov matrix (sparse)
    """
    # Handle axes
    dat = mnni_utils.rotate_obs_by_feat(dat = dat.data,
                                        feat_ax = feat_ax)
    # Reduce data
    #dat = mnni_utils.center_scale_reduce(dat,
    #                                     feat_ax = 1,
    #                                     n_comp = n_comp,
    #                                     centered=centered,
    #                                     scaled=scaled,
    #                                     svd = svd,
    #                                     seed = seed)
    # Generate Markov matrix
    knn = NearestNeighbors(n_neighbors = k, 
                           metric = 'euclidean').fit(dat)
    g_knn = knn.kneighbors_graph(dat,
                                 mode = 'connectivity')
    markov = compute_jaccard_weights(X = g_knn, 
                                     k = k, 
                                     as_sparse = True)
    markov = normalize_adj(adj=markov,
                           norm_ax=1,
                           offset=1e-5)
    return markov

def corr_to_adj(corr, 
                k, 
                mutual=False):
    """
    Generates an adjacency matrix from a correlation matrix
    
    Args:
        corr (dataframe): Dataframe of correlation values
        k (int): Number of nearest neighbors for kNN
        mutual (boolean): If true, requires mutual nearest neighbors
        
    Returns:
        knn_s (sparse matrix): Adjacency matrix
    """
    # knn from correlation
    corr = pd.DataFrame(corr)
    knn = ((-corr).rank(axis=1) <= k).astype(float)
    if mutual:
        assert knn.shape[0] == knn.shape[1]
        knn = knn*knn.T # element wise 
    knn_s = sparse.csr_matrix(knn)
    return knn_s

def gen_adj_mtx(adj_one,
                adj_two,
                k,
                idx,
                gsamp,
                gadj,
                samp_info,
                max_knn=30):
    """
    Generates adjacency matrix between two datasets
    
    Args:
        adj_one (array): Adjacency matrix from dataset one
        adj_two (array): Adjacency matrix from dataset two
        k (int): Number of nearest neighbors
        idx (int): Iteration number (if not called in a loop set to zero)
        gsamp (list): List of sample names
        gadj (array): Previous iteration's adjacency matrix (regenerated if idx = 0)
        samp_info (dataframe): Sample information
        max_knn (int): Maximum number of nearest neighbors for a cell
    
    Returns:
        gadj (array): Updated across dataset adjacency matrix
        gsamp (list): Updated list of smaple names
        samp_info (dataframe): Updated sample information
        
    Assumptions:
        Called as part of a loop to generate an adjacency matrix over multiple k values
    """
    adj_one = (adj_one.multiply(adj_two.T)).astype(int)
    # Get information
    rows, cols = adj_one.nonzero()
    samples = (np.sort(np.unique(rows)))
    # Update adjacency
    if idx == 0:
        gsamp = samples.copy()
        gadj = sparse.lil_matrix(adj_one.copy())
        samp_info.loc[samples, 'round'] = idx
        samp_info.loc[samples, 'ksamp'] = adj_one[samples, :].sum(axis=1).reshape(-1,)
    else:
        for j in samples: 
            if j not in gsamp:
                ksamp = int(adj_one[j, :].sum(axis=1))
                if ksamp <= max_knn:
                    samp_info.loc[j, 'round'] = idx
                    samp_info.loc[j, 'ksamp'] = ksamp
                    gadj[j, :] = adj_one[j, :]
                    gsamp = np.append(gsamp, j)
    return [gadj, 
            gsamp, 
            samp_info]

def gen_markov_bimodal(adj, 
                       samples, 
                       samp_info, 
                       unimodal_markov):
    """
    Generates Markov matrix across datasets
    
    Args:
        adj (sparase matrix): Adjacency matrix across datasets
        samples (list): List of samples to iterate over
        samp_info (dataframe): Information on samples
        unimodal_markov (array): Within-modality Markov matrix
        
    Returns:
        bimodal_markov (array): Across-modality Markov matrix
        unimodal_markov (array): Within-modality Markov matrix
        samp_info (dataframe): Information on samples
    """
    bimodal_markov = normalize_adj(adj=adj,
                                   norm_ax=0, #HACK DOUBLE CHECK AXIS IN FUTURE
                                   offset=1e-5)
    unimodal_markov = unimodal_markov.tolil()
    for i in range(unimodal_markov.shape[0]):
        if i in samples:
            unimodal_markov[i,:] = 0
            unimodal_markov[i,i] = 1
        else:
            samp_info.loc[i,'round'] = -1
            unimodal_markov[:,i] = 0
    unimodal_markov = unimodal_markov.tocsr()
    unimodal_markov = normalize_adj(adj=unimodal_markov,
                                    norm_ax = 0, # HACK, DOUBLE CHECK AXIS IN FUTURE 
                                    offset = 1e-5)
    return [bimodal_markov,
            unimodal_markov, 
            samp_info]

def perform_imputation(dat, 
                       unimodal_markov, 
                       bimodal_markov,
                       feat_ax,
                       observations,
                       features,
                       tuple_name='imputed'):
    """
    Performs imputation of data across modalities
    
    Args:
        dat (sparse_matrix): Matrix of count data
        unimodal_markov (array): Other modality's Markov matrix
        bimodal_markov (array): Across modality Markov matrix
        feat_ax (int/str): Axis where features are located. 0/rows: rows, 1/columns:columns
        observations (array): Feature labels for imputed data
        features (array): Feature labels for imputed data
        tuple_name (str): Name for named tuple output
    
    Returns:
        imputed (dataframe): Imputed counts for modality opposite of dat
    """
    
    
    # Rotate to observations by features (if necessary)
    dat = mnni_utils.rotate_obs_by_feat(dat = dat,
                                            feat_ax = feat_ax)
    # Impute data
    imputed = unimodal_markov.dot(bimodal_markov.dot(dat))
    
    
    # Remove cells with nonzero counts
    nz_obs = np.unique(imputed.nonzero()[0])
    res_tuple = collections.namedtuple(tuple_name,
                                       ['features',
                                        'observations',
                                        'data'])
    return res_tuple(features,
                     observations[nz_obs],
                     imputed.tocsr()[nz_obs,:])

def gen_markov_and_impute(dat_one,
                          dat_two,
                          corr,
                          feat_ax_one,
                          feat_ax_two,
                          max_knn=30,
                          k_one= 10, 
                          k_two = 10,
                          n_comp = 50,
                          centered = False, 
                          scaled = False, 
                          svd = True, 
                          seed=23):
    """
    Generates Markov matrices and imputes across modalities
    
    Args:
        dat_one (named tuple): Tuple of counts (data), features, and observations for dataset one
        dat_two (named tuple): Tuple of counts (data), features, and observations for dataset two
        feat_ax_one (int/str): Dataset one feature axis. 0/rows: rows, 1/columns: columns
        feat_ax_two (int/str): Dataset two feature axis. 0/rows: rows, 1/columns: columns
        max_knn (int): Maximum number of nearest neighbors
        k_one (int): Number of nearest neighbors for dataset one
        k_two (int): Number of nearest neighbors for dataset two
        n_comp (int): Number of components for dimensionality reduction
        centered (boolean): If true, center data before dimensionality reduction (removes sparseness!)
        scaled (boolean): If true, scales data before dimensionality reduction
        svd (boolean): If true, uses SVD for dimensionality reduction
        seed (int): Random seeed
    
    Returns:
        imputed_12 (named tuple): Imputed values from dataset one to dataset two
        imputed_21 (named tuple): Imputed values from dataset two to dataset one
    """
    # Handle axes
    feat_ax_one = mnni_utils.interpret_ax(feat_ax_one)
    feat_ax_two = mnni_utils.interpret_ax(feat_ax_two)
    # Generate within network Markov
    print('Generating Markov for dataset one')
    markov_one = gen_markov_unimodal(dat = dat_one, 
                                     feat_ax = feat_ax_one, 
                                     n_comp = n_comp, 
                                     k = k_one,
                                     centered = centered, 
                                     scaled = scaled, 
                                     svd = svd)
    print('Generating Markov for dataset two')
    markov_two = gen_markov_unimodal(dat = dat_two, 
                                     feat_ax = feat_ax_two, 
                                     n_comp = n_comp, 
                                     k = k_two, 
                                     centered = centered, 
                                     scaled = scaled, 
                                     svd = svd)
    # Get number of cells
    obs_ax_one = mnni_utils.transpose_ax(feat_ax_one)
    obs_ax_two = mnni_utils.transpose_ax(feat_ax_two)
    n_one = dat_one.data.shape[obs_ax_one]
    n_two = dat_two.data.shape[obs_ax_two]
    # Generate adjacency matrix
    samp_in_one_info = pd.DataFrame(index=np.arange(n_one), 
                                    columns=['round', 
                                             'ksamp'])
    samp_in_two_info = pd.DataFrame(index=np.arange(n_two), 
                                    columns=['round', 
                                             'ksamp'])
    print('Generating adjacency matrix')
    for idx, k in enumerate(range(10, 210, 10)): #140
        adj_one = corr_to_adj(corr, 
                              k) 
        adj_two = corr_to_adj(corr.T, 
                              k) 
        if idx == 0:
            gsamp_in_one = []
            gsamp_in_two = []
            gadj_12 = []
            gadj_21 = []
        gadj_12, gsamp_in_one, samp_in_one_info = gen_adj_mtx(adj_one = adj_one,
                                                                adj_two = adj_two,
                                                                k = k, idx=idx,
                                                                gsamp = gsamp_in_one,
                                                                gadj = gadj_12,
                                                                samp_info = samp_in_one_info,
                                                                max_knn = max_knn)
        gadj_21, gsamp_in_two, samp_in_two_info = gen_adj_mtx(adj_one = adj_two,
                                                                adj_two = adj_one,
                                                                k = k, idx=idx,
                                                                gsamp = gsamp_in_two,
                                                                gadj = gadj_21,
                                                                samp_info = samp_in_two_info,
                                                                max_knn = max_knn)
        print('k: {0}, sample 1 #: {1}, sample 2 #: {2}'.format(k, 
                                                            len(gsamp_in_one),
                                                            len(gsamp_in_two)))

    gsamp_in_one.sort() 
    gsamp_in_two.sort()
    # Generate across modality markov
    print('Generating bimodal markov for dataset one')
    markov_12, markov_one, samp_in_one_info = gen_markov_bimodal(adj = gadj_12, 
                                                                 samples = gsamp_in_one, 
                                                                 samp_info = samp_in_one_info,
                                                                 unimodal_markov = markov_one)
    markov_21, markov_two, samp_in_two_info = gen_markov_bimodal(adj = gadj_21, 
                                                                 samples = gsamp_in_two, 
                                                                 samp_info = samp_in_two_info,
                                                                 unimodal_markov = markov_two)
    # Impute across modality
    print('Imputing across modalities (one to two)')   
    imputed_12 = perform_imputation(dat = dat_two.data, 
                                    unimodal_markov = markov_one,
                                    bimodal_markov = markov_12, 
                                    feat_ax = feat_ax_one, 
                                    observations = dat_one.observations,
                                    features = dat_two.features,
                                    tuple_name = 'one_to_two')
    print('Imputing across modalities (two to one)')   
    imputed_21 = perform_imputation(dat = dat_one.data, 
                                    unimodal_markov = markov_two,
                                    bimodal_markov = markov_21,
                                    feat_ax = feat_ax_two, 
                                    observations = dat_two.observations,
                                    features = dat_one.features,
                                    tuple_name = 'two_to_one')
    return [imputed_12, 
            imputed_21]
          
    
def read_dat_file(dat_fn, 
                  feat_ax, 
                  is_sparse =False,
                  tuple_name='dat',
                  obs_fn = None, 
                  feat_fn = None, 
                  fsep = '\t', 
                  header = 0, 
                  index_col = 0):
    """
    Reads a raw data file into a named tuple

    Args:
        dat_fn (str): Path to count file
        feat_ax (int/str): Axis where features are located. 0/rows: rows, 1/columns:columns
        is_sparse (boolean): If true, dat_fn is a sparse matrix
        tuple_name (str): Name of named tuple output
        obs_fn (str): Path to observation labels array (for sparse matrices)
        feat_fn (str): Path to feature label array (for sparse matrices)
        fsep (str): If not sparse, delimiter for file_one (pandas syntax)
        header (int): If not sparse, row number for header (pandas syntax)
        index_col (int): If not sparse, column number for index (pandas syntax)

    Returns:
        dat(named tuple): Tuple of counts (data), features, and observations
    """
    # Read data
    if is_sparse:
        print('Loading {}'.format(dat_fn))
        dat = load_sparse(matrix_fn = dat_fn, 
                          feat_fn = feat_fn, 
                          obs_fn = obs_fn, 
                          feat_ax = feat_ax, 
                          tuple_name = tuple_name)
    else:
        print('Loading {}'.format(dat_fn))
        dat = read_and_convert_to_sparse(filename = dat_fn, 
                                         feat_ax = feat_ax,
                                         fsep = fsep, 
                                         header = header, 
                                         index_col = index_col, 
                                         tuple_name = tuple_name)
    return dat

def impute_across_modalities(file_one, 
                             file_two, 
                             feat_ax_one, 
                             feat_ax_two,
                             var_one, 
                             var_two, 
                             n_feat, 
                             correlation,   
                             fraction_covered = .01, 
                             sparse_one=False, 
                             sparse_two=False,
                             obs_one = None, 
                             feat_one = None,
                             obs_two = None, 
                             feat_two = None,
                             fsep_one='\t',
                             fsep_two='\t',
                             header_one = 0, 
                             index_one = 0,
                             header_two = 0, 
                             index_two = 0,
                             k_one= 10, 
                             k_two = 10, 
                             max_knn=30, 
                             n_comp = 50,
                             centered = False, 
                             scaled = True, 
                             svd = True, 
                             seed=23):     

    """
    Default pipeline for imputing across modalties
    _one refers to dataset one, _two refers to dataset two

    Args:
        file_one (str): Path to count file for dataset one
        file_two (str): Path to count file for dataset two
        feat_ax_one (int/str): Axis where features are located. 0/rows: rows, 1/columns:columns
        feat_ax_two (int/str): Axis where features are located. 0/rows:rows,1/columns:columns
        var_one (str): Variance measure for dataset one (sd,vmr,cv)
        var_two (str): Variance measure for dataset two (sd,vmr,cv)
        n_feat (int): Number of highly variable gene to consider
        correlation (str): Expected correlation between datasets (negative or positive)
        fraction_covered (float): Percentage of cells that must have non-zero counts at a feature
        sparse_one (boolean): If true, file_one is a sparse matrix
        sparse_two (boolean): If true, file_two is a sparse matrix
        obs_one (str): Path to observation labels array (for sparse matrices)
        feat_one (str): Path to feature label array (for sparse matrices)
        obs_two (str): Path to observation labels array (for sparse matrices)
        feat_two (str): Path to feature labels array (for sparse matrices)
        fsep_one (str): If not sparse, delimiter for file_one (pandas syntax)
        fsep_two (str): If not sparse, delimiter for file_two (pandas syntax)
        header_one (int): If not sparse, row number for header (pandas syntax)
        index_one (int): If not sparse, column number for index (pandas syntax)
        k_one (int): Number of nearest neighbors for dataset one
        k_two (int): Number of nearest neighbors for dataset two
        max_knn (int): Maximum number of mutual nearest neighbors
        centered (boolean): If true, mean centers before dimensionality reduction (removes sparsness!)
        scaled (boolean): If true, scales before dimensionality reduction
        svd (boolean): If true, uses SVD for dimensionality reduction (necessary for sparse)
        seed (int): Random seed

    Returns:
        imputed_12 (named tuple): Imputed counts (data), feature and observation labels
            Imputation of dataset one into the space of dataset two
        imputed_21 (named tuple): Imputed counts (data), feature and observation labels
            Imputation of dataset two into the space of dataset one

    """
    # Read data
          
    dat_one = read_dat_file(dat_fn = file_one, 
                            feat_ax = feat_ax_one, 
                            is_sparse = sparse_one, 
                            tuple_name='comb_one',
                            obs_fn = obs_one, 
                            feat_fn = feat_one, 
                            fsep = fsep_one, 
                            header = header_one, 
                            index_col = index_one)
    dat_two = read_dat_file(dat_fn = file_two, 
                            feat_ax = feat_ax_two, 
                            is_sparse = sparse_two, 
                            tuple_name='comb_two',
                            obs_fn = obs_two, 
                            feat_fn = feat_two, 
                            fsep = fsep_two, 
                            header = header_two, 
                            index_col = index_two)
    # Handle axes
    ax_obs = 0
    ax_feat = 1
    init_mess = 'file_{0} has {1} observations and {2} features'      
    print(init_mess.format('one',
                           len(dat_one.observations),
                           len(dat_one.features)))
    print(init_mess.format('two',
                           len(dat_two.observations),
                           len(dat_two.features)))
    # Restrict to features with coverage
    dat_one = restrict_covered_features(dat = dat_one,
                                        feat_ax=feat_ax_one,
                                        fraction_covered = fraction_covered)
    dat_two = restrict_covered_features(dat = dat_two,
                                        feat_ax=feat_ax_two,
                                        fraction_covered = fraction_covered)
    cov_mess = 'Dataset {0} has {1} features with coverage'
    print(cov_mess.format('one',len(dat_one.features)))
    print(cov_mess.format('two',len(dat_two.features)))
    # Get common, highly variable features between datasets
    common_feat = get_common_feat(dat_one=dat_one,
                                  dat_two=dat_two,
                                  feat_ax_one = ax_feat, 
                                  feat_ax_two = ax_feat,
                                  n_feat = n_feat, 
                                  var_one = var_one,
                                  var_two = var_two)
    # Restrict data to common, highly variable features
    dat_one = restrict_to_features(dat = dat_one,
                                   feat_ax=feat_ax_one,
                                   which_features = common_feat)
    dat_two = restrict_to_features(dat = dat_two,
                                   feat_ax=feat_ax_two,
                                   which_features = common_feat)
    res_mess = 'Correlations: dataset {0} has {1} features'
    print(res_mess.format('one',len(dat_one.features)))
    print(res_mess.format('two',len(dat_two.features)))
    # Restrict cells to those that have at least one non-zero feature
    dat_one = restrict_cells_with_features(dat = dat_one,
                                           feat_ax=feat_ax_one,
                                           which_features = common_feat)
    dat_two = restrict_cells_with_features(dat = dat_two,
                                           feat_ax=feat_ax_two,
                                           which_features = common_feat)
    obs_mess = 'Dataset {0} has {1} observations with coverage'
    print(obs_mess.format('one',len(dat_one.observations)))
    print(obs_mess.format('two',len(dat_two.observations)))
    # Correlate data
    corr_cross = corr_two_mods(dat_one = dat_one, 
                               dat_two = dat_two,
                               correlation = correlation,
                               features = common_feat,
                               feat_ax_one = ax_feat,
                               feat_ax_two = ax_feat)
    # Reload data for imputation
    dat_one = read_dat_file(dat_fn = file_one, 
                            feat_ax = feat_ax_one, 
                            is_sparse = sparse_one, 
                            tuple_name='comb_one',
                            obs_fn = obs_one, 
                            feat_fn = feat_one, 
                            fsep = fsep_one, 
                            header = header_one, 
                            index_col = index_one)
    dat_two = read_dat_file(dat_fn = file_two, 
                            feat_ax = feat_ax_two, 
                            is_sparse = sparse_two, 
                            tuple_name='comb_two',
                            obs_fn = obs_two, 
                            feat_fn = feat_two, 
                            fsep = fsep_two, 
                            header = header_two, 
                            index_col = index_two)
    # Impute data
    imputed_12, imputed_21 = gen_markov_and_impute(dat_one = dat_one,
                                                   dat_two = dat_two,
                                                   corr = corr_cross,
                                                   feat_ax_one = ax_feat,
                                                   feat_ax_two = ax_feat,
                                                   max_knn=max_knn, 
                                                   k_one=k_one, 
                                                   k_two = k_two, 
                                                   n_comp = n_comp,
                                                   centered = centered, 
                                                   scaled = scaled,
                                                   svd = svd, 
                                                   seed=seed)
    return [imputed_12, 
            imputed_21]
