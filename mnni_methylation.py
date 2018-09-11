"""
General utitilies for performing MNNI imputation

Written by Wayne Doyle unless noted
"""

# Import packages
import pandas as pd
import numpy as np
from scipy import sparse
import mnni_smooth
import mnni_utils
from sklearn.neighbors import NearestNeighbors

def split_mc_c(df,
               ax,
               mc_suffix = '_mC',
               c_suffix = '_C'):
    """
    Splits a dataframe into dataframes containing mC and C counts
    
    Args:
        df (dataframe): Contains both mC and C counts
        ax (str/int): Axis where suffices are located 
            0 or rows for rows
            1 or columns for columns
        mc_suffix (str): Suffix for mC cells
        c_suffix (str): Suffix for C cells
        
    Returns:
        df_mc (dataframe): mC counts per cell
        df_c (dataframe): C counts per cell
        
    Assumptions:
        Features or cells from mC or C will be marked by specific suffices
    
    Modified from code written by Fangming Xie
    """
    df_c = df.filter(regex="_c$", axis = ax)
    df_mc = df.filter(regex="_mc$", axis = ax)
    if ax == 0:
        df_c.index = [idx[:-len('_c')] for idx in df_c.index] 
        df_mc.index = [idx[:-len('_mc')] for idx in df_mc.index] 
    elif ax == 1:
        df_c.columns = [col[:-len('_c')] for col in df_c.columns] 
        df_mc.columns = [col[:-len('_mc')] for col in df_mc.columns] 
    else:
        raise ValueError('ax value ({}) is unsupported'.format(ax))
    return [df_mc,df_c]

def calculate_mc_over_c(df_mc,
                        df_c,
                        feat_ax,
                        basecall_cutoff = 100, 
                        sufficient_coverage = 1,
                        impute_missing=False):
    """
    Performs actual calculation of mC/C
    
    Args:
        df_mc(dataframe): mC counts per cell
        df_c(dataframe): C counts per cell
        feat_ax (int or str): Axis where features are located
            0 or rows for rows
            1 or columns for columns
        basecall_cutoff (int): Number of C's for a loci to be considered
        sufficient_coverage (int): Number of cells for a loci to be considered
        impute_missing (boolean): Impute missing mC/C at a feature 
            Imputation is with average of all cells
    
    Returns:
        df_mcc (dataframe): mC/C counts
    
    Modified from code written by Fangming Xie
    """
    # Handle axis
    feat_ax = mnni_utils.interpret_ax(feat_ax)
    ax = mnni_utils.transpose_ax(feat_ax)
    # Find if a gene is sufficiently covered in cells
    condition = (df_c > basecall_cutoff).sum(axis=ax) >= (sufficient_coverage * df_c.shape[ax])
    condition = condition[condition].index
    # Get mC/C matrix
    df_c_nan = df_c.copy()
    df_c_nan[df_c < basecall_cutoff] = np.nan
    if ax == 0:
        df_mcc = df_mc[condition].divide(df_c_nan[condition], axis = ax)
    else:
        df_mcc = df_mc.loc[condition].divide(df_c_nan.loc[condition],axis=ax)
    # Impute missing values
    if impute_missing:
        means = df_mcc.mean(axis = ax)
        if ax == 0:
            fill_value = pd.DataFrame({idx: means for idx in df_mcc.index}).T
        else:
            fill_value = pd.DataFrame({col: means for col in df_mcc.columns})
        df_mcc.fillna(fill_value,inplace=True,axis = ax)
    # Return data
    return df_mcc
    
def get_mcc(df,
            feat_ax,
            suffix_ax,
            basecall_cutoff = 100, 
            sufficient_coverage = 1,
            impute_missing=False,
            mc_suffix = '_mC',
            c_suffix = '_C'):
    """
    Calculates mC/C content per cell
    
    Args:
        df (dataframe): Contains both mC and C counts
        feat_ax (int or str): Axis where features are located
            0 or rows for rows
            1 or columns for columns
        suffix_ax (int/str): Axis where suffices are located 
            0 or rows for rows
            1 or columns for oclumns
        basecall_cutoff (int): Number of C's for a loci to be considered
        sufficient_coverage (int): Number of cells for a loci to be considered
        impute_missing (boolean): Impute missing mC/C at a feature 
            Imputation is with average of all cells
        mc_suffix (str): Suffix for mC cells
        c_suffix (str): Suffix for C cells
        
    
    Returns:
        df_mcc (dataframe): mC/C counts
    
    Modified from code written by Fangming Xie
    """
    # Handle axis
    feat_ax = mnni_utils.interpret_ax(feat_ax)
    suffix_ax = mnni_utils.interpret_ax(suffix_ax)
    if feat_ax == 0:
        ax = 1
    else:
        ax = 0
    # Split dataframe
    df_mc, df_c = split_mc_c(df=df,
                             ax=suffix_ax,
                             mc_suffix = mc_suffix,
                             c_suffix = c_suffix,)
    # Get mC/C
    df_mcc = calculate_mc_over_c(df_mc = df_mc,
                                 df_c = df_c,
                                 feat_ax = feat_ax,
                                 basecall_cutoff = basecall_cutoff, 
                                 sufficient_coverage = sufficient_coverage,
                                 impute_missing=impute_missing,)
    
    return df_mcc

def get_bin_mc_and_c(filename,
                     feat_ax,
                     suffix_ax,
                     basecall_cutoff=100, 
                     sufficient_coverage=1, 
                     impute_missing=True,
                     fsep='\t',
                     compression='infer',
                     mc_suffix = '_mC',
                     c_suffix = '_C'):
    """
    Determines mC/C for genomic regions
    
    Args:
        filename (str): Path to file containing count data
        fdf (dataframe): Contains both mC and C counts
        feat_ax (int or str): Axis where features are located
            0 or rows for rows
            1 or columns for columns
        suffix_ax (int/str): Axis where suffices are located 
            0 or rows for rows
            1 or columns for oclumns
        basecall_cutoff (int): Number of C's for a loci to be considered
        sufficient_coverage (int): Number of cells for a loci to be considered
        impute_missing (boolean): Impute missing mC/C at a feature 
            Imputation is with average of all cells
        fsep (str): Delimiter as defined by pandas read_table
        compression (str): Compression as defined by pandas read_table
        mc_suffix (str): Suffix for mC cells
        c_suffix (str): Suffix for C cells
        
    Returns:
        df (dataframe): mC/C values for all cells
        
    Assumptions:
        filename: Bins are indicated by a multi-index with the labels chr and bin
        
    """
    # Read dataframe
    feat_ax = mnni_utils.interpret_ax(feat_ax)
    suffix_ax = mnni_utils.interpret_ax(suffix_ax)
    if feat_ax == 0:
        df = pd.read_table(filename, 
                           sep = fsep, 
                           header = 0, 
                           index_col = None,
                           compression = compression,
                           dtype = str)
    elif feat_ax == 1:
        df = pd.read_table(filename,sep=fsep,
                           header=None,
                           index_col = 0,
                           compression=compression,
                           dtype=str)
        df = df.T
        suffix_ax = mnni_utils.transpose_ax(suffix_ax)
        feat_ax = 0
    else:
        raise ValueError('feat_ax value is unsupported')
    # Get rid of multi-index
    df['region'] = df['chr'] + '_' + df['bin']
    df = df.set_index('region',drop = True)
    df = df.drop(['chr','bin'], axis = 1)
    df = df.astype(int)
    # Get mC/C
    df = get_mcc(df = df,
                 feat_ax = feat_ax,
                 suffix_ax = suffix_ax,
                 basecall_cutoff = basecall_cutoff, 
                 sufficient_coverage = sufficient_coverage,
                 impute_missing=impute_missing,
                 mc_suffix = mc_suffix,
                 c_suffix = c_suffix)
    # Transpose to format expected by user
    if feat_ax == 0:
        df = df.T
    return df
        
def smooth_mcc(df,
               feat_ax,
               suffix_ax,
               distances = None,
               indices = None,
               t = 1, 
               k = 30, 
               ka = 4, 
               epsilon = 1, 
               p = 0.9,
               basecall_cutoff = 10, 
               sufficient_coverage = 1, 
               impute_missing = False,
               mc_suffix = '_mC', 
               c_suffix = '_C'):
    """
    Smooths mC and C, and get's mC/C for smoothed data
    
    Args:
        df (dataframe): Dataframe of observed mC and C values
        feat_ax (int or str): Axis where features are located
            0 or rows for rows
            1 or columns for columns
        suffix_ax (int/str): Axis where suffices are located 
            0 or rows for rows
            1 or columns for oclumns
        distances (dataframe): Distance values for KNN of df
        indices (dataframe): Indices for KNN of df
        t (int): Diffusion parameter for smoothing
        k (int): Number of nearest neighbors for smoothing
        ka (int): Normalization for smoothing
            Distances will be normalized to this value
        epsilon (int): 2 * error for Gaussian kernel (used in smoothing)
        p (float):  Proportion of smoothing from own cell
        basecall_cutoff (int): Number of C's for a loci to be considered
        sufficient_coverage (int): Number of cells for a loci to be considered
        impute_missing (boolean): Impute missing mC/C at a feature 
            Imputation is with average of all cells
        fsep (str): Delimiter as defined by pandas read_table
        compression (str): Compression as defined by pandas read_table
        mc_suffix (str): Suffix for mC cells
        c_suffix (str): Suffix for C cells
        
    Assumptions:
        Requires a pre-computed KNN of cells
    
    Returns:
        df_mcc (dataframe): Smoothed mC/C counts (observations x features)
    
    """
    # Get raw data
    feat_ax = mnni_utils.interpret_ax(feat_ax)
    suffix_ax = mnni_utils.interpret_ax(suffix_ax)
    if feat_ax == 0:
        df = df.T
        suffix_ax = mnni_utils.transpose_ax(suffix_ax)
    df_mc, df_c = split_mc_c(df=df,
                             ax=suffix_ax,
                             mc_suffix = mc_suffix,
                             c_suffix = c_suffix)
    # Smooth
    print('Smoothing cytosines')
    smooth_c = mnni_smooth.smooth(data = sparse.coo_matrix(df_c), 
                                  provide_knn = True,
                                  distances = distances.values, 
                                  indices = indices.values,
                                  t = t, 
                                  k = k, 
                                  ka = ka, 
                                  epsilon = epsilon,
                                  p = p)
    print('Smoothing methylcytosines')
    smooth_mc = mnni_smooth.smooth(data = sparse.coo_matrix(df_mc), 
                                   provide_knn = True,
                                   distances = distances.values, 
                                   indices = indices.values,
                                   t = t, 
                                   k = k, 
                                   ka = ka, 
                                   epsilon = epsilon,
                                   p = p)
    # Make into dataframe
    smooth_c = pd.DataFrame(smooth_c.todense(), 
                            columns = df_c.columns, 
                            index=df_c.index)
    smooth_mc = pd.DataFrame(smooth_mc.todense(), 
                             columns = df_mc.columns, 
                             index=df_mc.index)
    # Get mC/C
    df_mcc = calculate_mc_over_c(df_mc=smooth_mc,
                                 df_c=smooth_c,
                                 feat_ax=1,
                                 basecall_cutoff = basecall_cutoff, 
                                 sufficient_coverage = sufficient_coverage,
                                 impute_missing=impute_missing)
    # Transpose if user's desire
    if feat_ax == 0:
        df_mcc = df_mcc.T
    return df_mcc

def scale_methylation(df):
    """
    Scales mC/C values by standard deviation of all features in all cells
    
    Args:
        df (dataframe): Dataframe of mC/C values
    
    Returns:
        scaled (dataframe): Dataframe of mC/C values scaled by standard deviation of all loci/cells
    """
    scale_factor = np.std(df.values.flatten())
    scaled = df / scale_factor
    return scaled
