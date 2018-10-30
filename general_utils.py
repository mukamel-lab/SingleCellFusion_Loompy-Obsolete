"""
General-purpose utility functions

Written by Wayne Doyle unless otherwise noted
"""
import collections
import pandas as pd
import numpy as np
import os
import pwd
import datetime
import re
import subprocess
import pysam
import io
import gzip
import bz2 
from scipy import sparse

bin_dir = os.path.dirname(os.path.realpath(__file__))

def nat_sort(items):
    """
    Takes a list of items and sorts them in a natural order

    Args:
        items (list): List of items

    Copied from Mark Byer's post on StackOverflow:
    https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(items,key = alphanum_key)

def format_run_time(t0,t1):
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
                     include_y = False):
    """
    Returns a dictionary of chromosomes and their sizes (in bases)
    
    Args:
        prefix (bool): If true, include chr prefix
        include_y (bool): If true include Y chromosome
    
    Returns:
        chrom_dict (dict): keys are chromosomes, values are lengths
    """
    chrom_dict = {'1':195471971,
            '2':182113224,
            '3':160039680,
            '4':156508116,
            '5':151834684,
            '6':149736546,
            '7':145441459,
            '8':129401213,
            '9':124595110,
            '10':130694993,
            '11':122082543,
            '12':120129022,
            '13':120421639,
            '14':124902244,
            '15':104043685,
            '16':98207768,
            '17':94987271,
            '18':90702639,
            '19':61431566,
            'X':171031299,
            }
    if include_y:
        chrom_dict['Y'] = 91744698
    if prefix:
        mod = dict()
        for key in chrom_dict.keys():
            new_key = 'chr' + key
            mod[new_key] = chrom_dict[key] 
        chrom_dict = mod_dict
    return chrom_dict

def expand_sparse(mtx,
                  col_index = None,
                  row_index = None,
                  col_N = None,
                  row_N = None,
                  dtype=float):
    """
    Expands a sparse matrix
    
    Args:
        mtx (sparse 2D array): Matrix from a subset of loom file
        col_index (1D array): Numerical indices of columns included in mtx
        row_index (1D array): Numerical indices of rows included in mtx
        col_N (int): Number of loom file columns
        row_N (int): Number of loom file rows
        dtype (str): Type of data in output matrix
    
    Returns:
        mtx (sparse 2D array): mtx with missing values included as zeros

    Warning:
        Do not use on transposed matrices
    """
    if col_index is None and row_index is None:
        pass
    elif col_index is not None and col_N is None:
        raise ValueError('Must provide both col_index and col_N')
    elif row_index is not None and row_N is None:
        raise ValueError('Must provide both row_index and row_N')
    elif col_index is None:
        mtx = sparse.coo_matrix((mtx.data,
                                 (row_index[mtx.nonzero()[0]],
                                  mtx.nonzero()[1])),
                                shape = (row_N,mtx.shape[1]),
                                dtype = dtype)
    elif row_index is None:
        mtx = sparse.coo_matrix((mtx.data,
                                 (mtx.nonzero()[0],
                                  col_index[mtx.nonzero()[1]])),
                                shape = (mtx.shape[0],col_N),
                                dtype = dtype)
    else: 
        mtx = sparse.coo_matrix((mtx.data,
                                 (row_index[mtx.nonzero()[0]], 
                                  col_index[mtx.nonzero()[1]])),
                                shape = (row_N, col_N), 
                                dtype = dtype)
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
    gene_ids = np.array(list(map(lambda x: re.sub(r'\..*$','', x),gene_ids)))
    return gene_ids
