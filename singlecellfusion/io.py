"""
Functions used for input/output of loom files

Written by Wayne Doyle

(C) 2018 Mukamel Lab GPLv2

"""

import loompy
import numpy as np
import pandas as pd
import time
from scipy import sparse
import logging
import collections
import tables
import re
import time
from . import general_utils

# Start log
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__) 


def make_unique_ids(max_number):
    """
    Used to make an array of unique identifiers for a loom file
    """
    fstr = '0{}'.format(len(str(max_number)))
    init_list = list(range(0,max_number))
    id_arr = np.asarray([format(i,fstr) for i in init_list])
    return id_arr

def add_dense(count_file,
              loom_file,
              feature_axis,
              append = False,
              observation_id = None,
              feature_id = None,
              layer_id = '',
              sep = '\t',
              verbose = False, 
              **kwargs):
    """
    Adds a dense (non-sparse) data file to a loom file
    
    Args:
        count_file (str): Path to count data
        loom_file (str): Name of output loom file
        feature_axis (int/str): Axis containing features
            0 or rows for rows
            1 or columns for columns
        append (bool): If true, append data to loom_file. If false, generate new file
        observation_id (int): Number of row/column that specifies a unique cell label
            If None, auto-generated
            Same as CellID in loom documentation
        feature_id (int): Number of row/column that specifies a unique feature label
            If None, auto-generated
            Same as Accession in loom documentation
        layer_id (str): Name of layer to add count data to in loom_file
        sep (str): File delimiter. Same convention as pandas.read_table
        verbose (bool): If true, print logging messages
        **kwargs: Keyword arguments for pandas.read_table
    
    Returns:
        Generates loom file with:
            counts in layer specified by layer_id
            Column attribute CellID containing values from observation_id
            Row attribute Accession containing values from feature_id
    
    Assumptions:
        Expects at most one header column and one row column
    """
    # Start log
    logger.info('Adding {0} to {1}'.format(count_file,loom_file))
    # Read data
    if feature_axis == 0 or 'row' in feature_axis:
        dat = pd.read_table(filepath_or_buffer = count_file, 
                            sep = sep, 
                            header = observation_id,
                            index_col = feature_id,
                            **kwargs)
    elif feature_axis == 1 or 'col' in feature_axis:
        dat = pd.read_table(filepath_or_buffer = count_file,
                            sep = sep,
                            header = feature_id,
                            index_col = observation_id,
                            **kwargs)
        dat = dat.T
    # Prepare data for loom
    if feature_id is None:
        loom_feat = make_unique_ids(max_number = dat.shape[0])
    else:
        loom_feat = dat.index.values.astype(str)
    if observation_id is None:
        loom_obs = make_unique_ids(max_number = dat.shape[1])
    else:
        loom_obs = dat.columns.values.astype(str)
    dat = sparse.csc_matrix(dat.values)
    # Save to loom file
    if layer_id != '':
        if append:
            with loompy.connect(loom_file) as ds:
                ds.add_columns(layers = {'': sparse.csc_matrix(dat.shape,dtype=int),
                                         layer_id:dat},
                               row_attrs = {'Accession':loom_feat},
                               col_attrs = {'CellID':loom_obs})
        else:
            loompy.create(filename = loom_file,
                          layers = {'': sparse.csc_matrix(dat.shape,dtype=int),
                                    layer_id:dat},
                           row_attrs = {'Accession':loom_feat},
                           col_attrs = {'CellID':loom_obs})
    else:
        if append:
            with loompy.connect(loom_file) as ds:
                ds.add_columns(layers = {layer_id:dat},
                               row_attrs = {'Accession':loom_feat},
                               col_attrs = {'CellID':loom_obs})
        else:
            loompy.create(filename = loom_file,
                          layers = {layer_id:dat},
                           row_attrs = {'Accession':loom_feat},
                           col_attrs = {'CellID':loom_obs})
            
def batch_add_sparse(loom_file,
                     layers = dict(),
                     row_attrs = dict(),
                     col_attrs = dict(),
                     append = False,
                     empty_base = False,
                     batch_size = 512):
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
            WARNING: A dense array of all features features by batch_size observations will be generated
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
    batches = np.array_split(np.arange(start = 0, 
                                       stop = obs_size,
                                       step = 1),
                             np.ceil(obs_size/batch_size))
    for batch in batches:
        batch_layer = dict()
        if empty_base:
            batch_layer[''] = np.zeros((feat_size,batch.shape[0]), dtype = int)
        for key in layers:
            batch_layer[key] = layers[key].tocsc()[:,batch].toarray()
        batch_col = dict()
        for key in col_attrs:
            batch_col[key] = col_attrs[key][batch]
        if append:
            with loompy.connect(loom_file) as ds:
                ds.add_columns(layers = batch_layer,
                        row_attrs = row_attrs,
                        col_attrs = batch_col)
        else:
            loompy.create(filename = loom_file,
                          layers = batch_layer,
                          row_attrs = row_attrs,
                          col_attrs = batch_col)
            append = True

def find_10x_genome(filename):
    """
    Finds the name of the genome in a 10x Hd5 file
    
    Args:
        filename (str): Path to Hd5 10x count file
    
    Returns:
        genome (str): Name of genome identifier in 10x file
    """
    p = r'/(.*)/'
    genomes = set()
    with tables.open_file(filename, 'r') as f:
        for node in f.walk_nodes():
            s = str(node)
            match = re.search(p,s)
            if match:
                genomes.add(match.group(1))
    if len(genomes) == 1:
        return list(genomes)[0]
    else:
        raise ValueError('Too many genome options')

def h5_to_loom(h5_file,
               loom_file,
               genome = None,
               verbose=False):
    """
    Converts a 10x formatted H5 file into the loom format
    
    Args:
        h5_file (str): Name of input 10X h5 file
        loom_file (str): Name of output loom file
        genome (str): Name of genome in h5 file
            If None, automatically detects
        verbose (bool): If true, prints logging messages
    
    Modified from http://cf.10xgenomics.com/supp/cell-exp/megacell_tutorial-1.0.1.html
    """
    if genome is None:
        genome = find_10x_genome(filename = h5_file)
        if verbose:
            print('The 10x genome is {}'.format(genome))
    # Get relevent information from file
    if verbose:
        print('Finding 10x data in h5 file {}'.format(h5_file))
        t_search = time.time()
    with tables.open_file(h5_file, 'r') as f:
        try:
            dsets = {}
            for node in f.walk_nodes('/{}'.format(genome), 'Array'):
                dsets[node.name] = node.read()
        except tables.NoSuchNodeError:
            raise Exception('Genome {} does not exist in this file'.format(genome))
        except KeyError:
            raise 'File is missing one or more required datasets'
        if verbose:
            t_write = time.time()
            time_run, time_fmt = general_utils.format_run_time(t_search,t_write)
            print('Found data in {0:.2f} {1}'.format(time_run,time_fmt))
            print('Adding data to loom_file {}'.format(loom_file))
        matrix = sparse.csc_matrix((dsets['data'],
                                    dsets['indices'],
                                    dsets['indptr']),
                                    shape = dsets['shape'])
        row_attrs = {'Name':dsets['gene_names'].astype(str),
            'Accession':dsets['gene'].astype(str)}
        col_attrs = {'CellID':dsets['barcodes'].astype(str)}
        layers = {'counts':matrix}
        batch_add_sparse(loom_file = loom_file,
            layers = layers,
            row_attrs = row_attrs,
            col_attrs = col_attrs,
            append = False,
            empty_base = True,
            batch_size = 1000)
        if verbose:
            t_end = time.time()
            time_run,time_fmt = general_utils.format_run_time(t_search,t_write)
            print('Wrote loom file in {0:.2f} {1}'.format(t_write,t_end))
