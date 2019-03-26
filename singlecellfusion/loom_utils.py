"""
Collection of loom-specific utilities
    
Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
import logging

# Start log
lu_log = logging.getLogger(__name__)


def get_pct(loom_file,
            num_val,
            columns=True):
    if columns:
        axis = 1
    else:
        axis = 0
    with loompy.connect(filename=loom_file, mode='r') as ds:
        pct = num_val / ds.shape[axis] * 100
    return pct


def get_attr_index(loom_file,
                   attr=None,
                   columns=False,
                   as_bool=True,
                   inverse=False):
    """
    Gets index for desired attributes in a loom file
    
    Args:
        loom_file (str): Path to loom file
        attr (str): Optional, attribute used to restrict index
            If None, all elements are included
        columns (boolean): Specifies if pulling rows or columns
            True: column attributes
            False: row attributes
        as_bool (bool): Return as boolean (true) or numerical (false) array
        inverse (bool): If true, returns inverse of index
            All trues are false, all falses are true
            
    Returns:
        idx (1D array): Index of attributes to use
            boolean if as_bool, numerical if not as_bool
        
    Assumptions:
        attr specifies a boolean array attribute in loom_file
    """

    with loompy.connect(filename=loom_file, mode='r') as ds:
        if columns:
            if attr:
                idx = ds.ca[attr].astype(bool)
            else:
                idx = np.ones((ds.shape[1],), dtype=bool)
        else:
            if attr:
                idx = ds.ra[attr].astype(bool)
            else:
                idx = np.ones((ds.shape[0],), dtype=bool)
    if inverse:
        idx = np.logical_not(idx)
    if as_bool:
        pass
    else:  # ASSUMPTION: 1D array input
        idx = np.where(idx)[0]
    return idx


def make_layer_list(layers):
    """
    Makes a list of layers to include when looping over a loom file

    Args:
        layers (str/list): Layer(s) in loom file to include

    Returns:
        out (list): Layer(s) in loom file to include
            Transformed to list and '' is added if not included
    """
    if isinstance(layers, str):
        if layers == '':
            out = ['']
        else:
            out = ['', layers]
    elif isinstance(layers, list):
        layers = set(layers)
        if '' in layers:
            out = list(layers)
        else:
            layers.add('')
            out = list(layers)
    else:
        raise ValueError('Unsupported type for layers')
    return out
