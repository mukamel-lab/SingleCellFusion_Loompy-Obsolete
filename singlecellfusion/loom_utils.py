"""
Collection of loom-specific utilities
    
Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
import pandas as pd
import logging

# Start log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def transfer_attributes(loom_source,
                        loom_destination,
                        attributes,
                        id_source,
                        id_destination):
    """
    Transfers attributes from one loom file to another
    
    Args:
        loom_source (str): Path to loom file containing attributes
        loom_destination (str): Path to loom file receiving attributes
        attributes (dict): Dictionary of attributes to transfer
            key: Type of attribute (ca,ra,col_graphs,row_graphs)
            values: List of attributes in key
        id_source (str): loom_source attribute for matching order
            There must be shared values between id_source and id_destination
        id_destination (str): loom_destination attribute for matching order
            There must be shared values between id_source and id_destination
    """
    with loompy.connect(loom_source) as ds_src:
        with loompy.connect(loom_destination) as ds_dest:
            for key in attributes:
                if key == 'attrs':
                    for attr in attributes[key]:
                        if attr in ds_src.attrs.keys():
                            ds_dest.attrs[attr] = ds_src.attrs[attr]
                        else:
                            logger.warning('{} not in file'.format(attr))
                else:
                    if key == 'ca' or key == 'col_graphs':
                        lookup_src = ds_src.ca[id_source]
                        src_size = ds_src.shape[1]
                        lookup_dest = ds_dest.ca[id_destination]
                        dest_size = ds_dest.shape[1]
                    elif key == 'ra' or key == 'row_graphs':
                        lookup_src = ds_src.ra[id_source]
                        src_size = ds_src.shape[0]
                        lookup_dest = ds_dest.ra[id_destination]
                        dest_size = ds_dest.shape[0]
                    lookup_src = pd.DataFrame(np.arange(start=0,
                                                        stop=src_size,
                                                        step=1),
                                              columns=['src_idx'],
                                              index=lookup_src)
                    lookup_dest = pd.DataFrame(np.arange(start=0,
                                                         stop=dest_size,
                                                         step=1),
                                               columns=['dest_idx'],
                                               index=lookup_dest)
                    lookup = pd.merge(lookup_src,
                                      lookup_dest,
                                      how='left',
                                      left_index=True,
                                      right_index=True)
                    if np.any(lookup.isnull()):
                        raise ValueError('ID mismatch')
                    for attr in attributes[key]:
                        new_index = np.argsort(lookup['dest_idx'].values)
                        if key == 'ca':
                            if attr in ds_src.ca.keys():
                                tmp = ds_src.ca[attr]
                                tmp = tmp[new_index]
                                ds_dest.ca[attr] = tmp
                            else:
                                logger.warning('{} not in file'.format(attr))
                        elif key == 'ra':
                            if attr in ds_src.ra.keys():
                                tmp = ds_src.ra[attr]
                                tmp = tmp[new_index]
                                ds_dest.ra[attr] = tmp
                            else:
                                logger.warning('{} not in file'.format(attr))
                        elif key == 'col_graphs':
                            if attr in ds_src.col_graphs.keys():
                                tmp = ds_src.col_graphs[attr].tocsr()
                                tmp = tmp[new_index]
                                ds_dest.col_graphs[attr] = tmp
                            else:
                                logger.warning('{} not in file'.format(attr))
                        elif key == 'row_graphs':
                            if attr in ds_src.row_graphs.keys():
                                tmp = ds_src.row_graphs[attr].tocsr()
                                tmp = tmp[new_index, :]
                                ds_dest.row_graphs[attr] = tmp
                            else:
                                logger.warning('{} not in file'.format(attr))
                        else:
                            raise ValueError(
                                'Unsupported attribute {}'.format(key))
