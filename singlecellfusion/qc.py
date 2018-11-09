"""
Collection of functions used to perform quality control analysis
    
Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
import time
import logging
from . import general_utils
from . import loom_utils

# Start log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def label_covered_features(loom_file,
                           layer,
                           out_attr,
                           min_count=1,
                           fraction_covered=0.01,
                           col_attr=None,
                           row_attr=None,
                           batch_size=512,
                           verbose=False):
    """
    Finds features with at least n counts in m percent of cells
    
    Args:
        loom_file (str): Path to loom file
        layer (str): Layer of counts to consider
        out_attr (str): Name of row attribute specifying valid features
        min_count (int/float): Minimum count for a covered feature (>=)
        fraction_covered (float): Mininum fraction of cells with coverage (>=)
            If None, only oen cells is needed
        col_attr (str): Optional, attribute to restrict cells by
        row_attr (str): Optional, attribute to restrict features by
        batch_size (int): Size of chunks
            Dense array of batch_size by number of cells will be generated
        verbose (bool): Print logging messages
    """
    # Get indices for items of interest
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=col_attr,
                                        columns=True,
                                        as_bool=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=row_attr,
                                        columns=False,
                                        as_bool=True,
                                        inverse=False)
    layers = loom_utils.make_layer_list(layers=layer)
    if fraction_covered is None:
        fraction_covered = 0
    if verbose:
        logger.info('Finding valid features for {}'.format(loom_file))
        t0 = time.time()
    # Get index of valid features
    with loompy.connect(loom_file) as ds:
        valid_idx = np.zeros((ds.shape[0],), dtype=int)
        num_cells = np.sum(col_idx)
        for (_, selection, view) in ds.scan(items=row_idx,
                                            layers=layers,
                                            batch_size=batch_size,
                                            axis=0):
            min_num = np.sum(view.layers[layer][:, col_idx] >= min_count,
                             axis=1)
            if fraction_covered is None:
                valid_idx[selection] = (min_num / num_cells) > 0
            else:
                valid_idx[selection] = (min_num / num_cells) >= fraction_covered
        ds.ra[out_attr] = valid_idx
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        logger.info('Found features in {0:.2f} {1}'.format(time_run, time_fmt))


def label_covered_cells(loom_file,
                        layer,
                        out_attr,
                        min_count=1,
                        fraction_covered=None,
                        col_attr=None,
                        row_attr=None,
                        batch_size=512,
                        verbose=False):
    """
    Finds cells with at least n counts in m percent of features
    
    Args:
        loom_file (str): Path to loom file
        layer (str): Layer of counts to consider
        out_attr (str): Name of row attribute specifying valid features
        min_count (int/float): Mininum count for a covered feature (>=)
        fraction_covered (float): Mininum fraction of covered features (>=)
            If not provided, only one feature must have min_val
        col_attr (str): Optional, attribute to restrict cells by
        row_attr (str): Optional, attribute to restrict features by
        batch_size (int): Size of chunks
            Dense array of batch_size by number of cells will be generated
        verbose (bool): Print logging messages
    """
    # Get indices for items of interest
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=col_attr,
                                        columns=True,
                                        as_bool=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=row_attr,
                                        columns=False,
                                        as_bool=True,
                                        inverse=False)
    layers = loom_utils.make_layer_list(layers=layer)
    if verbose:
        logger.info('Finding valid cells for {}'.format(loom_file))
        t0 = time.time()
    # Get index of valid features
    with loompy.connect(loom_file) as ds:
        valid_idx = np.zeros((ds.shape[1],), dtype=int)
        num_feat = np.sum(row_idx)
        for (_, selection, view) in ds.scan(items=col_idx,
                                            layers=layers,
                                            batch_size=batch_size,
                                            axis=1):
            min_num = np.sum(view.layers[layer][row_idx, :] >= min_count,
                             axis=0)
            if fraction_covered is None:
                valid_idx[selection] = (min_num / num_feat) > 0
            else:
                valid_idx[selection] = (min_num / num_feat) >= fraction_covered
        ds.ca[out_attr] = valid_idx
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        logger.info('Found cells in {0:.2f} {1}'.format(time_run, time_fmt))
