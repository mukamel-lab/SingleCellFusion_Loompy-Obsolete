"""
Helper functions for manipulating features
The typical user will NOT need to use these functions

Written by Wayne Doyle

(C) 2019 Mukamel Lab GPLv2
"""
import loompy
import numpy as np
import pandas as pd
import time
import logging
from scipy.stats import kruskal
import functools
from . import utils

feat_log = logging.getLogger(__name__)


def perform_cluster_kruskal(loom_file,
                            layer='',
                            cluster_attr='ClusterID',
                            cell_attr='CellID',
                            feat_attr='Accession',
                            valid_ca=None,
                            valid_ra=None,
                            batch_size=512):
    """
    Performs a Kruskal-Wallis test per cluster for all valid genes

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer of counts
        cluster_attr (str): Column attribute containing cluster IDs
        cell_attr (str): Column attribute containing unique cell IDs
        feat_attr (str): Row attribute containing unique feature IDs
        valid_ra (str): Row attribute specifying valid features
        valid_ca (str): Column attribute specifying valid cells
        batch_size (int): Size of chunks for batches
    """
    # Set-up for subsequent steps
    col_idx = utils.get_attr_index(loom_file=loom_file,
                                   attr=valid_ca,
                                   columns=True,
                                   as_bool=False,
                                   inverse=False)
    row_idx = utils.get_attr_index(loom_file=loom_file,
                                   attr=valid_ra,
                                   columns=False,
                                   as_bool=False,
                                   inverse=False)
    layers = utils.make_layer_list(layer)
    result_dict = dict()
    # Perform Kruskal-Wallis test
    with loompy.connect(loom_file) as ds:
        # Get lookups
        gene_lookup = pd.DataFrame({'indices': np.arange(ds.shape[0])},
                                   index=ds.ra[feat_attr])
        cluster_lookup = pd.DataFrame({'clusters': ds.ca[cluster_attr]},
                                      index=ds.ca[cell_attr])
        cluster_lookup = cluster_lookup.iloc[col_idx, :]
        cell_lookup = pd.DataFrame({'indices': np.arange(ds.shape[1])},
                                   index=ds.ca[cell_attr])
        cell_lookup = cell_lookup.iloc[col_idx, :]
        unq_clusters = np.unique(cluster_lookup['clusters'])
        # Get cluster indices (so it does not have to be called in every loop)
        cluster_idx = dict()
        for cluster in unq_clusters:
            tmp_idx = cluster_lookup['clusters'] == cluster
            rel_cells = cluster_lookup.index.values[tmp_idx]
            rel_idx = cell_lookup.loc[rel_cells, 'indices'].values
            cluster_idx[cluster] = rel_idx
        # Loop over genes
        for (_, selection, view) in ds.scan(items=row_idx,
                                            axis=0,
                                            layers=layers,
                                            batch_size=batch_size):
            # Get data in chunks
            tmp_dat = view.layers[layer].sparse(rows=np.arange(view.shape[0]),
                                                cols=np.arange(view.shape[1]))
            # Loop over all genes
            for i in np.arange(view.shape[0]):
                gene_list = list()
                pct_list = list()
                curr_gene = view.ra[feat_attr][i]
                rel_dat = tmp_dat.tocsr()[i, :].copy()
                # Loop over all clusters
                for cluster in unq_clusters:
                    rel_idx = cluster_idx[cluster]
                    gene_dat = rel_dat.tocsc()[:, rel_idx]
                    gene_list.append(np.ravel(gene_dat.todense()))
                    pct_list.append(gene_dat.data.shape[0] / gene_dat.shape[1])
                # Perform kruskal-wallis test
                try:  # for situations in which all numbers are identical
                    hval, pval = kruskal(*gene_list)
                except RuntimeWarning:
                    hval = np.nan
                    pval = np.nan
                result_dict[curr_gene] = [hval, pval, np.max(pct_list)]
        # Make data frame of results
        result_df = pd.DataFrame.from_dict(result_dict,
                                           orient='index',
                                           columns=['H', 'pval',
                                                    'max_cluster_pct'])
        # Make merged dataframe
        merged = pd.merge(gene_lookup,
                          result_df,
                          left_index=True,
                          right_index=True,
                          how='left')
        merged['H'] = merged['H'].fillna(value=0)
        merged['pval'] = merged['pval'].fillna(value=1)
        merged['max_cluster_pct'] = merged['max_cluster_pct'].fillna(0)
        merged['cluster_attr'] = cluster_attr
        # Add data to file
        ds.ra['kruskal_H'] = merged['H'].values
        ds.ra['kruskal_pval'] = merged['pval'].values
        ds.ra['kruskal_max_cluster_pct'] = merged['max_cluster_pct'].values
        ds.ra['kruskal_cluster_attr'] = merged['cluster_attr'].values


def get_decile_variable(loom_file,
                        layer,
                        out_attr=None,
                        id_attr='Accession',
                        percentile=30,
                        measure='vmr',
                        valid_ra=None,
                        valid_ca=None,
                        batch_size=512,
                        verbose=False):
    """
    Generates an attribute indicating the highest variable features per decile

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer containing relevant counts
        out_attr (str): Name of output attribute which will specify features
            Defaults to hvf_{n}
        id_attr (str): Attribute specifying unique feature IDs
        percentile (int): Percent of highly variable features per decile
        measure (str): Method of measuring variance
            vmr: variance mean ratio
            sd/std: standard deviation
            cv: coefficient of variation
        valid_ra (str): Optional, attribute to restrict features by
        valid_ca (str): Optional, attribute to restrict cells by
        batch_size (int): Size of chunks
            Will generate a dense array of batch_size by cells
        verbose (bool): Print logging messages
    """
    if verbose:
        feat_log.info(
            'Finding {0}% variable features per decile for {1}'.format(
                percentile, loom_file))
        t0 = time.time()
    # Get valid indices
    row_idx = utils.get_attr_index(loom_file=loom_file,
                                   attr=valid_ra,
                                   columns=False,
                                   as_bool=True,
                                   inverse=False)
    # Determine variability
    my_mean, my_std = utils.batch_mean_and_std(loom_file=loom_file,
                                               layer=layer,
                                               axis=1,
                                               valid_ca=valid_ca,
                                               valid_ra=valid_ra,
                                               batch_size=batch_size,
                                               verbose=verbose)
    # Find highly variable
    with loompy.connect(filename=loom_file, mode='r') as ds:
        feat_labels = ds.ra[id_attr][row_idx]
        feat_idx = pd.Series(np.zeros((ds.shape[0])),
                             index=ds.ra[id_attr])
    if measure.lower() == 'sd' or measure.lower() == 'std':
        tmp_var = pd.Series(my_std, index=feat_labels)
    elif measure.lower() == 'vmr':
        my_var = my_std ** 2
        my_vmr = (my_var + 1) / (my_mean + 1)
        tmp_var = pd.Series(my_vmr, index=feat_labels)
    elif measure.lower() == 'cv':
        my_cv = (my_std + 1) / (my_mean + 1)
        tmp_var = pd.Series(my_cv, index=feat_labels)
    else:
        err_msg = 'Unsupported measure value ({})'.format(measure)
        if verbose:
            feat_log.error(err_msg)
        raise ValueError(err_msg)
    # Get variable
    my_mean = pd.Series(my_mean, index=feat_labels)
    feat_deciles = pd.qcut(my_mean,
                           q=10,
                           labels=False).to_frame('decile')
    hvf = []
    for decile, batch_info in feat_deciles.groupby('decile'):
        gene_group = batch_info.index.values
        var_gg = tmp_var.loc[gene_group]
        # Genes per decile
        hvf_group = gene_group[var_gg > np.percentile(var_gg, 100 - percentile)]
        if decile != 9:  # Ignore final decile
            hvf.append(hvf_group)
    # Add to loom file
    hvf = np.hstack(hvf)
    feat_idx.loc[hvf] = 1
    if out_attr is None:
        out_attr = 'hvf_decile_{}'.format(percentile)
    with loompy.connect(loom_file) as ds:
        ds.ra[out_attr] = feat_idx.values.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        feat_log.info(
            'Found {0} variable features in {1:.2f} {2}'.format(hvf.shape[0],
                                                                time_run,
                                                                time_fmt))


def get_n_variable_features(loom_file,
                            layer,
                            out_attr=None,
                            id_attr='Accession',
                            n_feat=4000,
                            measure='vmr',
                            valid_ra=None,
                            valid_ca=None,
                            batch_size=512,
                            verbose=False):
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
            cv: coefficient of variation
        valid_ra (str): Optional, attribute to restrict features by
        valid_ca (str): Optional, attribute to restrict cells by
        batch_size (int): Size of chunks
            Will generate a dense array of batch_size by cells
        verbose (bool): Print logging messages
    """
    if verbose:
        feat_log.info(
            'Finding {} variable features for {}'.format(n_feat, loom_file))
        t0 = time.time()
    # Get valid indices
    row_idx = utils.get_attr_index(loom_file=loom_file,
                                   attr=valid_ra,
                                   columns=False,
                                   as_bool=True,
                                   inverse=False)
    # Determine variability
    my_mean, my_std = utils.batch_mean_and_std(loom_file=loom_file,
                                               layer=layer,
                                               axis=1,
                                               valid_ca=valid_ca,
                                               valid_ra=valid_ra,
                                               batch_size=batch_size,
                                               verbose=verbose)
    # Find highly variable
    with loompy.connect(filename=loom_file, mode='r') as ds:
        feat_labels = ds.ra[id_attr][row_idx]
        feat_idx = pd.Series(np.zeros((ds.shape[0])),
                             index=ds.ra[id_attr])
    if measure.lower() == 'sd' or measure.lower() == 'std':
        tmp_var = pd.Series(my_std, index=feat_labels)
    elif measure.lower() == 'vmr':
        my_var = my_std ** 2
        my_vmr = (my_var + 1) / (my_mean + 1)
        tmp_var = pd.Series(my_vmr, index=feat_labels)
    elif measure.lower() == 'cv':
        my_cv = (my_std + 1) / (my_mean + 1)
        tmp_var = pd.Series(my_cv, index=feat_labels)
    else:
        err_msg = 'Unsupported measure value ({})'.format(measure)
        if verbose:
            feat_log.error(err_msg)
        raise ValueError(err_msg)
    # Get top n variable features
    n_feat = min(n_feat, tmp_var.shape[0])
    hvf = tmp_var.sort_values(ascending=False).head(n_feat).index.values
    feat_idx.loc[hvf] = 1
    if out_attr is None:
        out_attr = 'hvf_nfeat_{}'.format(n_feat)
    with loompy.connect(loom_file) as ds:
        ds.ra[out_attr] = feat_idx.values.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        feat_log.info(
            'Found {0} variable features in {1:.2f} {2}'.format(hvf.shape[0],
                                                                time_run,
                                                                time_fmt))


def prep_for_common(loom_file,
                    id_attr='Accession',
                    valid_ra=None,
                    remove_version=False):
    """
    Generates objects for find_common_features

    Args:
        loom_file (str): Path to loom file
        id_attr (str): Attribute specifying unique feature IDs
        remove_version (bool): Remove GENCODE gene versions from IDs
        valid_ra (str): Optional, attribute that specifies desired features

    Returns:
        features (ndarray): Array of unique feature IDs
    """
    valid_idx = utils.get_attr_index(loom_file=loom_file,
                                     attr=valid_ra,
                                     columns=False,
                                     as_bool=True,
                                     inverse=False)
    with loompy.connect(filename=loom_file, mode='r') as ds:
        features = ds.ra[id_attr][valid_idx]
        if remove_version:
            features = utils.remove_gene_version(gene_ids=features)
    return features


def add_common_features(loom_file,
                        id_attr,
                        common_features,
                        out_attr,
                        remove_version=False):
    """
    Adds index of common features to loom file (run with find_common_features)

    Args:
        loom_file (str): Path to loom file
        id_attr (str): Name of attribute specifying unique feature IDs
        common_features (ndarray): Array of common features
        out_attr (str): Name of output attribute specifying common features

        remove_version (bool): If true remove version ID
            Anything after the first period is dropped
            Useful for GENCODE gene IDs
    """
    # Make logical index of desired features
    feat_ids = prep_for_common(loom_file=loom_file,
                               id_attr=id_attr,
                               remove_version=remove_version,
                               valid_ra=None)
    with loompy.connect(filename=loom_file) as ds:
        logical_idx = pd.Series(data=np.zeros((ds.shape[0],),
                                              dtype=int),
                                index=feat_ids,
                                dtype=int)
        logical_idx.loc[common_features] = 1
        ds.ra[out_attr] = logical_idx.values


def find_common_features(loom_files,
                         feature_attrs,
                         out_attr,
                         valid_ras=None,
                         remove_version=False,
                         verbose=False):
    """
    Identifies common features among multiple loom files

    Args:
        loom_files (list): List of loom files that will be integrated together
        feature_attrs (list): List of row attributes with unique feature IDs
            Values must be the same across all loom_files
                GENCODE version IDs can be removed with remove_version
            Order must be the same as loom_files
        out_attr (str): Name of output attribute indicating common IDs
            Will be a boolean array indicating valid, common features in each loom file
        valid_ras (list): Optional, attributes that specifies desired features for comparison across all files
        remove_version (bool/list): If true remove version number
            Anything after the first period is dropped (useful for GENCODE IDs)
            If a list, will behave differently for each loom file
            If a boolean, will behave the same for each loom_file
        verbose (bool): If true, print logging messages
    """
    if verbose:
        feat_log.info('Finding common features')
    # Check inputs
    was_an_error = False
    if isinstance(loom_files, list) and isinstance(feature_attrs, list) and len(loom_files) == len(feature_attrs):
        pass
    else:
        was_an_error = True
    if valid_ras is None:
        valid_ras = [None] * len(loom_files)
    elif isinstance(valid_ras, list) and len(valid_ras) == len(loom_files):
        pass
    else:
        was_an_error = True
    if isinstance(remove_version, bool):
        remove_version = [remove_version] * len(loom_files)
    elif isinstance(remove_version, list) and len(remove_version) == len(loom_files):
        pass
    else:
        was_an_error = True
    if was_an_error:
        err_msg = 'Inputs must be lists of the same length'
        if verbose:
            feat_log.error(err_msg)
        raise ValueError(err_msg)
    # Get features
    all_features = []
    for i in np.arange(len(loom_files)):
        # Get current items
        curr_loom = loom_files[i]
        curr_feat = feature_attrs[i]
        curr_remove = remove_version[i]
        curr_valid = valid_ras[i]
        # Get features
        curr_items = prep_for_common(loom_file=curr_loom,
                                     id_attr=curr_feat,
                                     valid_ra=curr_valid,
                                     remove_version=curr_remove)
        all_features.append(curr_items)
    # Find common features
    common_feat = functools.reduce(np.intersect1d, all_features)
    num_comm = common_feat.shape[0]
    if num_comm == 0:
        err_msg = 'Could not identify any common features'
        if verbose:
            feat_log.error(err_msg)
        raise RuntimeError(err_msg)
    if verbose:
        feat_log.info('Found {} features in common'.format(num_comm))
    # Add indices
    for i in np.arange(len(loom_files)):
        # Get current items
        curr_loom = loom_files[i]
        curr_feat = feature_attrs[i]
        curr_remove = remove_version[i]
        # Add items
        add_common_features(loom_file=curr_loom,
                            id_attr=curr_feat,
                            common_features=common_feat,
                            out_attr=out_attr,
                            remove_version=curr_remove)
        if verbose:
            log_msg = '{0:.2f}% of features in {1} were in common'.format(utils.get_pct(loom_file=curr_loom,
                                                                                        num_val=num_comm,
                                                                                        axis=0),
                                                                          curr_loom)
            feat_log.info(log_msg)
