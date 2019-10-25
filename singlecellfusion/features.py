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
import warnings
import functools
from . import utils

feat_log = logging.getLogger(__name__)


def low_mem_kruskal(loom_file,
                    layer='',
                    cluster_attr='ClusterID',
                    cell_attr='CellID',
                    feat_attr='Accession',
                    valid_ca=None,
                    valid_ra=None,
                    batch_size=512):
    """
    Performs a Kruskal-Wallis test per cluster for all valid genes in a slow, low memory fashion

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


def high_mem_kruskal(loom_file,
                     layer='',
                     cluster_attr='ClusterID',
                     cell_attr='CellID',
                     feat_attr='Accession',
                     valid_ca=None,
                     valid_ra=None):
    """
    Performs a Kruskal-Wallis test per cluster for all valid genes in a fast, high memory fashion

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer of counts
        cluster_attr (str): Column attribute containing cluster IDs
        cell_attr (str): Column attribute containing unique cell IDs
        feat_attr (str): Row attribute containing unique feature IDs
        valid_ra (str): Row attribute specifying valid features
        valid_ca (str): Column attribute specifying valid cells
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
    result_dict = dict()
    # Perform Kruskal-Wallis test
    with loompy.connect(loom_file, mode='r') as ds:
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
        # Get data
        tmp_dat = ds.layers[layer].sparse(rows=row_idx,
                                          cols=col_idx)
        # Loop over all genes
        for i in np.arange(tmp_dat.shape[0]):
            gene_list = list()
            pct_list = list()
            curr_gene = gene_lookup.index.values[i]
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


def low_mem_decile(loom_file,
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
    Generates an attribute indicating the highest variable features per decile in a slow, low memory fashion

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
    my_mean, my_std = utils.low_mem_mean_and_std(loom_file=loom_file,
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
    else:
        raise ValueError('Unsupported measure value ({})'.format(measure))
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


def high_mem_decile(loom_file,
                    layer,
                    out_attr=None,
                    id_attr='Accession',
                    percentile=30,
                    measure='vmr',
                    valid_ra=None,
                    valid_ca=None,
                    verbose=False):
    """
    Generates an attribute indicating the highest variable features per decile in a fast, high memory fashion

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
                                   as_bool=False,
                                   inverse=False)
    # Get mean and standard deviation
    my_mean, my_std = utils.high_mem_mean_and_std(loom_file=loom_file,
                                                  layer=layer,
                                                  axis=1,
                                                  valid_ca=valid_ca,
                                                  valid_ra=valid_ra)
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
    else:
        raise ValueError('Unsupported measure value ({})'.format(measure))
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
        valid_ra (str/None): Optional, attribute that specifies desired features

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


def get_kruskal_common(loom_files,
                       out_attr,
                       feat_attrs='Accession',
                       valid_ras=None,
                       n_markers=8000,
                       remove_version=False,
                       verbose=False):
    """
    Finds common features using Kruskal-Wallis test results

    Args:
        loom_files (list): Path to loom files
        out_attr (str): Output row attribute specifying valid features
        feat_attrs(str/list): Row attribute containing unique feature IDs
        valid_ras (str/list/None): Row attribute specifying valid rows
        n_markers (int): Find the intersection of n_markers genes from each dataset
        remove_version (bool): If true, remove GENCODE version number
        verbose (bool): If true, print logging messages
    """
    if isinstance(loom_files, list):
        check_parameters = [feat_attrs,
                            valid_ras]
        checked = utils.mimic_list(parameters=check_parameters,
                                   list_len=len(loom_files))
        feat_attrs = checked[0]
        valid_ras = checked[1]
    elif isinstance(loom_files, str):
        raise ValueError('More than one loom file has to be provided')
    if verbose:
        feat_log.info('Finding common features')
    # Check inputs
    sample_num = []
    for i in np.arange(len(loom_files)):
        valid_idx = utils.get_attr_index(loom_file=loom_files[i],
                                         attr=valid_ras[i],
                                         columns=False,
                                         as_bool=False,
                                         inverse=False)
        sample_num.append(valid_idx.shape[0])
    min_num = np.min(sample_num)
    if min_num < n_markers:
        warnings.warn('Less valid genes than markers, setting to number of valid genes ({})'.format(min_num))
        n_markers = min_num
    # Get features
    marker_list = []
    for i in np.arange(len(loom_files)):
        valid_idx = utils.get_attr_index(loom_file=loom_files[i],
                                         attr=valid_ras[i],
                                         columns=False,
                                         as_bool=False,
                                         inverse=False)
        with loompy.connect(loom_files[i]) as ds:
            result_df = pd.DataFrame({'H': ds.ra['kruskal_H'],
                                      'pval': ds.ra['kruskal_pval'],
                                      'pct': ds.ra['kruskal_max_cluster_pct']},
                                     index=ds.ra[feat_attrs[i]])
            if remove_version:
                result_df.index = utils.remove_gene_version(result_df.index.values)
            result_df = result_df.iloc[valid_idx, :]
            result_df = result_df.sort_values(by='H', ascending=False).head(n=n_markers)
            marker_list.append(result_df.index.values)
    common_feat = functools.reduce(np.intersect1d, marker_list)
    num_comm = common_feat.shape[0]
    if num_comm == 0:
        RuntimeError('Could not identify any common features')
    if verbose:
        feat_log.info('Found {} features in common'.format(num_comm))
    # Add indices
    for i in np.arange(len(loom_files)):
        feat_ids = prep_for_common(loom_file=loom_files[i],
                                   id_attr=feat_attrs[i],
                                   remove_version=remove_version,
                                   valid_ra=valid_ras[i])
        with loompy.connect(filename=loom_files[i]) as ds:
            logical_idx = pd.Series(data=np.zeros((ds.shape[0],),
                                                  dtype=int),
                                    index=feat_ids,
                                    dtype=int)
            logical_idx.loc[common_feat] = 1
            ds.ra[out_attr] = logical_idx.values
        if verbose:
            log_msg = '{0:.2f}% of features in {1} were in common'.format(utils.get_pct(loom_file=loom_files[i],
                                                                                        num_val=num_comm,
                                                                                        axis=0),
                                                                          loom_files[i])
            feat_log.info(log_msg)


def get_decile_common(loom_files,
                      feat_attrs,
                      out_attr,
                      valid_ras=None,
                      remove_version=False,
                      verbose=False):
    """
    Identifies common features among multiple loom files by intersecting feature names

    Args:
        loom_files (list): List of loom
        feat_attrs (list/str): List of row attributes with unique feature IDs
            Values must be the same across all loom_files
        out_attr (str): Name of output attribute indicating common IDs
            Will be a boolean array indicating valid, common features in each loom file
        valid_ras (list/str/None): Optional, attributes that specifies desired features for comparison across all files
        remove_version (bool): If true, remove GENCODE version ID
        verbose (bool): If true, print logging messages
    """
    if isinstance(loom_files, list):
        check_parameters = [feat_attrs,
                            valid_ras]
        checked = utils.mimic_list(parameters=check_parameters,
                                   list_len=len(loom_files))
        feat_attrs = checked[0]
        valid_ras = checked[1]
    elif isinstance(loom_files, str):
        raise ValueError('More than one loom file has to be provided')
    if verbose:
        feat_log.info('Finding common features')
    # Get features
    marker_list = []
    for i in np.arange(len(loom_files)):
        # Get features
        marker_list.append(prep_for_common(loom_files[i],
                                           id_attr=feat_attrs[i],
                                           valid_ra=valid_ras[i],
                                           remove_version=remove_version))
    # Find common features
    common_feat = functools.reduce(np.intersect1d, marker_list)
    num_comm = common_feat.shape[0]
    if num_comm == 0:
        RuntimeError('Could not identify any common features')
    if verbose:
        feat_log.info('Found {} features in common'.format(num_comm))
    # Add indices
    for i in np.arange(len(loom_files)):
        feat_ids = prep_for_common(loom_file=loom_files[i],
                                   id_attr=feat_attrs[i],
                                   remove_version=remove_version,
                                   valid_ra=valid_ras[i])
        with loompy.connect(filename=loom_files[i]) as ds:
            logical_idx = pd.Series(data=np.zeros((ds.shape[0],),
                                                  dtype=int),
                                    index=feat_ids,
                                    dtype=int)
            logical_idx.loc[common_feat] = 1
            ds.ra[out_attr] = logical_idx.values
        if verbose:
            log_msg = '{0:.2f}% of features in {1} were in common'.format(utils.get_pct(loom_file=loom_files[i],
                                                                                        num_val=num_comm,
                                                                                        axis=0),
                                                                          loom_files[i])
            feat_log.info(log_msg)


def find_common_variable(loom_files,
                         layers='',
                         method='kruskal',
                         cell_attrs='CellID',
                         feat_attrs='Accession',
                         cluster_attrs='ClusterID',
                         common_attr='CommonVariable',
                         variable_attr='VariableFeatures',
                         valid_ras=None,
                         valid_cas=None,
                         kruskal_n=8000,
                         percentile=30,
                         low_mem=False,
                         batch_size=5000,
                         remove_version=False,
                         verbose=False):
    """
    Finds variable features that are in common across all datasets

    Args:
        loom_files (list): Loom files to compare
        layers (str/list): Layer(s) in loom files containing count data
            If a string, assumes the same layer name is used in all loom files
        method (str): Method for finding variable genes
            vmr: variance mean ratio (found in deciles)
            sd: standard deviation (found in deciles)
            kruskal: Kruskal-Wallis test (recommended)
        cell_attrs (str/list): Column attribute(s) in loom_files containing unique cell IDs
        feat_attrs (str/list): Row attribute(s) in loom_files containing unique feature IDs
        cluster_attrs (str/list/None): Column attribute(s) specifying cluster assignments for each cell
            Necessary if the kruskal method is used
        common_attr (str): Output row attribute which will specify valid common features
        variable_attr (str/None): Output row attribute which will specify valid, variable features
            Only used if method is sd or vmr
            If method is kruskal the following row attributes will automatically be added
                 kruskal_H: H statistic from Kruskal-Wallis test
                 kruskal_pval: p-value from Kruskal-Wallis test
                 kruskal_max_cluster_pct: Percentage of cells with non-zero counts in clusters
                    Percent is from the cluster that has the largest number of non-zero counts
                 kruskal_cluster_attr: Cluster attribute used for performing Kruskal-Wallis test
        valid_ras (str/list/None): Row attribute specifying valid features to include
        valid_cas (str/list/None): Column attribute specifying valid cells to include
        kruskal_n (int/None): Use the top kruskal_n number of genes from each loom file to find common features
            Only used if method is kruskal
        percentile (int): Whole number (0-100) percent of variable genes to select per decile of expression
            Only used if method is sd or vmr
        low_mem (bool): If true, perform low memory search for variable and common features
        batch_size (int): Size of chunks for batches if low_mem is True
        remove_version (bool): If true, remove GENCODE version IDs
        verbose (bool): If true, print logging messages
    """
    # Check inputs
    if isinstance(loom_files, list):
        check_parameters = [layers,
                            cell_attrs,
                            feat_attrs,
                            valid_ras,
                            valid_cas,
                            cluster_attrs]
        checked = utils.mimic_list(parameters=check_parameters,
                                   list_len=len(loom_files))
        layers = checked[0]
        cell_attrs = checked[1]
        feat_attrs = checked[2]
        valid_ras = checked[3]
        valid_cas = checked[4]
        cluster_attrs = checked[5]
    elif isinstance(loom_files, str):
        raise ValueError('More than one loom file has to be provided')
    # Find variable genes
    for i in np.arange(len(loom_files)):
        if verbose:
            feat_log.info('Finding variable features for {}'.format(loom_files[i]))
        if method == 'vmr' or method == 'sd':
            if low_mem:
                low_mem_decile(loom_file=loom_files[i],
                               layer=layers[i],
                               out_attr=variable_attr,
                               id_attr=feat_attrs[i],
                               percentile=percentile,
                               measure=method,
                               valid_ra=valid_ras[i],
                               valid_ca=valid_cas[i],
                               batch_size=batch_size,
                               verbose=verbose)
        elif method == 'kruskal':
            if low_mem:
                low_mem_kruskal(loom_file=loom_files[i],
                                layer=layers[i],
                                cluster_attr=cluster_attrs[i],
                                cell_attr=cell_attrs[i],
                                feat_attr=feat_attrs[i],
                                valid_ca=valid_cas[i],
                                valid_ra=valid_ras[i],
                                batch_size=batch_size)
            else:
                high_mem_kruskal(loom_file=loom_files[i],
                                 layer=layers[i],
                                 cluster_attr=cluster_attrs[i],
                                 cell_attr=cell_attrs[i],
                                 feat_attr=feat_attrs[i],
                                 valid_ca=valid_cas[i],
                                 valid_ra=valid_ras[i])
        else:
            raise ValueError('Unsupported value for method ({})'.format(method))
    # Determine common genes
    if method == 'vmr' or method == 'sd':
        get_decile_common(loom_files=loom_files,
                          feat_attrs=feat_attrs,
                          out_attr=common_attr,
                          valid_ras=variable_attr,
                          remove_version=remove_version,
                          verbose=verbose)
    elif method == 'kruskal':
        get_kruskal_common(loom_files=loom_files,
                           feat_attrs=feat_attrs,
                           out_attr=common_attr,
                           valid_ras=variable_attr,
                           n_markers=kruskal_n,
                           remove_version=remove_version,
                           verbose=verbose)
    else:
        raise ValueError('Unsupported method value ({})'.format(method))
