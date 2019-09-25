


def get_knn_for_mnn(loom_x,
                    layer_x,
                    neighbor_distance_x,
                    neighbor_index_x,
                    max_k_x,
                    loom_y,
                    layer_y,
                    neighbor_distance_y,
                    neighbor_index_y,
                    max_k_y,
                    direction,
                    feature_id_x,
                    feature_id_y,
                    valid_ca_x=None,
                    valid_ra_x=None,
                    valid_ca_y=None,
                    valid_ra_y=None,
                    n_trees=10,
                    seed=None,
                    batch_x=512,
                    batch_y=512,
                    remove_version=False,
                    verbose=False):
    """
    Gets kNN distances and indices by iterating over a loom file

    Args:
        loom_x (str): Path to loom file
        layer_x (str): Layer containing data for loom_x
        neighbor_distance_x (str): Output attribute for distances
        neighbor_index_x (str): Output attribute for indices
        max_k_x (int): Maximum number of nearest neighbors for x
        loom_y (str): Path to loom file
        layer_y (str): Layer containing data for loom_y
        neighbor_distance_y (str): Output attribute for distances
        neighbor_index_y (str): Output attribute for indices
        max_k_y  (int): Maximum number of nearest neighbors for y
        direction (str): Expected direction of relationship between x and y
            positive or +
            negative or -
        feature_id_x (str): Row attribute containing unique feature IDs
        feature_id_y (str): Row attribute containing unique feature IDs
        valid_ca_x (str): Column attribute specifying valid cells
        valid_ra_x (str): Row attribute specifying valid features
        valid_ca_y (str): Column attribute specifying valid cells
        valid_ra_y (str): Row attribute specifying valid features
        n_trees (int): Number of trees to use for kNN
            more trees = more precision
        seed (int): Seed for Annoy
        batch_x (int): Size of chunks for iterating over loom_x
        batch_y (int): Size of chunks for iterating over loom_y
        remove_version (bool): Remove GENCODE version IDs
        verbose (bool): Print logging messages
    """
    # Prep for function
    if verbose:
        help_log.info('Finding kNN distances and indices')
        t0 = time.time()
    # Prep for kNN
    col_x = utils.get_attr_index(loom_file=loom_x,
                                 attr=valid_ca_x,
                                 columns=True,
                                 as_bool=True,
                                 inverse=False)
    row_x = utils.get_attr_index(loom_file=loom_x,
                                 attr=valid_ra_x,
                                 columns=False,
                                 as_bool=True,
                                 inverse=False)
    col_y = utils.get_attr_index(loom_file=loom_y,
                                 attr=valid_ca_y,
                                 columns=True,
                                 as_bool=True,
                                 inverse=False)
    row_y = utils.get_attr_index(loom_file=loom_y,
                                 attr=valid_ra_y,
                                 columns=False,
                                 as_bool=True,
                                 inverse=False)
    # Make lookup
    lookup_x = pd.Series(np.where(col_x)[0],
                         index=np.arange(np.sum(col_x)))
    lookup_y = pd.Series(np.where(col_y)[0],
                         index=np.arange(np.sum(col_y)))

    # Get features
    with loompy.connect(filename=loom_x) as ds_x:
        x_feat = ds_x.ra[feature_id_x][row_x]
    with loompy.connect(filename=loom_y) as ds_y:
        y_feat = ds_y.ra[feature_id_y][row_y]
    if remove_version:
        x_feat = utils.remove_gene_version(x_feat)
        y_feat = utils.remove_gene_version(y_feat)
    if np.any(np.sort(x_feat) != np.sort(y_feat)):
        raise ValueError('Feature mismatch!')
    reverse_y = False
    if direction == '+' or direction == 'positive':
        reverse_x = False
    elif direction == '-' or direction == 'negative':
        reverse_x = True
    else:
        raise ValueError('Unsupported direction value')
    # Make temporary files holding zscores
    tmp_x = temp_zscore_loom(loom_file=loom_x,
                             raw_layer=layer_x,
                             feat_attr=feature_id_x,
                             valid_ca=valid_ca_x,
                             valid_ra=valid_ra_x,
                             batch_size=batch_x,
                             tmp_dir=None,
                             verbose=verbose)
    tmp_y = temp_zscore_loom(loom_file=loom_y,
                             raw_layer=layer_y,
                             feat_attr=feature_id_y,
                             valid_ca=valid_ca_y,
                             valid_ra=valid_ra_y,
                             batch_size=batch_y,
                             tmp_dir=None,
                             verbose=verbose)
    # Train kNN
    t_y2x = train_knn(loom_file=tmp_x,
                      layer='',
                      row_arr=row_x,
                      col_arr=col_x,
                      feat_attr=feature_id_x,
                      feat_select=x_feat,
                      reverse_rank=reverse_x,
                      remove_version=remove_version,
                      seed=seed,
                      batch_size=batch_x,
                      verbose=verbose)
    t_x2y = train_knn(loom_file=tmp_y,
                      layer='',
                      row_arr=row_y,
                      col_arr=col_y,
                      feat_attr=feature_id_y,
                      feat_select=x_feat,
                      reverse_rank=reverse_y,
                      remove_version=remove_version,
                      seed=seed,
                      batch_size=batch_y,
                      verbose=verbose)
    # Build trees
    t_x2y = build_knn(t=t_x2y,
                      n_trees=n_trees,
                      verbose=verbose)
    t_y2x = build_knn(t=t_y2x,
                      n_trees=n_trees,
                      verbose=verbose)
    # Get distances and indices
    dist_x, idx_x = report_knn(loom_file=tmp_x,
                               layer='',
                               row_arr=row_x,
                               col_arr=col_x,
                               feat_attr=feature_id_x,
                               feat_select=x_feat,
                               reverse_rank=reverse_x,
                               k=max_k_x,
                               t=t_x2y,
                               batch_size=batch_x,
                               remove_version=remove_version,
                               verbose=verbose)
    dist_y, idx_y = report_knn(loom_file=tmp_y,
                               layer='',
                               row_arr=row_y,
                               col_arr=col_y,
                               feat_attr=feature_id_y,
                               feat_select=x_feat,
                               reverse_rank=reverse_y,
                               k=max_k_y,
                               t=t_y2x,
                               batch_size=batch_y,
                               remove_version=remove_version,
                               verbose=verbose)
    # Get correct indices (import if restricted to valid cells)
    correct_idx_x = np.reshape(lookup_y.loc[np.ravel(idx_x).astype(int)].values,
                               idx_x.shape)
    correct_idx_y = np.reshape(lookup_x.loc[np.ravel(idx_y).astype(int)].values,
                               idx_y.shape)
    # Add data to files
    with loompy.connect(filename=loom_x) as ds:
        ds.ca[neighbor_distance_x] = dist_x
        ds.ca[neighbor_index_x] = correct_idx_x
    with loompy.connect(filename=loom_y) as ds:
        ds.ca[neighbor_distance_y] = dist_y
        ds.ca[neighbor_index_y] = correct_idx_y
    # Remove temporary files
    os.remove(tmp_x)
    os.remove(tmp_y)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = utils.format_run_time(t0, t1)
        help_log.info(
            'Found neighbors in {0:.2f} {1}'.format(time_run, time_fmt))








