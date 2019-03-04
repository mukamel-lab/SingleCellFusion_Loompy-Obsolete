"""
Collection of functions called by recipes in recipes.py

Written by Wayne Doyle unless noted

(C) 2019 Mukamel Lab GPLv2
"""

import loompy
from . import clustering
from . import decomposition
from . import qc
from . import counts

def qc_cells_and_features(loom_file,
                          layer,
                          valid_ca,
                          valid_ra,
                          min_feature_count,
                          feature_fraction,
                          min_cell_count,
                          cell_fraction,
                          batch_size,
                          verbose):
    qc.label_covered_cells(loom_file = loom_file,
                               layer = layer,
                               out_attr = valid_ca,
                               min_count = min_cell_count,
                               fraction_covered = cell_fraction,
                               col_attr = None,
                               row_attr = None,
                               batch_size = batch_size,
                               verbose = verbose)
    qc.label_covered_features(loom_file = loom_file,
                              layer = layer,
                              out_attr = valid_ra,
                              min_count = min_feature_count,
                              fraction_covered = feature_fraction,
                              col_attr = valid_ca,
                              row_attr = None,
                              batch_size = batch_size,
                              verbose = verbose)
                              
def louvain_tsne_umap(loom_file,
                      clust_attr,
                      id_attr,
                      valid_ca,
                      valid_ra,
                      pca_attr,
                      pca_layer,
                      n_pca,
                      drop_first,
                      scale_attr,
                      neighbor_attr,
                      distance_attr,
                      cluster_k,
                      jaccard_graph,
                      umap_attr,
                      umap_dist,
                      umap_k,
                      tsne_attr,
                      tsne_perp,
                      n_proc,
                      batch_size,
                      seed,
                      verbose):
    clustering.louvain_jaccard(loom_file = loom_file,
                                   clust_attr = clust_attr,
                                   cell_attr = id_attr,
                                   valid_attr = valid_ca,
                                   gen_pca = True,
                                   pca_attr = pca_attr,
                                   layer = pca_layer,
                                   n_pca = n_pca,
                                   drop_first=drop_first,
                                   row_attr = valid_ra,
                                   scale_attr = scale_attr,
                                   gen_knn = True,
                                   neighbor_attr = neighbor_attr,
                                   distance_attr = distance_attr,
                                   k = cluster_k,
                                   num_trees = 50,
                                   metric = 'euclidean',
                                   gen_jaccard = True,
                                   jaccard_graph = jaccard_graph,
                                   batch_size = batch_size,
                                   seed = seed,
                                   verbose = verbose)
    decomposition.run_umap(loom_file = loom_file,
                           cell_attr = id_attr,
                           out_attr=umap_attr,
                           valid_attr=valid_ca,
                           gen_pca=False,
                           pca_attr=pca_attr,
                           n_umap=2,
                           min_dist=umap_dist,
                           n_neighbors=umap_k,
                           metric='euclidean',
                           batch_size=batch_size,
                           verbose=verbose)
    decomposition.run_tsne(loom_file = loom_file,
                           cell_attr = id_attr,
                           out_attr=tsne_attr,
                           valid_attr=valid_ca,
                           gen_pca=False,
                           pca_attr=pca_attr,
                           perp=tsne_perp,
                           n_tsne=2,
                           n_proc=n_proc,
                           n_iter=1000,
                           batch_size=batch_size,
                           verbose=verbose)
    
def normalize_and_log_10x(loom_file,
                          in_layer,
                          norm_layer,
                          log_layer,
                          size_attr,
                          valid_ca,
                          valid_ra,
                          batch_size,
                          verbose):
    counts.normalize_10x(loom_file = loom_file,
                             in_layer = in_layer,
                             out_layer = norm_layer,
                             size_attr = size_attr,
                             gen_size = True,
                             col_attr = valid_ca,
                             row_attr = valid_ra,
                             batch_size = batch_size,
                             verbose = verbose)
    counts.log_transform_counts(loom_file = loom_file,
                                    in_layer = norm_layer,
                                    out_layer = log_layer,
                                    log_type = 'log10',
                                    verbose = verbose)

def normalize_and_log_atac_gene(loom_file,
                          in_layer,
                          norm_layer,
                          log_layer,
                          length_attr,
                          method,
                          valid_ca,
                          valid_ra,
                          batch_size,
                          verbose):
    counts.normalize_counts(loom_file = loom_file,
            method = method,
                             in_layer = in_layer,
                             out_layer = norm_layer,
                             length_attr = length_attr,
                             col_attr = valid_ca,
                             row_attr = valid_ra,
                             batch_size = batch_size,
                             verbose = verbose)
    counts.log_transform_counts(loom_file = loom_file,
                                    in_layer = norm_layer,
                                    out_layer = log_layer,
                                    log_type = 'log10',
                                    verbose = verbose)