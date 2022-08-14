import re
import logging
from collections import Counter
from itertools import product, combinations, chain
from typing import Dict, List, Tuple
from joblib import Parallel, delayed

import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from toolz import pipe
from umap import UMAP


clustering_params = [
    [KMeans, ["n_clusters", range(20, 50, 3)]],
    # [AffinityPropagation, ["damping", np.arange(0.5, 0.91, 0.1), "max_iter", [1000]]],
    [GaussianMixture, ["n_components", range(20, 50, 5)]],
]
search_params = product(range(5, 55, 5), np.arange(0.4, 1.1, 0.1))


def extract_clusters(
    feature_df: pd.DataFrame,
    pca: int,
    n_runs: int,
    comm_resolution: float,
    clustering_options: dict,
    cluster_funct,
) -> Tuple[pd.DataFrame, Dict]:
    """Function to extract cluster lookups and positions

    Args:
        feature_df: table with the observations we want to cluster and
            the features we want to use. NB observations ids in the index
        pca: number of components to use in PCA
        comm_resolution: resolution for community detection
        clustering_options: dictionary of clustering options

    Returns:
        A tuple containing a dataframe with the 2-d UMAP projection of the clusters
        and a dictionary looking up observation ids vs clusters
    """
    print("Reducing features")
    features_reduced = reduce_dim(feature_df, n_components_pca=pca)

    clustering, indices = cluster_funct(features_reduced, clustering_options, n_runs)

    cluster_lookup = extract_communities(clustering, comm_resolution, indices)

    umap_df = features_reduced.assign(
        cluster=lambda df: df.index.map(cluster_lookup)
    ).reset_index(drop=False)

    return umap_df, cluster_lookup


def reduce_dim(
    feature_df: pd.DataFrame, n_components_pca: int = 50, n_components_umap: int = 2
) -> pd.DataFrame:
    """Reduce dimensionality of feature df first via PCA and then via UMAP

    Args:
        feature_df: table with observations and features
        n_components_pca: number of components to use in PCA
        n_components_umap: number of components to use in UMAP

    Returns:
        A 2-d df with the UMAP projection of the observations
    """

    pca = PCA(n_components=n_components_pca)

    return pipe(
        feature_df,
        lambda df: pd.DataFrame(pca.fit_transform(feature_df), index=feature_df.index),
        lambda df: pd.DataFrame(
            UMAP(n_components=n_components_umap).fit_transform(df),
            index=df.index,
            columns=["x", "y"],
        ),
    )


def make_edges_from_cluster(
    features: pd.DataFrame, clustering_algorithm, parameters: Dict
) -> List:
    """
    Extract edges from clusters

    Args:
        features: Observations and their features
        clustering algorith: algorithm to run the cluster on
        parameters: parameters to use

    Returns:
        list of pairs of observations that are assigned to the
        same cluster
    """

    cl_assignments = clustering_algorithm(**parameters).fit_predict(features)
    index_cluster_pair = [(n, c) for n, c in enumerate(cl_assignments)]

    cluster_edges = chain(
        *chain(
            *[
                [
                    # This extracts pairs of observations ("edges")
                    # from each cluster of observations
                    combinations(
                        [assgn[0] for assgn in index_cluster_pair if assgn[1] == c],
                        2,
                    )
                ]
                for c in set(cl_assignments)
            ]
        )
    )

    return [frozenset(x) for x in cluster_edges]


def build_cluster_graph(
    vectors: pd.DataFrame,
    clustering_algorithms: list,
    n_runs: int = 10,
    sample: int = None,
) -> Tuple[nx.Graph, dict]:
    """Builds a cluster network based on observation co-occurrences
    in a clustering output
    Args:
        vectors: vectors to cluster
        clustering_algorithms: a list where the first element is the clustering
            algorithm and the second element are the parameter names and sets
        n_runs: number of times to run a clustering algorithm
        sample: size of the vector to sample.
    Returns:
        A network where the nodes are observations and their edges number
            of co-occurrences in the clustering
    #FIXUP: remove the nested loops
    """

    clustering_edges = []

    index_to_id_lookup = {n: ind for n, ind in enumerate(vectors.index)}

    logging.info("Running cluster ensemble")
    for cl in clustering_algorithms:

        logging.info(cl[0])

        algo = cl[0]

        parametres = [{cl[1][0]: v} for v in cl[1][1]]

        for par in parametres:

            logging.info(f"running {par}")

            edges_from_cluster = Parallel(n_jobs=5)(
                delayed(make_edges_from_cluster)(vectors, algo, par)
                for _ in range(n_runs)
            )

            clustering_edges.append(chain(*edges_from_cluster))

    edges_weighted = Counter(list(chain(*clustering_edges)))

    logging.info("Building cluster graph")

    edge_list = [(list(fs)[0], list(fs)[1]) for fs in edges_weighted.keys()]

    cluster_graph = nx.Graph()
    cluster_graph.add_edges_from(edge_list)

    for ed in cluster_graph.edges():

        cluster_graph[ed[0]][ed[1]]["weight"] = edges_weighted[
            frozenset([ed[0], ed[1]])
        ]

    return cluster_graph, index_to_id_lookup


def extract_communities(
    cluster_graph: nx.Graph, resolution: float, index_lookup: dict
) -> list:
    """Extracts community from the cluster graph and names them
    Args:
        cluster_graph: network object
        resolution: resolution for community detection
        index_lookup: lookup between integer indices and project ids

    Returns:
        a lookup between communities and the projects that belong to them
    """
    logging.info("Extracting communities")
    comms = community_louvain.best_partition(cluster_graph, resolution=resolution)

    comm_assignments = {
        comm: [index_lookup[k] for k, v in comms.items() if v == comm]
        for comm in set(comms.values())
    }

    return {
        obs: cluster
        for cluster, obs_list in comm_assignments.items()
        for obs in obs_list
    }
