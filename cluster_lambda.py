# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy import random as rng
import scipy.sparse as sparse

from sgcrf import SparseGaussianCRF

"""
Lets set up the clustering examples

Ported from Calvin McCarter!
"""

def cluster_lambda(q, cluster_size, avg_degree, within_frac, epsilon):
    graph = cluster_graph(q, cluster_size, avg_degree, within_frac)
    tol = 1e-8
    if q < 1e6:
        min_eig = sparse.linalg.eigsh(graph, k=q-1, return_eigenvectors=False, which='SM', tol=tol).min()
    else:
        min_eig = -10
    # maybe epsilon is machine epsilon? np.finfo(np.float32).eps
    L_diag = (-1 * min_eig + epsilon) * sparse.identity(q)

    L = graph + L_diag

    return L



def cluster_graph(q, cluster_size, avg_degree, within_frac):

    num_clusters = int(np.floor(q / cluster_size))
    assert cluster_size * num_clusters == q

    num_edges = q * avg_degree
    num_within_edges = num_edges * within_frac
    num_without_edges = num_edges - num_within_edges

    block_size = cluster_size**2

    within_area = 0.5 * (block_size * num_clusters - q)
    within_density = num_within_edges / within_area

    assert num_within_edges <= within_area - q, 'infeasible number of edges within clusters'

    without_area = 0.5 * (q**2 - block_size * num_clusters)
    without_density = 0
    if without_area != 0:
        without_density = num_without_edges / without_area

    assert num_without_edges <= without_area, 'infeasible number of edges outside clusters'

    # create inter-cluster edges
    graph_without = sp_rand_sym(q, without_density)
    # graph_without[:q+1:q**2] = 0 # this looks like its trying to set diagonal to zeros in matlab, but wont work with np arrays
    graph_without[np.diag_indices(q)] = 0 # this does that in python
    graph_without[graph_without != 0] = 1
    actual_without_edge = sparse.triu(graph_without, k=1).nnz

    blocks = []
    for cluster in range(num_clusters):
        begin_ix = cluster * cluster_size
        end_ix = (cluster + 1) * cluster_size
        block = sp_rand_sym(cluster_size, within_density)
        block -= graph_without[begin_ix: end_ix, begin_ix:end_ix]
        block[np.diag_indices(cluster_size)] = 0
        block[block != 0] = 1
        blocks.append(block)

    graph_within = sparse.block_diag(blocks)
    actual_within_edges = sparse.triu(graph_within, k=1).nnz

    graph = graph_within + graph_without
    assert graph.diagonal().sum() == 0

    actual_avg_degree = sparse.triu(graph, k=1).nnz / q # not used?

    return graph


def sp_rand_sym(rank, density=0.1, format=None, dtype=None, random_state=None):
    """
    from stack overflow
    http://stackoverflow.com/a/26895721/5257074
    looks aight
    """
    density = density / (2.0 - 1.0/rank)
    A = sparse.rand(rank, rank, density=density, format=format, dtype=dtype, random_state=random_state)
    return (A + A.transpose())/2


def linear_chain(n, a, b):
    """
    Generate a linear chain graph like:

    y1 ---- y2 ---- y3 -...- yn
     |       |       |        |
     |       |       |  ...   |
     |       |       |        |
    x1      x2      x3  ...  xn

    where <a> describes the conditional dependence between output variables,
    and <b> describes the relative influence of the input variables on the
    output variables.

    This is equivalent to nxn Lambda with <1>s along diagonal, and <a>s along
    super and infra diagonal, and nxn Theta with <a*b> along the diagonal.

    from appendix of
    Wytock and Kolter 2013
    Sparse Gaussian CRFs Algorithms Theory and Application
    """
    assert a > 0, 'a must be positive'
    assert b > 0, 'b must be positive'

    L = np.eye(n) + a * (np.diagflat(np.ones(n-1), 1) + np.diagflat(np.ones(n-1), -1))
    T = a*b * np.eye(n)

    return L, T
