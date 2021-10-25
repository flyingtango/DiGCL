import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os
import time
import torch
from scipy.linalg import fractional_matrix_power, inv
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected

from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops, to_dense_adj
from torch_scatter import scatter_add
import scipy
from cal_fast_pagerank import *
# import fast_pagerank


def get_undirected_adj(edge_index, num_nodes, dtype):
    edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                             device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def cal_fast_appr(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight == None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)
    fill_value = 1
    # print(alpha)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index

    # from tensor to csr matrix
    sparse_adj = sp.csr_matrix(
        (edge_weight.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())), shape=(num_nodes, num_nodes))

    tol = 1e-6
    L, _ = fast_appr_power(
        sparse_adj, alpha=alpha, tol=tol)

    L = L.tocoo()
    values = L.data
    indices = np.vstack((L.row, L.col))

    L_indices = torch.LongTensor(indices).to(edge_index.device)
    L_values = torch.FloatTensor(values).to(edge_index.device)

    edge_index = L_indices
    edge_weight = L_values

    # sys normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def get_pr_directed_adj(alpha, edge_index, num_nodes, dtype):
    edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                             device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight
    p_dense = torch.sparse.FloatTensor(
        edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    # pagerank p
    p_pr = (1.0-alpha) * p_dense + alpha / num_nodes * \
        torch.ones((num_nodes, num_nodes), dtype=dtype, device=p.device)

    eig_value, left_vector = scipy.linalg.eig(
        p_pr.numpy(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    # assert val[0] == 1.0

    pi = left_vector[:, ind[0]]  # choose the largest eig vector
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi < 0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_pr
    L = (torch.mm(torch.mm(pi_sqrt, p_pr), pi_inv_sqrt) +
         torch.mm(torch.mm(pi_inv_sqrt, p_pr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # # let little possbility connection to 0, make L sparse
    # L[ L < (1/num_nodes)] = 0
    # L[ L < 5e-4] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight == None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)
    fill_value = 1
    # print(alpha)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(
        edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1, num_nodes+1]))
    p_v[0:num_nodes, 0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes, 0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes, num_nodes] = alpha
    p_v[num_nodes, num_nodes] = 0.0
    p_ppr = p_v

    eig_value, left_vector = scipy.linalg.eig(
        p_ppr.numpy(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:, ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi
    # print(pi)
    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi < 0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    # print(pi_sqrt.device)
    # print(p_ppr.device)
    # print(pi_inv_sqrt.device)
    # exit()
    # print(p_ppr.numpy())
    pi_sqrt = pi_sqrt.to(p_ppr.device)
    pi_inv_sqrt = pi_inv_sqrt.to(p_ppr.device)
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) +
         torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0
    # print(L[7, 1198].numpy())
    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()
    # L_indices = torch.nonzero(L, as_tuple=False)

    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values
    # print(L_indices[:, 0:20])
    # print(L_indices.shape)
    # print(L_values[0:20])

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]