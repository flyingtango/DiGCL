import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj, to_undirected, is_undirected
from torch_geometric.nn import GCNConv

import numpy as np
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix

from datasets import get_citation_dataset
from model_digcl import Encoder, Model, drop_feature
from eval_digcl import label_classification
from get_adj import *

import warnings
warnings.filterwarnings('ignore')


def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()

    edge_index_1, edge_weight_1 = cal_fast_appr(
        alpha_1, edge_index, x.shape[0], x.dtype)
    edge_index_2, edge_weight_2 = cal_fast_appr(
        alpha_2, edge_index, x.shape[0], x.dtype)

    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)

    z1 = model(x_1, edge_index_1, edge_weight_1)
    z2 = model(x_2, edge_index_2, edge_weight_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, dataset, x, edge_index, edge_weight, y, final=False):
    model.eval()
    z = model(x, edge_index, edge_weight)
    label_classification(z, y, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config_digcl.yaml')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--recache', action="store_true",
                        help="clean up the old adj data", default=True)
    parser.add_argument('--normalize-features',
                        action="store_true", default=True)
    parser.add_argument('--adj-type', type=str, default='or')
    parser.add_argument('--curr-type', type=str, default='log')
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(2021)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU()})[
        config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    alpha_1 = 0.1

    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    path = osp.join(osp.expanduser('.'), 'datasets')
    print(args.normalize_features)
    dataset = get_citation_dataset(
        args.dataset, args.alpha, args.recache, args.normalize_features, args.adj_type)
    print("Num of edges ", dataset[0].num_edges)

    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    edge_index_init, edge_weight_init = cal_fast_appr(
        alpha_1, data.edge_index, data.x.shape[0], data.x.dtype)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        a = 0.9
        b = 0.1
        if args.curr_type == 'linear':
            alpha_2 = a-(a-b)/(num_epochs+1)*epoch
        elif args.curr_type == 'exp':
            alpha_2 = a - (a-b)/(np.exp(3)-1) * \
                (np.exp(3*epoch/(num_epochs+1))-1)
        elif args.curr_type == 'log':
            alpha_2 = a - (a-b)*(1/3*np.log(epoch/(num_epochs+1)+np.exp(-3)))
        elif args.curr_type == 'fixed':
            alpha_2 = 0.9
        else:
            print('wrong curr type')
            exit()

        loss = train(model, data.x, data.edge_index)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    test(model, dataset, data.x, edge_index_init,
         edge_weight_init, data.y, final=True)
