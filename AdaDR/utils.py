import csv
import random
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim

from scipy import sparse as sp
from collections import OrderedDict


class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()


def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) + \
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace('.', '_')


def common_loss(emb1, emb2):
    emb1 = emb1 - th.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - th.mean(emb2, dim=0, keepdim=True)
    emb1 = th.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = th.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = th.matmul(emb1, emb1.t())
    cov2 = th.matmul(emb2, emb2.t())
    cost = th.mean((cov1 - cov2) ** 2)
    return cost


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True

# def knn_graph(disMat, k):
#     num  = disMat.shape[0]
#     inds = []
#     for i in range(disMat.shape[0]):
#         ind = np.argpartition(disMat[i, :], kth=k)[:k]
#         inds.append(ind)

#     inds_ = []
#     for i, v in enumerate(inds):
#         for vv in v:
#             if vv == i:
#                 pass
#             else:
#                 inds_.append([i, vv])
    
#     inds_ = np.array(inds_)
#     edges = np.array([inds_[:, 0], inds_[:,1]]).astype(int)
#     edges_inver = np.array([inds_[:, 1], inds_[:, 0]]).astype(int)
#     edges_index = np.concatenate((edges, edges_inver), axis=1).T
#     # Remove repeating entry
#     edges_index = np.unique(edges_index, axis=0)
#     adjs = sp.coo_matrix((np.ones(edges_index.shape[0]), (edges_index[:, 0], edges_index[:, 1])),
#                                 shape=(num, num), dtype=np.float32)
#     return adjs + sp.eye(adjs.shape[0])


def knn_graph(disMat, k):
    k_neighbor = np.argpartition(-disMat, kth=k, axis=1)[:, :k]
    row_index = np.arange(k_neighbor.shape[0]).repeat(k_neighbor.shape[1])
    col_index = k_neighbor.reshape(-1)
    edges = np.array([row_index, col_index]).astype(int).T
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(disMat.shape[0], disMat.shape[0]),
                                dtype=np.float32)
    # Remove diagonal elements
    # drug_adj = drug_adj - sp.dia_matrix((drug_adj.diagonal()[np.newaxis, :], [0]), shape=drug_adj.shape)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    return adj