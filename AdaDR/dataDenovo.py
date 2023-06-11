# coding=UTF-8
import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import torch as th
import scipy.io as sio
from random import sample
import dgl
from dgl.data.utils import download, extract_archive, get_download_dir
from sklearn.model_selection import KFold
from utils import *

_paths = {
    'Gdataset': './raw_data/drug_data/Gdataset/Gdataset.mat',
    'Cdataset': './raw_data/drug_data/Cdataset/Cdataset.mat',
    'Ldataset': './raw_data/drug_data/Ldataset/lagcn',
    'lrssl': './raw_data/drug_data/lrssl',
}

READ_DATASET_PATH = get_download_dir()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


class DrugNovoData(object):
    def __init__(self, name, device,
                 use_one_hot_fea=True, symm=True,
                 test_ratio=0.1, valid_ratio=0.1, k=2):
        self._name = name
        self._device = device
        self._symm = symm
        self._dis_features_path = "./raw_data/drug_data/" + str(self._name) + "/dis_knn/c" + str(k) + ".txt"
        self._drug_features_path = "./raw_data/drug_data/" + str(self._name) + "/drug_knn/c" + str(k) + ".txt"
        # self.cv = cv
        self._dir = os.path.join(_paths[self._name])
        if self._name in ['Gdataset', 'Cdataset', 'lrssl', 'Ldataset']:

            print("Starting processing {} ...".format(self._name))
            self._dir = os.path.join(_paths[self._name])
            self.cv_data_dict = self._load_drugnovo_data(self._dir, self._name)
            print('Total drug number = {}, disease number = {}'.format(self._num_drug,
                                                                       self._num_disease))

            # Generate features
            if use_one_hot_fea:
                self.drug_feature = None
                self.disease_feature = None

            if self.drug_feature is None:
                self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
                self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)
                # drug_feat = np.loadtxt('drug_features.txt')
                self.drug_feature = th.cat(
                    [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
                     th.zeros([self.num_drug, 1])], 1).to(self._device)

                self.disease_feature = th.cat(
                    [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
                     th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1).to(self._device)

            info_line = "Feature dim: "
            info_line += "\ndrug: {}".format(self.drug_feature.size())
            info_line += "\ndisease: {}".format(self.disease_feature.size())
            print(info_line)
            self._generate_graph()
            self._generate_dis_drug__graph(self._dis_features_path, self._drug_features_path)
            self.possible_rating_values = self.values

        else:
            raise NotImplementedError

        print("train rating pairs : {}".format(self.train_data.shape[0]))
        print("Test rating pairs  : {}".format(self.test_data.shape[0]))

    def _generate_dis_drug__graph(self, dis_features_path, drug_features_path):

        self.dis_edges = np.loadtxt(dis_features_path)
        self.dis_edges = np.array(list(self.dis_edges), dtype=np.int32).reshape(self.dis_edges.shape)
        self.dis_adj = sp.coo_matrix((np.ones(self.dis_edges.shape[0]), (self.dis_edges[:, 0], self.dis_edges[:, 1])),
                                     shape=(self.num_disease, self.num_disease),
                                     dtype=np.float32)
        self.dis_adj = self.dis_adj + self.dis_adj.T.multiply(self.dis_adj.T > self.dis_adj) - self.dis_adj.multiply(
            self.dis_adj.T > self.dis_adj)
        self.dis_graph = self.normalize(self.dis_adj + sp.eye(self.dis_adj.shape[0]))
        self.dis_graph = sparse_mx_to_torch_sparse_tensor(self.dis_graph)

        self.drug_edges = np.loadtxt(drug_features_path)
        self.drug_edges = np.array(list(self.drug_edges), dtype=np.int32).reshape(self.drug_edges.shape)
        self.drug_adj = sp.coo_matrix(
            (np.ones(self.drug_edges.shape[0]), (self.drug_edges[:, 0], self.drug_edges[:, 1])),
            shape=(self.num_drug, self.num_drug),
            dtype=np.float32)
        self.drug_adj = self.drug_adj + self.drug_adj.T.multiply(
            self.drug_adj.T > self.drug_adj) - self.drug_adj.multiply(self.drug_adj.T > self.drug_adj)
        self.drug_graph = self.normalize(self.drug_adj + sp.eye(self.drug_adj.shape[0]))
        self.drug_graph = sparse_mx_to_torch_sparse_tensor(self.drug_graph)

    def _generate_graph(self):
        self.data_cv = {}
        for cv in range(0, len(self.row_idx)):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            # self.train_data, self.test_data, self.values = dataset[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_association_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_association_info = self.test_data
            self.possible_rating_values = self.values

            train_association_pairs, train_association_values = self._generate_pair_value(self.train_association_info)
            test_association_pairs, test_association_values = self._generate_pair_value(self.test_association_info)

            self.train_enc_graph = self._generate_enc_graph(train_association_pairs, train_association_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_association_pairs)
            self.train_truths = th.FloatTensor(train_association_values).to(self._device)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_association_pairs)
            self.test_truths = th.FloatTensor(test_association_values).to(self._device)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv

    def _load_drugnovo_data(self, file_path, data_name):
        self.row_idx = None
        if data_name in ['Gdataset', 'Cdataset']:
            data = sio.loadmat(file_path)
            association_matrix = data['didr'].T
            row_num_count = association_matrix.sum(axis=1)
            self.row_idx = np.where(row_num_count == 1)[0]
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']

        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            association_matrix = association_matrix
            row_num_count = association_matrix.sum(axis=1)
            self.row_idx = np.where(row_num_count == 1)[0]
            self.disease_sim_features = pd.read_csv(os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t')
            self.disease_sim_features = self.disease_sim_features.values
            self.drug_sim_features = pd.read_csv(os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t')
            self.drug_sim_features = self.drug_sim_features.values
        else:
            data = pd.read_csv(file_path, header=None)
            association_matrix = data.values
            row_num_count = association_matrix.sum(axis=1)
            self.row_idx = np.where(row_num_count == 1)[0]
            self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
        self.disease_sim_features = th.FloatTensor(self.disease_sim_features)
        self.drug_sim_features = th.FloatTensor(self.drug_sim_features)
        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]

        cv_num = 0
        cv_data = {}
        for idx in self.row_idx:
            test_value = association_matrix[idx, :]
            test_data = {
                'drug_idx': [idx] * len(test_value),
                'disease_idx': [col for col in range(0, self._num_disease)],
                'values': test_value
            }
            test_data_info = pd.DataFrame(test_data, index=None)

            association_matrix[idx, :] = 0
            pos_row, pos_col = np.nonzero(association_matrix)
            neg_row, neg_col = np.nonzero(1 - association_matrix)

            train_drug_idx = np.hstack([pos_row, neg_row])
            train_disease_idx = np.hstack([pos_col, neg_col])

            pos_values = [1] * len(pos_row)
            neg_values = [0] * len(neg_row)
            train_values = np.hstack([pos_values, neg_values])

            train_data = {
                'drug_idx': train_drug_idx,
                'disease_idx': train_disease_idx,
                'values': train_values
            }
            train_data_info = pd.DataFrame(train_data, index=None)

            values = np.unique(train_values)
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data

    def _make_labels(self, ratings):
        labels = th.LongTensor(np.searchsorted(self.possible_rating_values, ratings)).to(self._device)
        return labels

    def _generate_pair_value(self, association_info):
        rating_pairs = (association_info["drug_idx"].values.astype(np.int64),
                        association_info["disease_idx"].values.astype(np.int64))
        rating_values = association_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'disease': self._num_disease, 'drug': self._num_drug}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('disease', str(rating), 'drug'): (rcol, rrow),
                ('drug', 'rev-%s' % str(rating), 'disease'): (rrow, rcol)
            })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        # assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            disease_ci = []
            disease_cj = []
            drug_ci = []
            drug_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                disease_ci.append(graph['rev-%s' % r].in_degrees())
                drug_ci.append(graph[r].in_degrees())
                if self._symm:
                    disease_cj.append(graph[r].out_degrees())
                    drug_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    disease_cj.append(th.zeros((self.num_disease,)))
                    drug_cj.append(th.zeros((self.num_drug,)))
            disease_ci = _calc_norm(sum(disease_ci))
            drug_ci = _calc_norm(sum(drug_ci))
            if self._symm:
                disease_cj = _calc_norm(sum(disease_cj))
                drug_cj = _calc_norm(sum(drug_cj))
            else:
                disease_cj = th.ones(self.num_disease, )
                drug_cj = th.ones(self.num_drug, )
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(user_movie_ratings_coo, utype='_U', etype='_E', vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_drug(self):
        return self._num_drug

    @property
    def num_disease(self):
        return self._num_disease


if __name__ == '__main__':
    DrugNovoData("lrssl", device=th.device('cpu'), symm=True)
