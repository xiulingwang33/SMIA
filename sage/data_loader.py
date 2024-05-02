import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import pickle as pkl
import networkx as nx
import numpy as np
import random

class PokecDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='Pokec')

    def process(self,data_dir,ii,sed):
        # data_dir='./'
        f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
        adj, ft = pkl.load(f2, encoding='latin1')

        g = nx.Graph(adj)

        adj1 = np.array(adj.todense())

        feat_data = ft[:,1:]
        # print((ft))
        src=[]
        dst=[]
        wt=[]
        for eg in g.edges():
            src.append(eg[0])
            dst.append(eg[1])
            wt.append(1)



        gender_index = 1
        pub_index = 0
        age_index = 2
        height_index = 3
        weight_index = 4
        region_index = 5

        labels = np.array(ft)[:, pub_index]

        # nodes_data = pd.read_csv('./members.csv')
        # edges_data = pd.read_csv('./interactions.csv')
        node_features = torch.from_numpy(feat_data.to_numpy())
        node_labels = torch.from_numpy(labels.to_numpy())
        edge_features = torch.from_numpy(wt.to_numpy())
        edges_src = torch.from_numpy(src.to_numpy())
        edges_dst = torch.from_numpy(dst.to_numpy())

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=feat_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = feat_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.1)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


def process1(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:,1:])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def __getitem__(self, i):
    return self.graph

def __len__(self):
    return 1



def process2(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:,2:])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process2(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:,2:])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process3(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:,1:])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process4(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    idx=[1,3,4,5]

    feat_data = list(ft[:,idx])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_genderweight(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-weight.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    idx=[0,2,3,4,5]

    feat_data = list(ft[:,idx])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, gender_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_genderweight_noweight(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-weight.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    idx=[0,2,3,5]

    feat_data = list(ft[:,idx])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, gender_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_genderweight_noheightweight(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-weight.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    idx=[0,2,5]

    feat_data = list(ft[:,idx])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, gender_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_genderweight_noheightweightage(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-weight.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    idx=[0,5]

    feat_data = list(ft[:,idx])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, gender_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

# data_dir='/Users/xiulingwang/Downloads/'
# for sed in range(1,6):
#     for ii in range(-100,100):
#         dataset = process1(data_dir,ii,sed)
#         graph = dataset
#
#         print(graph)


# import dgl
# from dgl.data import DGLDataset
# import torch
# import os
#
# class KarateClubDataset(DGLDataset):
#     def __init__(self):
#         super().__init__(name='karate_club')
#
#     def process(self):
#         nodes_data = pd.read_csv('./members.csv')
#         edges_data = pd.read_csv('./interactions.csv')
#         node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
#         node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())
#         edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
#         edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
#         edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
#
#         self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
#         self.graph.ndata['feat'] = node_features
#         self.graph.ndata['label'] = node_labels
#         self.graph.edata['weight'] = edge_features
#
#         # If your dataset is a node classification dataset, you will need to assign
#         # masks indicating whether a node belongs to training, validation, and test set.
#         n_nodes = nodes_data.shape[0]
#         n_train = int(n_nodes * 0.6)
#         n_val = int(n_nodes * 0.2)
#         train_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         val_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         test_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         train_mask[:n_train] = True
#         val_mask[n_train:n_train + n_val] = True
#         test_mask[n_train + n_val:] = True
#         self.graph.ndata['train_mask'] = train_mask
#         self.graph.ndata['val_mask'] = val_mask
#         self.graph.ndata['test_mask'] = test_mask
#
#     def __getitem__(self, i):
#         return self.graph
#
#     def __len__(self):
#         return 1
#
# dataset = KarateClubDataset()
# graph = dataset[0]
#
# print(graph)

def process_genderweight_randomregion(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-weight.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    idx=[0,5]

    # feat_data = list(ft[:,idx])
    ft_tmp=[]
    # features = ft[:,idx]
    f_tmp=np.random.randint(0,2,np.shape(adj1)[0])
    for i in f_tmp:
        ft_tmp.append([i])

    feat_data = ft_tmp
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, gender_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

# data_dir='/Users/xiulingwang/Downloads/'
# for sed in range(1,6):
#     for ii in range(-100,100):
#         dataset = process1(data_dir,ii,sed)
#         graph = dataset
#
#         print(graph)


# import dgl
# from dgl.data import DGLDataset
# import torch
# import os
#
# class KarateClubDataset(DGLDataset):
#     def __init__(self):
#         super().__init__(name='karate_club')
#
#     def process(self):
#         nodes_data = pd.read_csv('./members.csv')
#         edges_data = pd.read_csv('./interactions.csv')
#         node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
#         node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())
#         edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
#         edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
#         edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
#
#         self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
#         self.graph.ndata['feat'] = node_features
#         self.graph.ndata['label'] = node_labels
#         self.graph.edata['weight'] = edge_features
#
#         # If your dataset is a node classification dataset, you will need to assign
#         # masks indicating whether a node belongs to training, validation, and test set.
#         n_nodes = nodes_data.shape[0]
#         n_train = int(n_nodes * 0.6)
#         n_val = int(n_nodes * 0.2)
#         train_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         val_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         test_mask = torch.zeros(n_nodes, dtype=torch.bool)
#         train_mask[:n_train] = True
#         val_mask[n_train:n_train + n_val] = True
#         test_mask[n_train + n_val:] = True
#         self.graph.ndata['train_mask'] = train_mask
#         self.graph.ndata['val_mask'] = val_mask
#         self.graph.ndata['test_mask'] = test_mask
#
#     def __getitem__(self, i):
#         return self.graph
#
#     def __len__(self):
#         return 1
#
# dataset = KarateClubDataset()
# graph = dataset[0]
#
# print(graph)

def process_fb_edu_gender(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/fb-adj-feat-{1}-{2}-edu-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = 1600
    num_feats = 1283

    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, 77)

    feat_data = list(ft[:, idx])

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels = np.array((ft)[:, g_index]-1,dtype='int64')
    print((labels))


    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)
def process_pubmed(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats-1)

    feat_data = list(ft[:, idx])

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    # g_index = 77  # gender
    # r_index = 1154  # religion
    # p_index = 1278  # political
    # e_index = 53  # education_type

    labels = np.array((ft)[:, num_feats-1]-1,dtype='int64')
    print((labels))


    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 4000
    n_val = 1000
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_genderweight_predefense(data_dir,ii,sed,ratio):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-weight.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    num_edges = g.number_of_edges()

    num_add = int(ratio * num_edges)

    neg_edges = []

    num_nodes = 10000

    idx_adj=np.array(range(0,num_nodes*num_nodes))
    idx_edges=[]


    for eg in g.edges():
        nd0=eg[0]
        nd1 = eg[1]

        if (nd1>nd0):
            tmp=nd0
            nd0=nd1
            nd1=tmp

        edge_idx=nd0*num_nodes+nd1
        idx_edges.append(edge_idx)
    idx_edges=np.array(idx_edges)

    idx_neg_edges=np.delete(idx_adj, idx_edges)

    idx_add_edges = random.sample(list(idx_neg_edges), num_add)

    add_edges=[]

    for ed in idx_add_edges:
        nd1=int(ed / num_nodes)
        nd2=ed % num_nodes
        add_edges.append([nd1,nd2])
        add_edges.append([nd2, nd1])
    g.add_edges_from(add_edges)


    adj = nx.adjacency_matrix(g)


    idx=[0,2,3,4,5]

    feat_data = list(ft[:,idx])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, gender_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pubmed_predefense(data_dir,ii,sed,ratio):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats-1)

    feat_data = list(ft[:, idx])

    num_edges = g.number_of_edges()

    num_add = int(ratio * num_edges)

    neg_edges = []

    idx_adj=np.array(range(0,num_nodes*num_nodes))
    idx_edges=[]


    for eg in g.edges():
        nd0=eg[0]
        nd1 = eg[1]

        if (nd1>nd0):
            tmp=nd0
            nd0=nd1
            nd1=tmp

        edge_idx=nd0*num_nodes+nd1
        idx_edges.append(edge_idx)
    idx_edges=np.array(idx_edges)

    idx_neg_edges=np.delete(idx_adj, idx_edges)

    idx_add_edges = random.sample(list(idx_neg_edges), num_add)

    add_edges=[]

    for ed in idx_add_edges:
        nd1=int(ed / num_nodes)
        nd2=ed % num_nodes
        add_edges.append([nd1,nd2])
        add_edges.append([nd2, nd1])
    g.add_edges_from(add_edges)

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    # g_index = 77  # gender
    # r_index = 1154  # religion
    # p_index = 1278  # political
    # e_index = 53  # education_type

    labels = np.array((ft)[:, num_feats-1]-1,dtype='int64')
    print((labels))


    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 4000
    n_val = 1000
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pubmed_predefense_fulledges(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    # g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())
    adj=np.ones((num_nodes,num_nodes))
    g = nx.Graph(adj)

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats-1)

    feat_data = list(ft[:, idx])

    num_edges = g.number_of_edges()

    # num_add = int(ratio * num_edges)
    #
    # neg_edges = []
    #
    # idx_adj=np.array(range(0,num_nodes*num_nodes))
    # idx_edges=[]
    #
    #
    # for eg in g.edges():
    #     nd0=eg[0]
    #     nd1 = eg[1]
    #
    #     if (nd1>nd0):
    #         tmp=nd0
    #         nd0=nd1
    #         nd1=tmp
    #
    #     edge_idx=nd0*num_nodes+nd1
    #     idx_edges.append(edge_idx)
    # idx_edges=np.array(idx_edges)
    #
    # idx_neg_edges=np.delete(idx_adj, idx_edges)
    #
    # idx_add_edges = random.sample(list(idx_neg_edges), num_add)
    #
    # add_edges=[]
    #
    # for ed in idx_add_edges:
    #     nd1=int(ed / num_nodes)
    #     nd2=ed % num_nodes
    #     add_edges.append([nd1,nd2])
    #     add_edges.append([nd2, nd1])
    # g.add_edges_from(add_edges)

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    # g_index = 77  # gender
    # r_index = 1154  # religion
    # p_index = 1278  # political
    # e_index = 53  # education_type

    labels = np.array((ft)[:, num_feats-1]-1,dtype='int64')
    print((labels))


    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 4000
    n_val = 1000
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_genderweight_predefense_fulledges(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-weight.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())
    adj = np.ones((num_nodes, num_nodes))
    g = nx.Graph(adj)

    adj = nx.adjacency_matrix(g)


    idx=[0,2,3,4,5]

    feat_data = list(ft[:,idx])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, gender_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_fb_interintra(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/fb-adj-feat-{1}-{2}-weight.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = 1600
    num_feats = 1283

    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, 77)
    idx = np.delete(idx, 53)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels = np.array((ft)[:, e_index]-1,dtype='int64')
    print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_fb_edu_gender2(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/fb-adj-feat-{1}-{2}-edu-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    # num_nodes = 1600
    num_feats = 1283

    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, 53)

    feat_data = list(ft[:, idx])

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels = np.array((ft)[:, e_index]-1,dtype='int64')
    print((labels))


    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    # n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process3(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:,1:])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_pub_gender_interintra(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:,2:])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pubmed_interintra(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    # g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    feat_data = list(ft[:, idx])

    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')
    print((labels))


    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_pub_gender_interintra_small(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:,1:])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_fb_interintra_small(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/fb-adj-feat-{1}-{2}-weight.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = 1283

    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    # idx = np.delete(idx, 77)
    idx = np.delete(idx, 53)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels = np.array((ft)[:, e_index]-1,dtype='int64')
    print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pubmed_interintra_small(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    # g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    feat_data = list(ft[:, idx])

    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')
    print((labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)



def process_popets(ego_user,g,ft,labels):
    # data_dir='./'

    g1 = g
    adj = nx.adjacency_matrix(g1)


    labels = labels[:,1]
    print(set(labels))

    # if ego_user=='citeseer':
    #     idx=np.where(labels==100)
    #
    #     labels[idx]=6

    print(set(labels))

    feat_data = list(ft)
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)


    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    # n_train = g1.number_of_nodes()
    n_val = int(0.1*g1.number_of_nodes())
    n_train = int(0.7*g1.number_of_nodes())
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pubmed_interintra_small(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    # g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    feat_data = list(ft[:, idx])

    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')
    print((labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)



def process_pubmed_small(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    feat_data = list(ft[:, idx])

    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')
    print((labels))
    print(set(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_fb_edu_gender_small(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/fb-adj-feat-{1}-{2}-edu-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = 1600
    num_feats = 1283

    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, 53)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels = np.array((ft)[:, e_index]-1,dtype='int64')
    print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pokec_gender_small(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:,1:])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pubmed_small_2classes(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    feat_data = list(ft[:, idx])

    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')
    # print((labels))
    # print(set(labels))


    for la in range(len(labels)):
        if labels[la] == 2:
            labels[la] = 0

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pubmed_interintra_small_2classes(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    # g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    feat_data = list(ft[:, idx])

    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')
    print((labels))

    for la in range(len(labels)):
        if labels[la] == 2:
            labels[la] = 1

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_pub_gender_interintra_small_nogender(data_dir, ii, sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:, 2:])
    # print((ft))
    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index], dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_pub_gender_interintra_small_nogenderweight(data_dir, ii, sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    ft_idx = [2, 3, 5]

    feat_data = list(ft[:, ft_idx])
    # print((ft))
    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index], dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_pub_gender_interintra_small_nogenderweightheight(data_dir, ii, sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    ft_idx = [2, 5]

    feat_data = list(ft[:, ft_idx])
    # print((ft))
    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index], dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_pub_gender_interintra_small_random(data_dir, ii, sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    # ft_idx = [2, 5]
    #
    # feat_data = list(ft[:, ft_idx])

    num_nodes = g.number_of_nodes()

    # feat_data = list(np.random.randint(0, 2, num_nodes))
    # print((ft))
    features1 = np.random.randint(0, 2, num_nodes)
    features2 = np.random.randint(0, 2, num_nodes)
    features1=features1.reshape(-1,1)
    features2=features2.reshape(-1, 1)
    # print(np.shape(features1))
    feat_data = list(np.concatenate((features1,features2),axis=1))
    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index], dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_pub_gender_interintra_small_identical(data_dir, ii, sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    # ft_idx = [2, 5]
    #
    # feat_data = list(ft[:, ft_idx])

    num_nodes = g.number_of_nodes()

    # feat_data = list(np.random.randint(0, 2, num_nodes))
    # print((ft))
    # features1 = np.random.randint(0, 2, num_nodes)
    # features2 = np.random.randint(0, 2, num_nodes)
    # features1=features1.reshape(-1,1)
    # features2=features2.reshape(-1, 1)
    # # print(np.shape(features1))
    # feat_data = list(np.concatenate((features1,features2),axis=1))

    feat_data = list(np.ones((num_nodes, 2)))
    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index], dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)



def process_genderweight_identical(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-weight.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    idx=[0,5]

    # # feat_data = list(ft[:,idx])
    # ft_tmp=[]
    # # features = ft[:,idx]
    # f_tmp=np.random.randint(0,2,np.shape(adj1)[0])
    # for i in f_tmp:
    #     ft_tmp.append([i])
    num_nodes = g.number_of_nodes()

    feat_data = list(np.ones((num_nodes, 2)))
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, gender_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_gender_identical(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())
    num_nodes = g.number_of_nodes()

    # feat_data = list(ft[:,1:])
    feat_data = list(np.ones((num_nodes, 2)))
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)



def process_gender_nogenderweight(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    ft_idx=[2,3,5]

    feat_data = list(ft[:,ft_idx])
    # feat_data = list(np.ones((num_nodes, 2)))
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_gender_nogenderweightheight(data_dir, ii, sed):
    # data_dir='./'
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    ft_idx = [2, 5]

    feat_data = list(ft[:, ft_idx])
    # feat_data = list(np.ones((num_nodes, 2)))
    # print((ft))
    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index], dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_gender_pubflip(data_dir, ii, sed):
    # data_dir='./'
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    ft_idx = [2, 5]

    # feat_data = list(ft[:, ft_idx])
    num_nodes = g.number_of_nodes()
    feat_data = list(np.ones((num_nodes, 2)))
    # print((ft))
    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    # labels = np.array((ft)[:, pub_index], dtype='int64')
    labels1 = np.array(ft)[:, gender_index]
    lb1_idx = np.where(labels1 == 0)[0]
    lb2_idx = np.where(labels1 == 1)[0]

    print(lb1_idx, lb2_idx)

    lb1_idx_1 = np.array(random.sample(list(lb1_idx), int(0.5 * len(lb1_idx))))
    lb2_idx_1 = np.array(random.sample(list(lb2_idx), int(0.5 * len(lb2_idx))))

    labels = np.ones(len(labels1), dtype='int64')

    labels[lb1_idx_1] = 0
    labels[lb2_idx_1] = 0

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)



def process_pokec_pub_gender_interintra_small_pubflip(data_dir, ii, sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    # ft_idx = [2, 5]
    #
    # feat_data = list(ft[:, ft_idx])

    num_nodes = g.number_of_nodes()

    # feat_data = list(np.random.randint(0, 2, num_nodes))
    # print((ft))
    # features1 = np.random.randint(0, 2, num_nodes)
    # features2 = np.random.randint(0, 2, num_nodes)
    # features1=features1.reshape(-1,1)
    # features2=features2.reshape(-1, 1)
    # # print(np.shape(features1))
    # feat_data = list(np.concatenate((features1,features2),axis=1))

    feat_data = list(np.ones((num_nodes, 2)))
    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels1 = np.array(ft)[:, gender_index]
    lb1_idx = np.where(labels1 == 0)[0]
    lb2_idx = np.where(labels1 == 1)[0]

    print(lb1_idx, lb2_idx)

    lb1_idx_1 = np.array(random.sample(list(lb1_idx), int(0.5 * len(lb1_idx))))
    lb2_idx_1 = np.array(random.sample(list(lb2_idx), int(0.5 * len(lb2_idx))))

    labels = np.ones(len(labels1), dtype='int64')

    labels[lb1_idx_1] = 0
    labels[lb2_idx_1] = 0

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_gender_pubflip0(data_dir, ii, sed):
    # data_dir='./'
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    ft_idx = [2, 5]

    # feat_data = list(ft[:, ft_idx])
    num_nodes = g.number_of_nodes()
    feat_data = list(ft[:, 1:])
    # feat_data = list(np.ones((num_nodes, 2)))
    # print((ft))
    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    # labels = np.array((ft)[:, pub_index], dtype='int64')
    labels1 = np.array(ft)[:, gender_index]
    lb1_idx = np.where(labels1 == 0)[0]
    lb2_idx = np.where(labels1 == 1)[0]

    print(lb1_idx, lb2_idx)
    lb1_idx_1 = np.array(random.sample(list(lb1_idx), int(0.5 * len(lb1_idx))))
    lb2_idx_1 = np.array(random.sample(list(lb2_idx), int(0.5 * len(lb2_idx))))

    labels = np.ones(len(labels1), dtype='int64')

    labels[lb1_idx_1] = 0
    labels[lb2_idx_1] = 0

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pokec_pub_gender_interintra_small_pubflip0(data_dir, ii, sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    # ft_idx = [2, 5]
    #
    # feat_data = list(ft[:, ft_idx])

    num_nodes = g.number_of_nodes()

    # feat_data = list(np.random.randint(0, 2, num_nodes))
    # print((ft))
    # features1 = np.random.randint(0, 2, num_nodes)
    # features2 = np.random.randint(0, 2, num_nodes)
    # features1=features1.reshape(-1,1)
    # features2=features2.reshape(-1, 1)
    # # print(np.shape(features1))
    # feat_data = list(np.concatenate((features1,features2),axis=1))

    feat_data = list(ft[:,1:])
    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels1 = np.array(ft)[:, gender_index]
    lb1_idx = np.where(labels1 == 0)[0]
    lb2_idx = np.where(labels1 == 1)[0]

    print(lb1_idx, lb2_idx)

    lb1_idx_1 = np.array(random.sample(list(lb1_idx), int(0.5 * len(lb1_idx))))
    lb2_idx_1 = np.array(random.sample(list(lb2_idx), int(0.5 * len(lb2_idx))))

    labels = np.ones(len(labels1), dtype='int64')

    labels[lb1_idx_1] = 0
    labels[lb2_idx_1] = 0

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_pub_gender_interintra_small_switch(data_dir, ii, sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-switch.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    ft_idx = [1,2,3,4,5]
    #
    # for jj in range(np.shape(ft)[0]):
    #     if ft[jj][1]==0:
    #         ft[jj][1]=1
    #     elif ft[jj][1]==1:
    #         ft[jj][1] = 0
    #     else:
    #         print('error')

    feat_data = list(ft[:, ft_idx])


    num_nodes = g.number_of_nodes()


    src = []
    dst = []
    wt = []
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index], dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_gender_switch(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    ft_idx = [1, 2, 3, 4, 5]
    #
    for jj in range(np.shape(ft)[0]):
        if ft[jj][1] == 0:
            ft[jj][1] = 1
        elif ft[jj][1] == 1:
            ft[jj][1] = 0
        else:
            print('error')

    feat_data = list(ft[:, ft_idx])
    # feat_data = list(np.ones((num_nodes, 2)))
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)



def process_fb_edu_gender2_switch(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/fb-adj-feat-{1}-{2}-edu-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = 1600
    num_feats = 1283

    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, 53)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))


    for jj in range(np.shape(ft)[0]):
        if ft[jj][77] == 1:
            ft[jj][77] = 2
        elif ft[jj][77] == 2:
            ft[jj][77] = 1
        else:
            print('error')


    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels = np.array((ft)[:, e_index]-1,dtype='int64')
    print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_fb_interintra_small_switch(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/fb-adj-feat-{1}-{2}-switch.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = 1283

    # for jj in range(np.shape(ft)[0]):
    #     if ft[jj][77] == 1:
    #         ft[jj][77] = 2
    #     elif ft[jj][77] == 2:
    #         ft[jj][77] = 1
    #     else:
    #         print('error')

    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    # idx = np.delete(idx, 77)
    idx = np.delete(idx, 53)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels = np.array((ft)[:, e_index]-1,dtype='int64')
    print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pubmed_interintra_small_switch(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    for jj in range(np.shape(ft)[0]):
        if ft[jj][162] == 0:
            ft[jj][162] = 1
        elif ft[jj][162] != 0:
            ft[jj][162] = 0
        else:
            print('error')


    # g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    feat_data = list(ft[:, idx])

    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')
    print((labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pubmed_small_switch(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    for jj in range(np.shape(ft)[0]):
        if ft[jj][162] == 0:
            ft[jj][162] = 1
        elif ft[jj][162] != 0:
            ft[jj][162] = 0
        else:
            print('error')

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    feat_data = list(ft[:, idx])

    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')
    print((labels))
    print(set(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_fb_edu_ego(data_dir,nd):
    # data_dir='./'
    f2 = open("./698-adj-feat.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)

    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    adj1=np.array(adj.todense())



    ft = np.delete(ft, nd,0)

    print(np.shape(ft))
    print(np.shape(adj1))

    adj1 = np.delete(adj1, nd,0)
    adj1 = np.delete(adj1, nd,1)
    print(np.shape(adj1))

    g1 = nx.Graph(adj1)

    adj=nx.adjacency_matrix(g1)

    labels = np.array((ft)[:, 26],dtype='int64')
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    idx = np.delete(idx, 26)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = g1.number_of_nodes()
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_fb_edu_ego_link(data_dir, nd):
    # data_dir='./'
    f2 = open("./698-adj-feat.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)

    ed = list(g.edges())[nd]
    e1 = ed[0]
    e2 = ed[1]

    adj1 = np.array(adj.todense())
    adj1[e1][e2] = 0
    adj1[e2][e1] = 0

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)

    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, 26], dtype='int64')
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    idx = np.delete(idx, 26)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src = []
    dst = []
    wt = []
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = g1.number_of_nodes()
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_fb_edu_ego_orig(data_dir, nd):
    # data_dir='./'
    f2 = open("./698-adj-feat.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)

    g1 = g
    adj = nx.adjacency_matrix(g1)

    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, 26], dtype='int64')
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    idx = np.delete(idx, 26)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src = []
    dst = []
    wt = []
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = g1.number_of_nodes()
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_fb_edu_ego_orig_noise(data_dir, nd):
    # data_dir='./'
    f2 = open("./698-adj-feat.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)

    g1 = g
    adj = nx.adjacency_matrix(g1)

    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, 26], dtype='int64')
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    idx = np.delete(idx, 26)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src = []
    dst = []
    wt = []
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data, dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt, dtype='float32'))
    edges_src = torch.from_numpy(np.array(src, dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst, dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(0.7*g1.number_of_nodes())
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)





def process_pokec_ego(data_dir,nd):
    # data_dir='./'
    f2 = open("./pokec-ego.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)

    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    adj1=np.array(adj.todense())



    ft = np.delete(ft, nd,0)

    print(np.shape(ft))
    print(np.shape(adj1))

    adj1 = np.delete(adj1, nd,0)
    adj1 = np.delete(adj1, nd,1)
    print(np.shape(adj1))

    g1 = nx.Graph(adj1)

    adj=nx.adjacency_matrix(g1)

    labels = np.array((ft)[:, 0],dtype='int64')
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    idx = np.delete(idx, 0)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = g1.number_of_nodes()
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_ego_link(data_dir,nd):
    # data_dir='./'
    f2 = open("./pokec-ego.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)

    ed=list(g.edges())[nd]
    e1=ed[0]
    e2=ed[1]

    adj1=np.array(adj.todense())
    adj1[e1][e2]=0
    adj1[e2][e1] = 0

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)


    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, 0],dtype='int64')
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    idx = np.delete(idx, 0)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = g1.number_of_nodes()
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pokec_ego_orig(data_dir,nd):
    # data_dir='./'
    f2 = open("./pokec-ego.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)


    g1 = g
    adj = nx.adjacency_matrix(g1)


    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, 0],dtype='int64')
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    idx = np.delete(idx, 0)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = g1.number_of_nodes()
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_ego_orig_noise(data_dir,nd):
    # data_dir='./'
    f2 = open("./pokec-ego.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)


    g1 = g
    adj = nx.adjacency_matrix(g1)


    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, 0],dtype='int64')
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    idx = np.delete(idx, 0)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(0.7 * g1.number_of_nodes())
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)





def process_pubmed_ego(data_dir,nd):
    # data_dir='./'
    f2 = open("./pubmed-ego.pkl", 'rb')
    adj, ft,lb = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)

    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    adj1=np.array(adj.todense())



    # ft = np.delete(ft, nd,0)

    print(np.shape(ft))
    print(np.shape(adj1))

    adj1 = np.delete(adj1, nd,0)
    adj1 = np.delete(adj1, nd,1)
    print(np.shape(adj1))

    g1 = nx.Graph(adj1)

    adj=nx.adjacency_matrix(g1)

    labels = lb-1
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    idx = np.delete(idx, 0)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = g1.number_of_nodes()
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pubmed_ego_link(data_dir,nd):
    # data_dir='./'
    f2 = open("./pubmed-ego.pkl", 'rb')
    adj, ft,lb = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)

    ed=list(g.edges())[nd]
    e1=ed[0]
    e2=ed[1]

    adj1=np.array(adj.todense())
    adj1[e1][e2]=0
    adj1[e2][e1] = 0

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)


    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    # labels = np.array((ft)[:, 0],dtype='int64')
    labels=lb-1
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    # idx = np.delete(idx, 0)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = g1.number_of_nodes()
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pubmed_ego_orig(data_dir,nd):
    # data_dir='./'
    f2 = open("./pubmed-ego.pkl", 'rb')
    adj, ft,lb = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)


    g1 = g
    adj = nx.adjacency_matrix(g1)


    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    # labels = np.array((ft)[:, 0],dtype='int64')
    labels=lb-1
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    # idx = np.delete(idx, 0)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = g1.number_of_nodes()
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pubmed_ego_orig_noise(data_dir,nd):
    # data_dir='./'
    f2 = open("./pubmed-ego.pkl", 'rb')
    adj, ft,lb = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)


    g1 = g
    adj = nx.adjacency_matrix(g1)


    # num_nodes = 1600
    # num_feats = 1283

    # adj1 = np.array(adj.todense())

    # labels = np.array((ft)[:, 0],dtype='int64')
    labels=lb-1
    print(set(labels))

    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    # idx = np.delete(idx, 0)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g1.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    # labels = np.array((ft)[:, e_index]-1,dtype='int64')
    # print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(0.7*g1.number_of_nodes())
    n_val = 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_region_age(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    idx=[0,1,2,3,4]

    feat_data = list(ft[:,idx])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, region_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)





