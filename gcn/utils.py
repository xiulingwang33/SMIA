import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
import networkx as nx
import random
import scipy.sparse


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def load_data_fb(seed,path="../data/", dataset="facebook"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 22470
    num_feats = 4714
    f2 = open('./combined-adj-feat-train-{}-12-2.pkl'.format(seed), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft

    labels = []

    gindex = 77
    for i, n in enumerate(g.nodes()):
        if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
            ginfo = 1  # male
            labels.append(ginfo)
        elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
            ginfo = 2  # female
            labels.append(ginfo)

        else:
            print('***')
            ginfo = 0  # unknow gender
            labels.append(ginfo)

        # print(ginfo)

        g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = sp.csr_matrix(ft, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    print(adj)
    print(np.shape(adj))

    idx_test = range(1800,1977)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1800)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test




def load_data_fb_link_prediction(seed,path="../data/", dataset="facebook"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # num_nodes = 22470
    # num_feats = 4714
    f2 = open('./combined-adj-feat-train-{}-12-2.pkl'.format(seed), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    features = scipy.sparse.csr_matrix(ft, dtype=np.float32)
    # print(ft)
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)

    num_nodes=g.number_of_nodes()
    # num_edges = g.number_of_edges()


    adj1=np.array(adj.todense())
    # print(adj1)

    # feat_data = ft

    # labels = []
    #
    # gindex = 77
    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo


    edges=[]
    num_edges=0
    for i in range(np.shape(adj1)[0]):
        for j in range(i,np.shape(adj1)[0]):
            if adj1[i][j]==1:
                edges.append([i,j,1])
                num_edges+=1

    idx = int((num_edges) * 0.8)

    all = list(range(num_edges))

    train_idx = random.sample(all, idx)

    test_idx = np.array(all)[~np.in1d(np.array(all), np.array(train_idx))]

    # edge_train=random.sample(edges, idx)
    # edge_test = np.array(edges)[~np.in1d(np.array(edges), np.array(edge_train))]

    edge_train=np.array(edges)[train_idx]
    edge_test = np.array(edges)[test_idx]


    adj_train = np.zeros((num_nodes, num_nodes))
    adj_train_edges=[]
    for i in edge_train:
        adj_train_edges.append([i[0],i[1]])
        adj_train_edges.append([i[1], i[0]])
        adj_train[i[0]][i[1]]=1.0
        adj_train[i[1]][i[0]] = 1.0



    g_train=nx.Graph(adj_train)

    adj_train0=nx.adjacency_matrix(g_train)

    # build symmetric adjacency matrix
    adj1 = adj_train0 + adj_train0.T.multiply(adj_train0.T >adj_train0) - adj_train0.multiply(adj_train0.T > adj_train0)
    features = normalize(features)
    adj1 = normalize(adj1 + scipy.sparse.eye(adj1.shape[0]))


    neg_edges=[]

    # adj_train0=np.array(adj_train0.todense())
    adj2 = np.array(adj.todense())
    for nd in g_train.nodes():
        neg_nodes = []
        dgr=g.degree(nd)
        neg_nd=int(dgr**(3/4))
        for n in range(g.number_of_nodes()):
            if adj2[nd][n]==0:
                neg_nodes.append(n)

        random.shuffle(neg_nodes)
        neg_sample=random.sample(neg_nodes,neg_nd)
        for sp in neg_sample:
            neg_edges.append([nd,sp,0])



    random.shuffle(neg_edges)

    all = list(range(np.shape(neg_edges)[0]))

    train_idx = random.sample(all, int(0.8*np.shape(neg_edges)[0]))

    test_idx = np.array(all)[~np.in1d(np.array(all), np.array(train_idx))]

    neg_edges_train=np.array(neg_edges)[train_idx]
    neg_edges_test = np.array(neg_edges)[test_idx]

    print(np.shape(neg_edges_train))

    print(np.shape(neg_edges_test))


    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # print(adj)
    # print(np.shape(adj))



    # idx_test = range(idx,num_nodes)
    # idx_val = range(1800, 1977)
    # idxneg_train = range(0,idx)
    train_edges=np.concatenate((np.array(edge_train), np.array(neg_edges_train)), axis=0)
    test_edges = np.concatenate((np.array(edge_test), np.array(neg_edges_test)), axis=0)


    train_lables=[1] * np.shape(edge_train)[0] + [0] * np.shape(neg_edges_train)[0]
    test_lables = [1] * np.shape(edge_test)[0] + [0] * np.shape(neg_edges_test)[0]
    # print(train_lables)

    # print(np.shape(train_lables))
    # print(np.sum(train_lables))

    all = list(range(np.shape(train_edges)[0]))

    sp_idx = random.sample(all, np.shape(train_edges)[0])

    train_edges1=train_edges[np.array(sp_idx)]
    train_labels1=np.array(train_lables)[np.array(sp_idx)]
    # print(train_lables)

    # print(np.shape(train_labels1))
    # print(np.sum(train_labels1))
    # exit()

    all = list(range(np.shape(test_edges)[0]))

    sp_idx = random.sample(all, np.shape(test_edges)[0])

    test_edges1=test_edges[np.array(sp_idx)]
    test_labels1=np.array(test_lables)[np.array(sp_idx)]


    print(np.shape(train_lables), np.shape(train_edges), np.shape(test_lables), np.shape(test_edges))

    features = torch.FloatTensor(np.array(features.todense()))
    train_labels1 = torch.LongTensor(train_labels1)
    test_labels1 = torch.LongTensor(test_labels1)

    adj1 = sparse_mx_to_torch_sparse_tensor(adj1)

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    train_edges1=torch.FloatTensor(np.array(train_edges1))
    test_edges1=torch.FloatTensor(np.array(test_edges1))

    return adj1, features, train_labels1,test_labels1,train_edges1,test_edges1


def load_data_pokec(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    f2 = open("../data/gender/pokec-adj-feat-{0}-{1}-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,1:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_ruikai(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora o
    nly for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 351
    if ii<0:
        f2 = open("../data/pokec-interintra-small-00/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    else:
        f2 = open("../data/pokec-interintra-small-11/pokec-adj-feat-{0}-{1}.pkl".format(str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,1:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test




def load_data_pokec2(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    f2 = open("../data/gender/pokec-adj-feat-{0}-{1}-gender-original.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    # print((ft))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,2:]
    # print(np.shape(features))
    features = sp.csr_matrix(features, dtype=np.float16)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_noage(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    f2 = open("../data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    # print((ft))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx=[1,3,4,5]
    features = ft[:,idx]
    # print(np.shape(features))
    features = sp.csr_matrix(features, dtype=np.float16)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_age(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    f2 = open("../data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    # print((ft))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx=[1,2,3,4,5]
    features = ft[:,idx]
    # print(np.shape(features))
    features = sp.csr_matrix(features, dtype=np.float16)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_feature_selection(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    f2 = open("../data/gender/pokec-adj-feat-{0}-{1}-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())




    feat_data = ft
    # print((ft))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,2:]
    # print(np.shape(features))
    features = sp.csr_matrix(features, dtype=np.float16)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_gender_weight(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    # f2 = open("/content/drive/My Drive/pygcn-master2/data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    f2 = open("../data/weight/pokec-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,gender_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()

    idx=[0,2,3,4,5]
    features = ft[:,idx]
    # print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test



def load_data_pokec_gender_weight_noheightweight(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    # f2 = open("/content/drive/My Drive/pygcn-master2/data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    f2 = open("../data/weight/pokec-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,gender_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()

    idx=[0,2,5]
    features = ft[:,idx]
    # print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_gender_weight_noweight(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    # f2 = open("/content/drive/My Drive/pygcn-master2/data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    f2 = open("../data/weight/pokec-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,gender_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()

    idx=[0,2,3,5]
    features = ft[:,idx]
    # print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test



def load_data_pokec_gender_weight_noheightweightage(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    # f2 = open("/content/drive/My Drive/pygcn-master2/data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    f2 = open("../data/weight/pokec-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,gender_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()

    idx=[0,5]
    features = ft[:,idx]
    # print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_gender_weight_randomregion(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    # f2 = open("/content/drive/My Drive/pygcn-master2/data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    f2 = open("../data/weight/pokec-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,gender_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()

    idx=[0,5]
    ft_tmp=[]
    # features = ft[:,idx]
    f_tmp=np.random.randint(0,2,num_nodes)
    for i in f_tmp:
        ft_tmp.append([i])

    features = np.array(ft_tmp)
    # print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test



def load_data_fb_edu_gender(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-edu-gender/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,g_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,77)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pubmed(ii,sed,path="../data/", dataset="pubmed"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    f2 = open("../data/pubmed/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(4000,6000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,4000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_gender_weight_predefense(ii,sed,ratio,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    # f2 = open("/content/drive/My Drive/pygcn-master2/data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    f2 = open("../data/weight/pokec-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)

    print(type(adj))
    adj1 = np.array(adj.todense())

    num_edges=g.number_of_edges()

    num_add = int(ratio* num_edges)

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


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,gender_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()

    idx=[0,2,3,4,5]
    features = ft[:,idx]
    # print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_gender_weight_predefense_delete(ii,sed,ratio,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    # f2 = open("/content/drive/My Drive/pygcn-master2/data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    f2 = open("../data/weight/pokec-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)

    # print(type(adj))
    adj1 = np.array(adj.todense())

    num_edges=g.number_of_edges()

    num_add = int(ratio* num_edges)

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


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,gender_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()

    idx=[0,2,3,4,5]
    features = ft[:,idx]
    # print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pubmed_predefense(ii,sed,ratio,path="../data/", dataset="pubmed"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    f2 = open("../data/pubmed/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]

    g = nx.Graph(adj)

    # print(type(adj))
    adj1 = np.array(adj.todense())

    num_edges=g.number_of_edges()

    num_add = int(ratio* num_edges)

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


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))

    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(4000,6000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,4000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test




def load_data_pokec_gender_weight_predefense_fulledges(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    # f2 = open("/content/drive/My Drive/pygcn-master2/data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    f2 = open("../data/weight/pokec-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)

    print(type(adj))
    adj1 = np.array(adj.todense())

    num_edges=g.number_of_edges()

    # num_add = int(ratio* num_edges)
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

    adj=np.ones((num_nodes,num_nodes))
    g = nx.Graph(adj)
    adj=nx.adjacency_matrix(g)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,gender_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()

    idx=[0,2,3,4,5]
    features = ft[:,idx]
    # print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


# def load_data_pokec_gender_weight_predefense_fulledges(ii,sed,path="../data/", dataset="pokec"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#
#     num_nodes = 10000
#     num_feats = 351
#     # f2 = open("/content/drive/My Drive/pygcn-master2/data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
#     f2 = open("../data/weight/pokec-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')
#
#     adj, ft = pkl.load(f2, encoding='latin1')
#     print(np.shape(ft))
#     # print(adj[0])
#     # print(adj[10000])
#     # print(adj[12470])
#     # print(adj[22469])
#
#     g = nx.Graph(adj)
#
#     print(type(adj))
#     adj1 = np.array(adj.todense())
#
#     num_edges=g.number_of_edges()
#
#     # num_add = int(ratio* num_edges)
#     #
#     # idx_adj=np.array(range(0,num_nodes*num_nodes))
#     # idx_edges=[]
#     #
#     #
#     # for eg in g.edges():
#     #     nd0=eg[0]
#     #     nd1 = eg[1]
#     #
#     #     if (nd1>nd0):
#     #         tmp=nd0
#     #         nd0=nd1
#     #         nd1=tmp
#     #
#     #     edge_idx=nd0*num_nodes+nd1
#     #     idx_edges.append(edge_idx)
#     # idx_edges=np.array(idx_edges)
#     #
#     # idx_neg_edges=np.delete(idx_adj, idx_edges)
#     #
#     # idx_add_edges = random.sample(list(idx_neg_edges), num_add)
#     #
#     # add_edges=[]
#     #
#     # for ed in idx_add_edges:
#     #     nd1=int(ed / num_nodes)
#     #     nd2=ed % num_nodes
#     #     add_edges.append([nd1,nd2])
#     #     add_edges.append([nd2, nd1])
#     # g.add_edges_from(add_edges)
#
#     adj=np.ones((num_nodes,num_nodes))
#     g = nx.Graph(adj)
#     adj=nx.adjacency_matrix(g)
#
#
#     # adj1=np.array(adj.todense())
#
#
#     feat_data = ft
#     print((np.shape(ft)))
#
#
#     gender_index = 1
#     pub_index = 0
#     age_index = 2
#     height_index = 3
#     weight_index = 4
#     region_index = 5
#
#     labels=np.array(ft)[:,gender_index]
#
#     # for i, n in enumerate(g.nodes()):
#     #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
#     #         ginfo = 1  # male
#     #         labels.append(ginfo)
#     #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
#     #         ginfo = 2  # female
#     #         labels.append(ginfo)
#     #
#     #     else:
#     #         print('***')
#     #         ginfo = 0  # unknow gender
#     #         labels.append(ginfo)
#     #
#     #     # print(ginfo)
#     #
#     #     g.nodes[n]['gender'] = ginfo
#
#     # print(feat_data)
#     # adj_lists = defaultdict(set)
#     # #g = nx.Graph(adj)
#     # adj_dense=np.array(adj.todense())
#     # #print(adj_dense)
#     #
#     #
#     # for i in range(np.shape(adj_dense)[0]):
#     #     for j in range(i,np.shape(adj_dense)[0]):
#     #         if (adj_dense[i][j]==1):
#     #             adj_lists[i].add(j)
#     #             adj_lists[j].add(i)
#
#     #labels = np.empty((num_nodes, 1), dtype=np.int64)
#
#     # with open('../data/musae_facebook-target.txt') as tfile:
#     #     Lines = tfile.readlines()
#     #     labels = []
#     #     for line in Lines:
#     #         arr = line.strip().split(',')
#     #         labels.append(int(arr[1]))
#     #
#     # print(labels)
#     # print(np.shape(labels))
#
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#     # print()
#
#     idx=[0,2,3,4,5]
#     features = ft[:,idx]
#     # print(np.shape(features))
#     # exit()
#     features = sp.csr_matrix(features, dtype=np.float32)
#
#     features = normalize(features)
#     adj = normalize(adj + sp.eye(adj.shape[0]))
#     # print(adj)
#     # print(np.shape(adj))
#
#     idx_test = range(7000,10000)
#     # idx_val = range(1800, 1977)
#     idx_train = range(0,7000)
#
#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(labels)
#
#     adj = sparse_mx_to_torch_sparse_tensor(adj)
#
#     idx_train = torch.LongTensor(idx_train)
#     # idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)
#
#     # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#     #                                     dtype=np.dtype(str))
#     # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     # labels = encode_onehot(idx_features_labels[:, -1])
#     #
#     # # build graph
#     # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     # idx_map = {j: i for i, j in enumerate(idx)}
#     # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#     #                                 dtype=np.int32)
#     # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#     #                  dtype=np.int32).reshape(edges_unordered.shape)
#     # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#     #                     shape=(labels.shape[0], labels.shape[0]),
#     #                     dtype=np.float32)
#     #
#     # # build symmetric adjacency matrix
#     # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#     #
#     # features = normalize(features)
#     # adj = normalize(adj + sp.eye(adj.shape[0]))
#     #
#     # idx_train = range(677)
#     # idx_val = range(677, 1354)
#     # idx_test = range(1354, 2708)
#     #
#     # features = torch.FloatTensor(np.array(features.todense()))
#     # labels = torch.LongTensor(np.where(labels)[1])
#     # adj = sparse_mx_to_torch_sparse_tensor(adj)
#     #
#     # idx_train = torch.LongTensor(idx_train)
#     # idx_val = torch.LongTensor(idx_val)
#     # idx_test = torch.LongTensor(idx_test)
#
#     return adj, features, labels, idx_train, idx_test
#


def load_data_fb_edu_gender_interintra(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-edu-gender/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,g_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,77)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_fb_interintra(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-interintra/fb-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)
    train_index = int(num_nodes * 0.7)

    idx_test = range(train_index, 10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)

    # adj1=np.array(adj.todense())




    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,77)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))


    num_nodes=np.shape(ft)[0]
    train_index = int(num_nodes * 0.7)

    idx_test = range(train_index, num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)

    # idx_test = range(1200,1600)
    # # idx_val = range(1800, 1977)
    # idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_fb_edu_gender2(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-edu-gender2/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,53)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec2(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    f2 = open("../data/gender/pokec-adj-feat-{0}-{1}-gender-original.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    # print((ft))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,2:]
    # print(np.shape(features))
    features = sp.csr_matrix(features, dtype=np.float16)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_pub_gender_interintra(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-gender-interintra2/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,2:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(num_nodes*0.7)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pubmed_interintra(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pubmed-interintra/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    print(np.shape(adj))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int32')


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    train_index=int(num_nodes*0.7)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_small(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 351
    f2 = open("../data/pokec-gender-small/pokec-adj-feat-{0}-{1}-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,1:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_small_new(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 351
    f2 = open("../data/pokec-small-3.16-new/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,1:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test






def load_data_fb_edu_gender_small(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-edu-gender-small/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,53)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pubmed_small(ii,sed,path="../data/", dataset="pubmed"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    f2 = open("../data/pubmed-gender-small/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_pub_gender_interintra_small(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-gender-interintra-small-size/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,1:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(1200)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_fb_interintra_small(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-interintra-small-size/fb-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)
    train_index = 1200

    idx_test = range(train_index, 1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)

    # adj1=np.array(adj.todense())




    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,53)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))


    num_nodes=np.shape(ft)[0]
    train_index = 1200

    idx_test = range(train_index, num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)

    # idx_test = range(1200,1600)
    # # idx_val = range(1800, 1977)
    # idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pubmed_interintra_small(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pubmed-interintra-small-size/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    print(np.shape(adj))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int32')


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    train_index=1200

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_small2(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 351
    f2 = open("../data/pokec-small-3.16/pokec-adj-feat-{0}-{1}-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,1:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_fb_edu_gender_small2(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-edu-gender-small/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,53)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pubmed_small2(ii,sed,path="../data/", dataset="pubmed"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    f2 = open("../data/pubmed-small-3.16/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_pub_gender_interintra_small2(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-interintra-small-3.16/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,1:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(1200)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_fb_interintra_small2(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-interintra-small-3.16/fb-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)
    train_index = 1200

    idx_test = range(train_index, 1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)

    # adj1=np.array(adj.todense())




    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,53)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))


    num_nodes=np.shape(ft)[0]
    train_index = 1200

    idx_test = range(train_index, num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)

    # idx_test = range(1200,1600)
    # # idx_val = range(1800, 1977)
    # idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test
def load_data_fb_interintra_small_ruikai(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    if ii<0:

        f2 = open("../data/fb-interintra-small-0/fb-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')

    else:
        f2 = open("../data/fb-interintra-small-1/fb-adj-feat-{0}-{1}.pkl".format(str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)
    train_index = 1200

    idx_test = range(train_index, 1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)

    # adj1=np.array(adj.todense())




    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,53)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))


    num_nodes=np.shape(ft)[0]
    train_index = 1200

    idx_test = range(train_index, num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)

    # idx_test = range(1200,1600)
    # # idx_val = range(1800, 1977)
    # idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test



def load_data_popets(dataset,g,ft,labels):
    print('Loading {} dataset...'.format(dataset))

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]

    adj=nx.adjacency_matrix(g)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()


    features = ft
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))


    num_nodes=np.shape(ft)[0]
    train_index = int(num_nodes * 0.7)

    idx_test = range(train_index, num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)


    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    return adj, features, labels, idx_train, idx_test



def load_data_pubmed_interintra_small2(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pubmed-interintra-small-3.16/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    print(np.shape(adj))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int32')


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    train_index=1200

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pubmed_small2_2classes(ii,sed,path="../data/", dataset="pubmed"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    f2 = open("../data/pubmed-small-3.16-classes/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')

    for la in range(len(labels)):
        if labels[la] == 2:
            labels[la] = 0

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pubmed_interintra_small2_2classes(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pubmed-interintra-small-3.16/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    print(np.shape(adj))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int32')

    for la in range(len(labels)):
        if labels[la] == 2:
            labels[la] = 1



    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    train_index=1200

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_fb_edu_gender_small2(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-edu-gender/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,g_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,77)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_pub_gender_interintra_small2_nogender(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-interintra-small-3.16/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features = ft[:,2:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(1200)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_pub_gender_interintra_small2_nogenderweight(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-interintra-small-3.16/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    ft_idx=[2,3,5]
    features = ft[:,ft_idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(1200)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_pub_gender_interintra_small2_nogenderweightheight(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-interintra-small-3.16/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    ft_idx=[2,5]
    features = ft[:,ft_idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(1200)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_pub_gender_interintra_small2_random(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-interintra-small-3.16/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features1 = np.random.randint(0, 2, num_nodes)
    features2 = np.random.randint(0, 2, num_nodes)
    features1.reshape(-1,1)
    features2.reshape(-1, 1)
    features=np.concatenate((features1,features2),axis=1)
    # features=features.T
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(1200)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_pub_gender_interintra_small2_identical(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-interintra-small-3.16/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    # features1 = np.random.randint(0, 2, num_nodes)
    # features2 = np.random.randint(0, 2, num_nodes)
    # features1.reshape(-1,1)
    # features2.reshape(-1, 1)
    # features=np.concatenate((features1,features2),axis=1)

    features=np.ones((num_nodes,2))

    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(1200)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_gender_weight_identical(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    # f2 = open("/content/drive/My Drive/pygcn-master2/data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    f2 = open("../data/weight/pokec-adj-feat-{0}-{1}-weight.pkl".format(str(ii),str(sed)), 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,gender_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()

    idx=[0,5]
    # ft_tmp=[]
    # # features = ft[:,idx]
    # f_tmp=np.random.randint(0,2,num_nodes)
    # for i in f_tmp:
    #     ft_tmp.append([i])
    #
    # features = np.array(ft_tmp)

    features = np.ones((num_nodes, 2))
    # print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_nogenderweight(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 351
    f2 = open("../data/pokec-small-3.16/pokec-adj-feat-{0}-{1}-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    ft_idx=[2,3,5]
    features = ft[:,ft_idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test




def load_data_pokec_nogenderweightheight(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 351
    f2 = open("../data/pokec-small-3.16/pokec-adj-feat-{0}-{1}-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    ft_idx=[2,5]
    features = ft[:,ft_idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_identical(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 351
    f2 = open("../data/pokec-small-3.16/pokec-adj-feat-{0}-{1}-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features=np.ones((num_nodes,2))
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_pubflip(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 351
    f2 = open("../data/pokec-small-3.16/pokec-adj-feat-{0}-{1}-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels1 = np.array(ft)[:, gender_index]
    lb1_idx=np.where(labels1==0)[0]
    lb2_idx = np.where(labels1 == 1)[0]

    print(lb1_idx,lb2_idx)

    lb1_idx_1 = np.array(random.sample(list(lb1_idx), int(0.5 * len(lb1_idx))))
    lb2_idx_1 = np.array(random.sample(list(lb2_idx), int(0.5 * len(lb2_idx))))



    labels=np.ones(len(labels1))

    labels[lb1_idx_1]=0
    labels[lb2_idx_1] = 0

    # labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    features=np.ones((num_nodes,2))
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_pub_gender_interintra_small2_pubflip(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-interintra-small-3.16/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels1 = np.array(ft)[:, gender_index]
    lb1_idx = np.where(labels1 == 0)[0]
    lb2_idx = np.where(labels1 == 1)[0]

    # print(lb1_idx, lb2_idx)

    lb1_idx_1 = np.array(random.sample(list(lb1_idx), int(0.5 * len(lb1_idx))))
    lb2_idx_1 = np.array(random.sample(list(lb2_idx), int(0.5 * len(lb2_idx))))

    labels = np.ones(len(labels1))

    labels[lb1_idx_1] = 0
    labels[lb2_idx_1] = 0

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    # features1 = np.random.randint(0, 2, num_nodes)
    # features2 = np.random.randint(0, 2, num_nodes)
    # features1.reshape(-1,1)
    # features2.reshape(-1, 1)
    # features=np.concatenate((features1,features2),axis=1)

    features=np.ones((num_nodes,2))

    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(1200)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_pubflip0(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 351
    f2 = open("../data/pokec-small-3.16/pokec-adj-feat-{0}-{1}-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels1 = np.array(ft)[:, gender_index]
    lb1_idx=np.where(labels1==0)[0]
    lb2_idx = np.where(labels1 == 1)[0]

    print(lb1_idx,lb2_idx)

    lb1_idx_1 = np.array(random.sample(list(lb1_idx), int(0.5 * len(lb1_idx))))
    lb2_idx_1 = np.array(random.sample(list(lb2_idx), int(0.5 * len(lb2_idx))))



    labels=np.ones(len(labels1))

    labels[lb1_idx_1]=0
    labels[lb2_idx_1] = 0

    # labels=np.array(ft)[:,pub_index]

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    # features=np.ones((num_nodes,2))
    features = feat_data[:, 1:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pokec_pub_gender_interintra_small2_pubflip0(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-interintra-small-3.16/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels1 = np.array(ft)[:, gender_index]
    lb1_idx = np.where(labels1 == 0)[0]
    lb2_idx = np.where(labels1 == 1)[0]

    # print(lb1_idx, lb2_idx)

    lb1_idx_1 = np.array(random.sample(list(lb1_idx), int(0.5 * len(lb1_idx))))
    lb2_idx_1 = np.array(random.sample(list(lb2_idx), int(0.5 * len(lb2_idx))))

    labels = np.ones(len(labels1))

    labels[lb1_idx_1] = 0
    labels[lb2_idx_1] = 0

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    # features1 = np.random.randint(0, 2, num_nodes)
    # features2 = np.random.randint(0, 2, num_nodes)
    # features1.reshape(-1,1)
    # features2.reshape(-1, 1)
    # features=np.concatenate((features1,features2),axis=1)

    # features=np.ones((num_nodes,2))

    features = feat_data[:, 1:]

    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(1200)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_pub_gender_interintra_small2_switch(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pokec-interintra-small-3.16/pokec-adj-feat-{0}-{1}-switch.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = 351

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array(ft)[:, gender_index]
    ft_idx = [1, 2, 3, 4, 5]
    #
    # for jj in range(np.shape(ft)[0]):
    #     if ft[jj][1] == 0:
    #         ft[jj][1] = 1
    #     elif ft[jj][1] == 1:
    #         ft[jj][1] = 0
    #     else:
    #         print('error')
    #
    # # feat_data = list(ft[:, ft_idx])

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    # features1 = np.random.randint(0, 2, num_nodes)
    # features2 = np.random.randint(0, 2, num_nodes)
    # features1.reshape(-1,1)
    # features2.reshape(-1, 1)
    # features=np.concatenate((features1,features2),axis=1)

    # features=np.ones((num_nodes,2))

    features = ft[:, 1:]

    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    train_index=int(1200)

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pokec_switch(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 351
    f2 = open("../data/pokec-small-3.16/pokec-adj-feat-{0}-{1}-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array(ft)[:, gender_index]
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

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    # features=np.ones((num_nodes,2))
    features = feat_data[:, 1:]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_fb_edu_gender2_switch(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-edu-gender2/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))


    for jj in range(np.shape(ft)[0]):
        if ft[jj][77] == 1:
            ft[jj][77] = 2
        elif ft[jj][77] == 2:
            ft[jj][77] = 1
        else:
            print('error')


    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,53)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_fb_interintra_small2_switch(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283
    f2 = open("../data/fb-edu-gender2/fb-adj-feat-{0}-{1}-switch.pkl".format(str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)
    train_index = 1200

    idx_test = range(train_index, 1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)

    # adj1=np.array(adj.todense())

    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,53)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))


    num_nodes=np.shape(ft)[0]
    train_index = 1200

    idx_test = range(train_index, num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0, train_index)

    # idx_test = range(1200,1600)
    # # idx_val = range(1800, 1977)
    # idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_pubmed_interintra_small2_switch(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pubmed-interintra-small-3.16/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    print(np.shape(adj))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]

    for jj in range(np.shape(ft)[0]):
        if ft[jj][162] == 0:
            ft[jj][162] = 1
        elif ft[jj][162] != 0:
            ft[jj][162] = 0
        else:
            print('error')

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int32')


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    train_index=1200

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pubmed_small2_switch(ii,sed,path="../data/", dataset="pubmed"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    f2 = open("../data/pubmed-small-3.16/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

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

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_pubmed_interintra_small2_switch2(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))



    f2 = open("../data/pubmed-interintra-small-3.16/pubmed-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    print(np.shape(adj))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])
    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]

    for jj in range(np.shape(ft)[0]):
        if ft[jj][162] == 0:
            ft[jj][162] = 1
        elif ft[jj][162] != 0:
            ft[jj][162] = 0
        else:
            print('error')

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int32')


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    features = (ft[:, idx])
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    train_index=1200

    idx_test = range(train_index,num_nodes)
    # idx_val = range(1800, 1977)
    idx_train = range(0,train_index)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test



def load_data_fb_ego_698(nd,gindex):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./698-adj-feat.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())



    ft = np.delete(ft, nd,0)

    print(np.shape(ft))
    print(np.shape(adj1))

    adj1 = np.delete(adj1, nd,0)
    adj1 = np.delete(adj1, nd,1)
    print(np.shape(adj1))

    g1 = nx.Graph(adj1)

    adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    g_index = 26  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,g_index])
    print(set(labels))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    idx=np.delete(idx,26)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g1.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train



def load_data_fb_ego_698_edge(nd,gindex):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./698-adj-feat.pkl", 'rb')
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


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    g_index = 26  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,g_index])
    print(set(labels))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    idx=np.delete(idx,26)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train


def load_data_fb_ego_698_orig(nd,gindex):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./698-adj-feat.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)
    # ed=list(g.edges())[nd]
    # e1=ed[0]
    # e2=ed[1]
    #
    # adj1=np.array(adj.todense())
    # adj1[e1][e2]=0
    # adj1[e2][e1] = 0
    adj1=adj

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    g_index = 26  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,g_index])
    print(set(labels))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    idx=np.delete(idx,26)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train

def load_data_fb_ego_698_orig_noise(nd,gindex):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./698-adj-feat.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)
    # ed=list(g.edges())[nd]
    # e1=ed[0]
    # e2=ed[1]
    #
    # adj1=np.array(adj.todense())
    # adj1[e1][e2]=0
    # adj1[e2][e1] = 0
    adj1=adj

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    g_index = 26  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,g_index])
    print(set(labels))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    idx=np.delete(idx,26)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(int(0.7*g.number_of_nodes()),g.number_of_nodes())
    # idx_val = range(1800, 1977)
    idx_train = range(0,int(0.7*g.number_of_nodes()))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train,idx_test



def load_data_pokec_ego(nd):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./pokec-ego.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())



    ft = np.delete(ft, nd,0)

    print(np.shape(ft))
    print(np.shape(adj1))

    adj1 = np.delete(adj1, nd,0)
    adj1 = np.delete(adj1, nd,1)
    print(np.shape(adj1))

    g1 = nx.Graph(adj1)

    adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    g_index = 26  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,0])
    print(set(labels))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    idx=np.delete(idx,0)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g1.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train



def load_data_pokec_ego_edge(nd):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

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


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    g_index = 26  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,0])
    print(set(labels))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    idx=np.delete(idx,0)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train


def load_data_pokec_ego_orig(nd):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./pokec-ego.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)
    # ed=list(g.edges())[nd]
    # e1=ed[0]
    # e2=ed[1]
    #
    # adj1=np.array(adj.todense())
    # adj1[e1][e2]=0
    # adj1[e2][e1] = 0
    adj1=adj

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    # g_index = 26  # gender
    # r_index = 1154  # religion
    # p_index = 1278  # political
    # e_index = 53  # education_type

    labels = np.array((ft)[:, 0])
    print(set(labels))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    idx=np.delete(idx,0)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train


def load_data_pokec_ego_orig_noise(nd):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./pokec-ego.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)
    # ed=list(g.edges())[nd]
    # e1=ed[0]
    # e2=ed[1]
    #
    # adj1=np.array(adj.todense())
    # adj1[e1][e2]=0
    # adj1[e2][e1] = 0
    adj1=adj

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    # g_index = 26  # gender
    # r_index = 1154  # religion
    # p_index = 1278  # political
    # e_index = 53  # education_type

    labels = np.array((ft)[:, 0])
    print(set(labels))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    idx=np.delete(idx,0)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    # idx_train = range(0,g.number_of_nodes())


    idx_test = range(int(0.7*g.number_of_nodes()),g.number_of_nodes())
    # idx_val = range(1800, 1977)
    idx_train = range(0,int(0.7*g.number_of_nodes()))


    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train,idx_test





def load_data_pubmed_ego(nd):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./pubmed-ego.pkl", 'rb')
    adj, ft,lb = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())



    ft = np.delete(ft, nd,0)

    print(np.shape(ft))
    print(np.shape(adj1))

    adj1 = np.delete(adj1, nd,0)
    adj1 = np.delete(adj1, nd,1)
    print(np.shape(adj1))

    g1 = nx.Graph(adj1)

    adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    g_index = 26  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels = lb-1
    print(set(labels))

    # for l in range(len(labels)):
    #     if labels[l] >= 0:
    #         labels[l] = 1


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    # idx=np.delete(idx,126)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g1.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train



def load_data_pubmed_ego_edge(nd):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

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


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    g_index = 26  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type
    labels = lb-1

    # labels = np.array((ft)[:, 126])
    print(set(labels))

    # for l in range(len(labels)):
    #     if labels[l] >= 0:
    #         labels[l] = 1

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats = np.shape(ft)[1]
    idx = np.arange(num_feats)
    # idx = np.delete(idx, 126)
    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train


def load_pubmed_ego_orig(nd):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./pubmed-ego.pkl", 'rb')
    adj, ft,lb = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)
    # ed=list(g.edges())[nd]
    # e1=ed[0]
    # e2=ed[1]
    #
    # adj1=np.array(adj.todense())
    # adj1[e1][e2]=0
    # adj1[e2][e1] = 0
    adj1=adj

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    # g_index = 26  # gender
    # r_index = 1154  # religion
    # p_index = 1278  # political
    # e_index = 53  # education_type

    labels=lb-1
    print(set(labels))

    # for l in range(len(labels)):
    #     if labels[l]>=0:
    #         labels[l]=1

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    # idx=np.delete(idx,126)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train


def load_pubmed_ego_orig(nd):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./pubmed-ego.pkl", 'rb')
    adj, ft,lb = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)
    # ed=list(g.edges())[nd]
    # e1=ed[0]
    # e2=ed[1]
    #
    # adj1=np.array(adj.todense())
    # adj1[e1][e2]=0
    # adj1[e2][e1] = 0
    adj1=adj

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    # g_index = 26  # gender
    # r_index = 1154  # religion
    # p_index = 1278  # political
    # e_index = 53  # education_type

    labels=lb-1
    print(set(labels))

    # for l in range(len(labels)):
    #     if labels[l]>=0:
    #         labels[l]=1

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    # idx=np.delete(idx,126)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train,

def load_pubmed_ego_orig_noise(nd):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./pubmed-ego.pkl", 'rb')
    adj, ft,lb = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)
    # ed=list(g.edges())[nd]
    # e1=ed[0]
    # e2=ed[1]
    #
    # adj1=np.array(adj.todense())
    # adj1[e1][e2]=0
    # adj1[e2][e1] = 0
    adj1=adj

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    # g_index = 26  # gender
    # r_index = 1154  # religion
    # p_index = 1278  # political
    # e_index = 53  # education_type

    labels=lb-1
    print(set(labels))

    # for l in range(len(labels)):
    #     if labels[l]>=0:
    #         labels[l]=1

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    # idx=np.delete(idx,126)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    # idx_train = range(0,g.number_of_nodes())
    idx_test = range(int(0.7*g.number_of_nodes()),g.number_of_nodes())
    # idx_val = range(1800, 1977)
    idx_train = range(0,int(0.7*g.number_of_nodes()))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train,idx_test



def load_data_fb_node_influence(nd,gindex,ii,sed):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("../data/fb-edu-gender2/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii), str(sed)), 'rb')

    #
    # f2 = open("./698-adj-feat.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())
    #


    ft = np.delete(ft, nd,0)

    print(np.shape(ft))
    print(np.shape(adj1))

    adj1 = np.delete(adj1, nd,0)
    adj1 = np.delete(adj1, nd,1)
    print(np.shape(adj1))

    g1 = nx.Graph(adj1)

    adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    g_index = 26  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,g_index])
    print(set(labels))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    idx=np.delete(idx,26)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g1.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train

def load_data_pokec_age_region(ii,sed,path="../data/", dataset="pokec"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 10000
    num_feats = 351
    f2 = open("../data/pokec/pokec-adj-feat-{0}-{1}.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    adj1=np.array(adj.todense())


    feat_data = ft
    # print((ft))


    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels=np.array(ft)[:,region_index]
    print(labels)

    # for i, n in enumerate(g.nodes()):
    #     if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
    #         ginfo = 1  # male
    #         labels.append(ginfo)
    #     elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
    #         ginfo = 2  # female
    #         labels.append(ginfo)
    #
    #     else:
    #         print('***')
    #         ginfo = 0  # unknow gender
    #         labels.append(ginfo)
    #
    #     # print(ginfo)
    #
    #     g.nodes[n]['gender'] = ginfo

    # print(feat_data)
    # adj_lists = defaultdict(set)
    # #g = nx.Graph(adj)
    # adj_dense=np.array(adj.todense())
    # #print(adj_dense)
    #
    #
    # for i in range(np.shape(adj_dense)[0]):
    #     for j in range(i,np.shape(adj_dense)[0]):
    #         if (adj_dense[i][j]==1):
    #             adj_lists[i].add(j)
    #             adj_lists[j].add(i)

    #labels = np.empty((num_nodes, 1), dtype=np.int64)

    # with open('../data/musae_facebook-target.txt') as tfile:
    #     Lines = tfile.readlines()
    #     labels = []
    #     for line in Lines:
    #         arr = line.strip().split(',')
    #         labels.append(int(arr[1]))
    #
    # print(labels)
    # print(np.shape(labels))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx=[0,1,2,3,4]
    features = ft[:,idx]
    # print(np.shape(features))
    features = sp.csr_matrix(features, dtype=np.float16)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(7000,10000)
    # idx_val = range(1800, 1977)
    idx_train = range(0,7000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_fb_edu_gender2_2(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-edu-gender2/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,53)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    idx_test = range(0,1600,3)
    # idx_val = range(1800, 1977)
    idx_train1 = range(1,1600,3)
    idx_train2 = range(2, 1600, 3)

    idx_train=list(idx_train1)+list(idx_train2)
    print(len(idx_train))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test

def load_data_fb_edu_gender2_3(ii,sed,path="../data/", dataset="fb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 1283

    f2 = open("../data/fb-edu-gender-2.10-small-group-diff/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii),str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)


    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))

    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels=np.array((ft)[:,e_index]-1)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx= np.arange(num_feats)
    idx=np.delete(idx,53)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(0,1600,3)
    # # idx_val = range(1800, 1977)
    # idx_train1 = range(1,1600,3)
    # idx_train2 = range(2, 1600, 3)
    #
    # idx_train=list(idx_train1)+list(idx_train2)
    # print(len(idx_train))

    idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_fb_edu_other(ii,sed,path="../data/", dataset="fb"):
    print('Loading {} dataset...'.format(dataset))

    num_nodes = 1600
    num_feats = 2

    f2 = open(
        "../data/fb-edu-others-nooverlap/fb-adj-feat-{0}-{1}-edu-gender.pkl".format(str(ii), str(sed)),
        'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))
    # print(adj[0])
    # print(adj[10000])
    # print(adj[12470])
    # print(adj[22469])

    g = nx.Graph(adj)

    # adj1=np.array(adj.todense())


    feat_data = ft
    print((np.shape(ft)))

    # g_index = 77  # gender
    # r_index = 1154  # religion
    # p_index = 1278  # political
    # e_index = 53  # education_type
    e_index=0
    other_index=1

    labels = np.array((ft)[:, other_index])

    for lb in range(len(labels)):
        if labels[lb]==2:
            labels[lb]=1
    # print(set(list(labels)))
    # print(set(list(labels[0:1200])))
    # print(set(list((ft)[:, e_index])))

    # print(len(np.where(labels==0)[0]))
    # print(len(np.where(labels == 1)[0]))
    # print(len(np.where(labels == 2)[0]))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    idx = np.arange(num_feats)
    idx = np.delete(idx, 1)

    features = ft[:, idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(0,1600,3)
    # # idx_val = range(1800, 1977)
    # idx_train1 = range(1,1600,3)
    # idx_train2 = range(2, 1600, 3)
    #
    # idx_train=list(idx_train1)+list(idx_train2)
    # print(len(idx_train))

    idx_test = range(1200, 1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0, 1200)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def load_data_fb_ego_698_orig2(nd,gindex):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # num_nodes = 1600
    # num_feats = 1283

    f2 = open("./698-adj-feat.pkl", 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')
    print(np.shape(ft))

    g = nx.Graph(adj)
    # ed=list(g.edges())[nd]
    # e1=ed[0]
    # e2=ed[1]
    #
    # adj1=np.array(adj.todense())
    # adj1[e1][e2]=0
    # adj1[e2][e1] = 0
    adj1=adj

    g1 = nx.Graph(adj1)
    adj = nx.adjacency_matrix(g1)


    #
    #
    # ft = np.delete(ft, nd,0)
    #
    # print(np.shape(ft))
    # print(np.shape(adj1))
    #
    # adj1 = np.delete(adj1, nd,0)
    # adj1 = np.delete(adj1, nd,1)
    # print(np.shape(adj1))
    #
    # g1 = nx.Graph(adj1)
    #
    # adj=nx.adjacency_matrix(g1)

    feat_data = ft
    # print((np.shape(ft)))

    g_index = 26  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 12  # education_type

    labels=np.array((ft)[:,e_index])
    print(set(labels))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print()
    num_feats=np.shape(ft)[1]
    idx= np.arange(num_feats)
    idx=np.delete(idx,12)

    features = ft[:,idx]
    print(np.shape(features))
    # exit()
    features = sp.csr_matrix(features, dtype=np.float32)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj)
    # print(np.shape(adj))

    # idx_test = range(1200,1600)
    # idx_val = range(1800, 1977)
    idx_train = range(0,g.number_of_nodes())

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
    #                                     dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    #
    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                     shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    #
    # # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #
    # features = normalize(features)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    #
    # idx_train = range(677)
    # idx_val = range(677, 1354)
    # idx_test = range(1354, 2708)
    #
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)




