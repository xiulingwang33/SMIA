"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from data_loader import *
from utils import *
import pandas as pd

import pickle

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
class GraphSAGE_pia(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE_pia, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        embed=[]
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                embed.append(h)
                h = self.activation(h)
                h = self.dropout(h)
        return h,embed


def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits,e = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        # out = scores.cpu().detach().numpy()

        prob = torch.softmax(logits, dim=1)

        prob = prob.cpu().detach().numpy()

        y_pred = indices.cpu().detach().numpy()
        y_label = labels.cpu().detach().numpy()
        acc = accuracy_score(y_label, y_pred)
        recall = recall_score(y_label, y_pred)
        precision = precision_score(y_label, y_pred)
        f1 = f1_score(y_label, y_pred)
        # auc = roc_auc_score(y_label, out)

        auc = 0
        res=[acc, recall, precision, f1, auc]

        return correct.item() * 1.0 / len(labels),logits,labels,res


def evaluate_(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits,e = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        # out = scores.cpu().detach().numpy()

        prob = torch.softmax(logits, dim=1)

        prob = prob.cpu().detach().numpy()

        y_pred = indices.cpu().detach().numpy()
        y_label = labels.cpu().detach().numpy()
        acc = accuracy_score(y_label, y_pred)

        print(y_label,set(y_label))

        recall = recall_score(y_label, y_pred, average='macro')
        precision = precision_score(y_label, y_pred, average='macro')
        f1 = f1_score(y_label, y_pred, average='macro')
        # auc = roc_auc_score(y_label, prob, average='macro', multi_class="ovr")
        auc=0

        res=[acc, recall, precision, f1, auc]


        return correct.item() * 1.0 / len(labels),logits,labels,res



def evaluate0(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits,e = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        scores, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        out = scores.cpu().detach().numpy()
        y_pred = indices.cpu().detach().numpy()
        y_label = labels.cpu().detach().numpy()
        acc = accuracy_score(y_label, y_pred)

        return acc


def main(args,data,result_train,result_test,res_dir,Flag ):
    # load and preprocess dataset
    # data = load_data(args)
    # g = data[0]

    g = data
    features = g.ndata['feat']
    print(np.shape(features))
    labels = g.ndata['label']
    print(np.shape(labels))
    print(labels.int().sum().item())
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    test_mask2 = g.ndata['test_mask2']
    in_feats = features.shape[1]
    # n_classes = data.num_classes
    # n_edges = data.graph.number_of_edges()
    num_feats = features.shape[1]
    if args.dataset =='Facebook':
        n_classes = 2
    elif args.dataset =='cora':
        n_classes = 7
    elif args.dataset == 'citeseer':
        n_classes = 7
    elif args.dataset == 'lastfm':
        n_classes = 18

    n_edges=len(g.edata['weight'])
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    test_nid2 = test_mask2.nonzero().squeeze()

    # graph preprocess and calculate normalization factor
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()

    if args.early_stop:
        stopper = EarlyStopping(patience=100)

    if cuda:
        g = g.int().to(args.gpu)

    # create GraphSAGE model
    model = GraphSAGE_pia(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits,embed = model(g, features)
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)



        acc= evaluate0(model, g, features, labels, val_nid)

        if epoch % 20 == 0:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                acc, n_edges / np.mean(dur) / 1000))

        if args.early_stop:
            if stopper.step(acc, model):
                break

    # print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    # acc = evaluate(model, features, labels, test_mask)
    # print("Test Accuracy {:.4f}".format(acc))


    if args.dataset == 'Facebook':

        acc1, score1, pred1, eval_train = evaluate(model, g,features, labels, train_mask)

        acc2, score2, pred2, eval_test = evaluate(model, g,features, labels, test_mask2)

    else:
        acc1, score1, pred1, eval_train = evaluate_(model, g,features, labels, train_mask)

        acc2, score2, pred2, eval_test = evaluate_(model, g,features, labels, test_mask2)

    # print(ii,sed)
    print("Test Accuracy {:.4f}".format(acc2))


    emb_matrix1 = embed[0].cpu().detach().numpy()
    emb_matrix2 = embed[1].cpu().detach().numpy()

    output_train = score1.cpu().detach().numpy()
    output_test = score2.cpu().detach().numpy()

    result_train.append(eval_train)
    result_test.append(eval_test)

    para= {}
    cnt = 0

    # for p in model.parameters():
    #     # print(p)
    #     p = p.detach().numpy()
    #     # print(p)
    #     para[cnt] = p
    #     cnt += 1

    # print(embed,emb_matrix1,emb_matrix2)
    # print(np.shape(emb_matrix1), np.shape(emb_matrix2))

    # for name, param in model.named_parameters():
    #     print(name,param.size())

    # layers.0.bias torch.Size([16])
    # layers.0.fc_neigh.weight torch.Size([16, 5])
    # layers.1.bias torch.Size([16])
    # layers.1.fc_neigh.weight torch.Size([16, 16])
    # layers.2.bias torch.Size([2])
    # layers.2.fc_neigh.weigh torch.Size([2, 16])
    METHOD = 'sage'
    savepath = res_dir + METHOD + '-embeds1-' + Flag + '-' + str(ego_user)
    print('***')
    np.save(savepath, np.array(emb_matrix1))

    savepath = res_dir + METHOD + '-embeds2-' + Flag + '-' + str(ego_user)
    print('***')
    np.save(savepath, np.array(emb_matrix2))

    output = np.concatenate((output_train, output_test), axis=0)

    savepath = res_dir + 'posterior-' + str(ego_user)
    print('***')
    np.save(savepath, np.array(output))



    # rnd += 1
    #
    # if (rnd + 1) % 100 == 0:
    #     data1 = pd.DataFrame(result_train)
    #     data1.to_csv('./fb-edu-gender/result_train-fb-edu-gender-%s.csv' % (rnd))
    #
    #     data2 = pd.DataFrame(result_test)
    #     data2.to_csv('./fb-edu-gender/result_test-fb-edu-gender-%s.csv' % (rnd))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=2000,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    args = parser.parse_args()
    print(args)


    G_EGO_USERS = ['Facebook', 'cora', 'lastfm','citeseer']

    G_EGO_USERS = ['citeseer']


    for ego_user in G_EGO_USERS:

        args.dataset = ego_user

        dp = 0
        sigma = 4000
        if dp == 1:
            Flag = '114-sage-' + str(ego_user) + '-' + str(dp) + '-' + str(sigma)

        else:
            Flag = '131-sage-' + str(ego_user) + '-' + str(dp)


        res_dir = '%s/'%(ego_user)

        feat_dir = '/Wang-ds/xwang193/deepwalk-master//data/' + str(ego_user) + '-adj-feat.pkl'
        # feat_dir = '/Users/xiulingwang/Downloads/facebook-data/data/' + str(ego_user) + '-adj-feat.pkl'
        # feat_dir = 'E:/python/banlance/code/google+-raw-data-3/gplus-processed-test1/' + str(ego_user) + '-adj-feat.pkl'
        # feat_dir = 'E:/python/banlance/code/dblp-data/data/' + str(ego_user) + '-adj-feat.pkl'
        f2 = open(feat_dir, 'rb')

        adj, ft = pkl.load(f2, encoding='latin1')
        # adj, ft = pk.load(f2)
        # print(adj)
        # print(np.shape(adj))
        # print(ft)
        # print(np.shape(ft))


        g = nx.Graph(adj)
        # print(g.nodes)
        # print(g.edges)

        labels = []

        if ego_user == 'Facebook':
            f = open('/Wang-ds/xwang193/deepwalk-master//data/Facebook_feature_map.txt', 'r')
            invert_index = eval(f.readline())
            f.close()

            g1_index = invert_index[('gender', 77)]
            g2_index = invert_index[('gender', 78)]

            count_nodes = [0, 0]


            features = ft

            for n in range(np.shape(features)[0]):
                if features[n][g1_index] == 1:
                    ginfo = 0
                    count_nodes[0] += 1
                elif features[n][g2_index] == 1:
                    ginfo = 1
                    count_nodes[1] += 1
                else:
                    print('error')
                    print(features[n][g1_index], features[n][g2_index])
                    exit()
                g.nodes[n]['gender'] = ginfo

                labels.append([n, ginfo])

            num_feats = np.shape(ft)[1]
            idx = np.arange(num_feats)
            idx = np.delete(idx, g2_index)
            idx = np.delete(idx, g1_index)
            feat_data = ft[:, idx]

            ft=feat_data


        elif ego_user == 'cora' or ego_user == 'citeseer' or ego_user == 'lastfm':
            with open('/Wang-ds/xwang193/deepwalk-master//data/' + str(ego_user) + '-target.txt') as tfile:
                Lines = tfile.readlines()
                target = []
                for line in Lines:
                    arr = line.strip().split(',')
                    target.append(int(arr[1]))

            for n in range(g.number_of_nodes()):
                # print(n)
                g.nodes[n]['gender'] = target[n]
                labels.append([n,target[n]])

        elif ego_user == 'dblp':
            gindex = 0
            for i, n in enumerate(g.nodes()):
                if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
                    ginfo = 1  # male
                elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
                    ginfo = 2  # female

                else:
                    print('***')
                    ginfo = 0  # unknow gender

                # print(ginfo)

                g.nodes[n]['gender'] = ginfo
                labels.append([n, ginfo])

        labels=np.array(labels)

        dt=ego_user

        res_dir0 = '/Wang-ds/xwang193/deepwalk-master/%s/' % (dt)
        f2 = open('%s/%s-train_test_split' % (res_dir0, dt), 'rb')
        train_test_split = pkl.load(f2, encoding='latin1')

        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split

        g = nx.Graph(adj_train)


        result_train = []
        result_test = []
        dataset = process_popets(ego_user,g,ft,labels)
        # graph = dataset
        #
        # print(graph)

        main(args,dataset,result_train,result_test,res_dir,Flag )