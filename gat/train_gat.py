"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from gat import GAT,GAT_pia
from utils import EarlyStopping
from data_loader import *

# from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd

import pickle as pkl

from scipy.special import softmax,log_softmax

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate0(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits,e = model(features)
        logits = logits[mask]
        labels = labels[mask]

        return accuracy(logits, labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits,e = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        out = logits.cpu().detach().numpy()[:,1]

        # prob=torch.nn.Softmax(logits)
        # prob = prob.cpu().detach().numpy()

        # prob=softmax(out)
        # prob = log_softmax(out)
        prob =torch.softmax(logits,dim=1)

        prob = prob.cpu().detach().numpy()

        prob=prob[:,1]


        y_pred = indices.cpu().detach().numpy()
        y_label = labels.cpu().detach().numpy()
        acc = accuracy_score(y_label, y_pred)
        recall = recall_score(y_label, y_pred)
        precision = precision_score(y_label, y_pred)
        f1 = f1_score(y_label, y_pred)
        # auc = roc_auc_score(y_label, prob)
        auc=0
        res=[acc, recall, precision, f1, auc]
        return accuracy(logits, labels),logits,labels,res
def evaluate2(feats, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feats.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0., 1, 0)
        score = f1_score(labels.data.cpu().numpy(),
                         predict, average='micro')
        return score, loss_data.item()
def evaluate_(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits,e = model(features)
        logits = logits[mask]
        print(logits)
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        out = logits.cpu().detach().numpy()

        # prob=torch.nn.Softmax(logits)
        # print(prob)
        #
        # prob = prob.cpu().detach().numpy()


        # prob = log_softmax(out)

        prob = torch.softmax(logits,dim=1)
        prob = prob.cpu().detach().numpy()
        print(prob)



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


        return accuracy(logits, labels),logits,labels,res



def main(args,data,result_train,result_test,res_dir,Flag ):
    # load and preprocess dataset
    # if args.dataset == 'cora':
    #     data = CoraGraphDataset()
    # elif args.dataset == 'citeseer':
    #     data = CiteseerGraphDataset()
    # elif args.dataset == 'pubmed':
    #     data = PubmedGraphDataset()
    # else:
    #     raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data

    print(g)
    # exit()
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    # print(features)
    # print(type(features))
    labels = g.ndata['label']
    # print(labels)
    # print(type(labels))
    train_mask = g.ndata['train_mask']
    # print(train_mask)
    # print(type(train_mask))
    # exit()
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    test_mask2 = g.ndata['test_mask2']
    num_feats = features.shape[1]

    if args.dataset =='Facebook' or args.dataset =='Gplus' :
        n_classes = 2
    elif args.dataset =='cora':
        n_classes = 7
    elif args.dataset == 'citeseer':
        n_classes = 6
    elif args.dataset == 'lastfm':
        n_classes = 18


    n_edges = len(g.edata['weight'])
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

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT_pia(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    # print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits,embed = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])


        if args.dataset =='Facebook':
            if args.fastmode:
                val_acc = accuracy(logits[val_mask], labels[val_mask])
            else:
                val_acc = evaluate0(model, features, labels, val_mask)
                if args.early_stop:
                    if stopper.step(val_acc, model):
                        break

        else:
            if args.fastmode:
                val_acc = accuracy(logits[val_mask], labels[val_mask])
            else:
                val_acc = evaluate0(model, features, labels, val_mask)
                if args.early_stop:
                    if stopper.step(val_acc, model):
                        break

        # print()

        if epoch %20==0:


            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                  " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                  format(epoch, np.mean(dur), loss.item(), train_acc,
                         val_acc, n_edges / np.mean(dur) / 1000))

    print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    # acc = evaluate(model, features, labels, test_mask)


    if args.dataset == 'Facebook' or args.dataset == 'Gplus':

        acc1, score1, pred1, eval_train = evaluate(model, features, labels, train_mask)

        acc2, score2, pred2, eval_test = evaluate(model, features, labels, test_mask2)

    else:
        acc1, score1, pred1, eval_train = evaluate_(model, features, labels, train_mask)

        acc2, score2, pred2, eval_test = evaluate_(model, features, labels, test_mask2)

    print("Test Accuracy {:.4f}".format(acc2))

    emb_matrix1 = embed[0].cpu().detach().numpy()
    emb_matrix2 = embed[1].cpu().detach().numpy()

    output_train = score1.cpu().detach().numpy()
    output_test = score2.cpu().detach().numpy()

    result_train.append(eval_train)
    result_test.append(eval_test)

    # para= {}
    # cnt = 0
    #
    # for p in model.parameters():
    #     # print(p)
    #     p = p.detach().numpy()
    #     # print(p)
    #     para[cnt] = p
    #     cnt += 1

    # for name, param in model.named_parameters():
    #     print(name,param.size())


    # gat_layers.0.attn_l torch.Size([1, 8, 8])
    # gat_layers.0.attn_r torch.Size([1, 8, 8])
    # gat_layers.0.bias torch.Size([64])
    # gat_layers.0.fc.weight torch.Size([64, 5])
    # gat_layers.1.attn_l torch.Size([1, 8, 8])
    # gat_layers.1.attn_r torch.Size([1, 8, 8])
    # gat_layers.1.bias torch.Size([64])
    # gat_layers.1.fc.weight torch.Size([64, 64])
    # gat_layers.2.attn_l torch.Size([1, 1, 2])
    # gat_layers.2.attn_r torch.Size([1, 1, 2])
    # gat_layers.2.bias torch.Size([2])
    # gat_layers.2.fc.weight torch.Size([2, 64])


    METHOD='gat'
    savepath = res_dir + METHOD+ '-embeds1-' + Flag + '-' + str(ego_user)
    print('***')
    np.save(savepath, np.array(emb_matrix1))

    savepath = res_dir+ METHOD + '-embeds2-' + Flag + '-' + str(ego_user)
    print('***')
    np.save(savepath, np.array(emb_matrix2))

    output=np.concatenate((output_train,output_test),axis=0)

    savepath = res_dir + 'posterior-' + str(ego_user)
    print('***')
    np.save(savepath, np.array(output))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=3,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)


    G_EGO_USERS = ['Facebook', 'cora', 'lastfm','citeseer']

    G_EGO_USERS = ['Gplus']


    for ego_user in G_EGO_USERS:

        args.dataset = ego_user

        dp = 0
        sigma = 4000
        if dp == 1:
            Flag = '114-gat-' + str(ego_user) + '-' + str(dp) + '-' + str(sigma)

        else:
            Flag = '131-gat-' + str(ego_user) + '-' + str(dp)


        res_dir = '%s/'%(ego_user)

        feat_dir = '/Wang-ds/xwang193/PyGCL-main/examples/' + str(ego_user) + '-adj-feat.pkl'
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


        elif ego_user == 'Gplus':

            f = open('/Wang-ds/xwang193/PyGCL-main/examples//Gplus_feature_map.txt', 'r')
            invert_index = eval(f.readline())
            f.close()

            features = ft

            g1_index = invert_index[('gender', '1')]
            g2_index = invert_index[('gender', '2')]

            count_nodes = [0, 0]

            labels = []
            target=[]

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
                # g.nodes[n]['gender'] = ginfo

                labels.append([n,ginfo])
                target.append(ginfo)

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
        #
        # res_dir0 = '/Wang-ds/xwang193/deepwalk-master/%s/' % (dt)
        # f2 = open('%s/%s-train_test_split' % (res_dir0, dt), 'rb')
        # train_test_split = pkl.load(f2, encoding='latin1')

        # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split

        g = nx.Graph(adj)


        result_train = []
        result_test = []
        dataset = process_popets(ego_user,g,ft,labels)
        # graph = dataset
        #
        # print(graph)

        main(args,dataset,result_train,result_test,res_dir,Flag )

        # exit()
