from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import GCN_pia

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd
import pickle as pkl
import networkx as nx

# from keras.layers import Input, Dense
# from keras.models import Model

def evaluation(output, labels, name):
    preds = output.max(1)[1].type_as(labels)
    out = output.max(1)[0]
    # print(out,torch.max(output.data,1))
    # preds = torch.round(output)
    out=out.detach().numpy()
    y_pred = preds.detach().numpy()
    y_label = labels.detach().numpy()
    # print(y_pred,y_label)
    acc = accuracy_score(y_label, y_pred)
    recall = recall_score(y_label, y_pred)
    precision = precision_score(y_label, y_pred)
    f1 = f1_score(y_label, y_pred)
    auc = roc_auc_score(y_label, out)

    print('{} accuracy:'.format(name), acc,
          '{} f1:'.format(name), f1,
          '{} auc:'.format(name), auc)

    return [acc, recall, precision, f1, auc]



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.02,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# d_dir='./fb-interintra-small-ruikai'
# tp='fb-interintra-small-ruikai'
# rnd=0
test_loss_acc=[]
train_loss_acc=[]
result_train=[]
result_test=[]
G_EGO_USERS = ['Facebook', 'cora', 'lastfm','citeseer']

G_EGO_USERS = ['combined','Gplus','lastfm']
G_EGO_USERS = ['citeseer']

for ego_user in G_EGO_USERS:

    args.dataset = ego_user

    dp = 0
    sigma = 4000
    if dp == 1:
        Flag = '114-gcn-' + str(ego_user) + '-' + str(dp) + '-' + str(sigma)

    else:
        Flag = '131-gcn-' + str(ego_user) + '-' + str(dp)


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
        f = open('/data/Facebook_feature_map.txt', 'r')
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
        with open('/data/' + str(ego_user) + '-target.txt') as tfile:
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

    elif ego_user == 'Gplus':
        features = ft
        f = open('/data/Gplus_feature_map.txt', 'r')
        invert_index = eval(f.readline())
        f.close()

        g1_index = invert_index[('gender', '1')]
        g2_index = invert_index[('gender', '2')]

        count_nodes = [0, 0]

        labels = []

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

            labels.append([n, ginfo])


    elif ego_user == 'combined':
        features = ft
        print(sum(features[-1]-features[-2]))
        gindex=77
        count_nodes = [0, 0]

        labels = []

        for n in range(np.shape(features)[0]):
            ginfo=ft[n][gindex]+ft[n][gindex+1]-2


            labels.append([n, ginfo])

        idx=list(range(np.shape(features)[1]))
        idx.remove(gindex)
        idx.remove(gindex+1)

        idx=np.array(idx)

        features=features[:,idx]
        ft = ft[:, idx]

    labels=np.array(labels)
    if ego_user == 'citeseer':
        idx_100 = np.where(labels[:, 1] == 100)[0]
        labels[:, 1][idx_100] = 6
    print(set(list(labels[:, 1])))

    dt=ego_user

    labels=np.array(labels)

    dt=ego_user

    # res_dir0 = '/Wang-ds/xwang193/deepwalk-master/%s/' % (dt)
    # f2 = open('%s/%s-train_test_split' % (res_dir0, dt), 'rb')
    # train_test_split = pkl.load(f2, encoding='latin1')
    #
    # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split

    g = nx.Graph(adj)

    # Load data
    adj, features, labels, idx_train, idx_test = load_data_popets(ego_user,g,ft,labels[:,1])
    # Model and optimizer
    model = GCN_pia(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        # idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output,embed1,embed2 = model(features, adj)
        # print(output[idx_train])
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        # print(labels[idx_train])
        # exit()
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output,embed1,embed2 = model(features, adj)

        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        if (epoch+1) % 100==0:
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  # 'loss_val: {:.4f}'.format(loss_val.item()),
                  # 'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))
        return loss_train.item(),acc_train.item(),output,embed1,embed2


    def test():
        model.eval()
        para={}
        cnt=0
        for p in model.parameters():
            # print(p)
            p = p.detach().numpy()
            # print(p)
            para[cnt]=p
            cnt+=1

        output,embed1,embed2 = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return output,embed1,embed2,loss_test.item(),acc_test.item(),para

    def save_model(net,seed):
        PATH = './fb-adj-feat-train-{}_net-12-13-edu-gender.pth'.format(seed)
        torch.save(net.state_dict(), PATH)



    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        loss_train, acc_train,output_train,embed1,embed2=train(epoch)
    train_loss_acc.append([loss_train, acc_train])

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print(output_train)
    # print(np.shape(output_train))
    # Testing
    output_test,embed1,embed2,loss_test, acc_test,para=test()
    # save_model(model,seed)
    test_loss_acc.append([loss_test, acc_test])

    # eval_train = evaluation(output_train[idx_train], labels[idx_train],str(sed)+'-'+str(ii))
    # eval_test = evaluation(output_test[idx_test], labels[idx_test],str(sed)+'-'+str(ii))


    # result_train.append(eval_train)
    # result_test.append(eval_test)


    emb_matrix1=embed1.detach().numpy()
    emb_matrix2 = embed2.detach().numpy()
    # emb_matrix3 = embed3.detach().numpy()
    # print(emb_matrix)
    # print(np.shape(emb_matrix))
    output_train = output_train.detach().numpy()
    output_test = output_test.detach().numpy()

    # with open("{}/pokec-para-{}-{}-12-13-{}.pkl".format(d_dir,str(ii), str(sed),tp), "wb") as f:
    #     pickle.dump(para, f)


    METHOD='gcn'
    savepath = res_dir + METHOD+ '-embeds1-' + Flag + '-' + str(ego_user)+'-layer2'
    print('***')
    np.save(savepath, np.array(emb_matrix1))

    savepath = res_dir+ METHOD + '-embeds2-' + Flag + '-' + str(ego_user)+'-layer2'
    print('***')
    np.save(savepath, np.array(emb_matrix2))

    # savepath = res_dir+ METHOD + '-embeds3-' + Flag + '-' + str(ego_user)
    # print('***')
    # np.save(savepath, np.array(emb_matrix2))

    # output=np.concatenate((output_train,output_test),axis=0)

    savepath = res_dir + 'posterior-' + str(ego_user)+'-layer2'
    print('***')
    np.save(savepath, np.array(output_test))


