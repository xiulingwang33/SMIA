from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
# from models import GCN_pmia2

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd
import pickle as pkl
import networkx as nx

import itertools

import random

import matplotlib.pyplot as plt

import sparse
import copy

# from keras.layers import Input, Dense
# from keras.models import Model
def sigmoid(x):
   if x>=0: #对sigmoid函数优化，避免出现极大的数据溢出
       return 1.0 / (1 + np.exp(-x))
   else:
       return np.exp(x)/(1+np.exp(x))


def add_laplace_noise(data_list, b,ratio,indicex,u=0):
    data_list=data_list[:,indicex]
    emb_dim=np.shape(data_list)[1]
    data_list_unimportant =data_list[:,int(ratio*emb_dim):]

    laplace_noise = np.random.laplace(u, b, np.shape(data_list_unimportant ))
    data_list_unimportant_=laplace_noise + data_list_unimportant

    data=np.concatenate((data_list[:,0:int(ratio*emb_dim)],data_list_unimportant_),axis=1)

    return data

def get_edge_embeddings(edge_list, emb_matrixs):
    embs = []
    i = 0
    for edge in edge_list:
        node1 = int(edge[0])
        node2 = int(edge[1])
        emb = []
        # print(i)
        # print(idx_epoches_all[i,:])
        # print(len(idx_epoches_all[i,:]))

        emb1 = emb_matrixs[node1]
        # print(np.shape(emb1))
        emb2 = emb_matrixs[node2]
        edge_emb = np.multiply(emb1, emb2)
        sim1 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 0.0000000000000000000000000000001)

        sim2 = np.dot(emb1, emb2)

        sim3 = np.linalg.norm(np.array(emb1) - np.array(emb2))

        # edge_emb = np.array(emb1) + np.array(emb2)
        # print(np.shape(edge_emb))
        # emb.append(sim1)
        # emb.append(sim2)
        i += 1
        embs.append([sim1, sim2, sim3])
    embs = np.array(embs)
    return embs

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
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
G_EGO_USERS = ['Facebook', 'cora', 'lastfm','citeseer','Gplus','combined']

G_EGO_USERS = ['citeseer','combined','Gplus','lastfm']


sigmas =[0.1,0.5,1,5,10,50]
# sigmas =[0]
ratios=[0.2,0.4,0.6,0.8]

mes=['pert','rf','shap']
targets=['4cliques']

for me in mes:
    for target in targets:

        for ratio in ratios:

            results_all=[]
            for ego_user in G_EGO_USERS:
                results_att = []
                args.dataset = ego_user

                dp = 0
                sigma0 = 1
                if dp == 1:
                    Flag = '114-gcn-' + str(ego_user) + '-' + str(dp) + '-' + str(sigma0)

                else:
                    Flag = '131-gcn-' + str(ego_user) + '-' + str(dp)


                res_dir = '%s/'%(ego_user)
                res_dir0 = '/Wang-ds/xwang193/pygcn-popets-triangle/%s/' % (ego_user)

                feat_dir = '/Wang-ds/xwang193/deepwalk-master/data/' + str(ego_user) + '-adj-feat.pkl'
                # feat_dir = '/Users/xiulingwang/Downloads/facebook-data/data/' + str(ego_user) + '-adj-feat.pkl'
                # feat_dir = 'E:/python/banlance/code/google+-raw-data-3/gplus-processed-test1/' + str(ego_user) + '-adj-feat.pkl'
                # feat_dir = 'E:/python/banlance/code/dblp-data/data/' + str(ego_user) + '-adj-feat.pkl'
                f2 = open(feat_dir, 'rb')

                adj_orig, ft = pkl.load(f2, encoding='latin1')

                g = nx.Graph(adj_orig)

                METHOD = 'gcn'
                savepath = res_dir0 + METHOD + '-disconnected_edges_sampled-4clique' + Flag + '-' + str(
                    ego_user) + '-new-629-lap-multi.npy'
                print('***')
                disconnected_edges_sampled = np.load(savepath)

                METHOD = 'gcn'
                savepath = res_dir0 + METHOD + '-pos_sampled-4clique' + Flag + '-' + str(
                    ego_user) + '-new-629-lap-multi.npy'
                print('***')
                cliqs = np.load(savepath)

                num_triad = np.shape(cliqs)[0]

                idx = num_triad

                # savepath = res_dir + 'posterior-' + str(ego_user)+'-layer2.npy'
                # output_test=np.load(savepath)
                #
                # output_test =np.exp(output_test)

                # METHOD = 'gcn'
                # savepath = res_dir + METHOD + '-embed-' + Flag + '-' + str(ego_user) + '-new-629-'+me+'.npy'
                # print('***')
                # emb_matrix0 = np.load(savepath)
                #
                # emb_matrix2=emb_matrix0

                for b in sigmas:

                    sigma = b

                    dp = 7
                    # sigma = 50
                    if dp == 7:
                        Flag = '114-sage-' + str(ego_user) + '-' + str(dp) + '-' + str(b) + '-' + str(ratio)

                    else:
                        Flag = '131-sage-' + str(ego_user) + '-' + str(dp)

                    METHOD = 'sage'
                    savepath = res_dir + METHOD + '-embed-' + Flag + '-' + str(ego_user) + '-new-629-' + me + '.npy'
                    print('***')
                    emb_matrix0 = np.load(savepath)

                    emb_matrix2 = emb_matrix0

                    mem = []
                    for path in cliqs:
                        prob0 = []
                        prob1 = []
                        prob2 = []
                        prob3 = []
                        nd1_ = path[0]
                        nd2_ = path[1]
                        nd3_ = path[2]
                        nd4_ = path[3]

                        emb1 = emb_matrix2[nd1_]
                        emb2 = emb_matrix2[nd2_]
                        emb3 = emb_matrix2[nd3_]
                        emb4 = emb_matrix2[nd4_]

                        embs = [emb1, emb2, emb3, emb4]

                        for ii in range(4):
                            for jj in range(ii + 1, 4):
                                emb1 = embs[ii]
                                emb2 = embs[jj]

                                prob0.append(np.dot(emb1, emb2))
                                prob1.append(
                                    np.dot(emb1, emb2) / (
                                        np.linalg.norm(emb1) * np.linalg.norm(
                                            emb2) + 0.0000000000000000000000000000001))
                                prob2.append(np.linalg.norm(np.array(emb1) - np.array(emb2)))
                                prob3.append(sigmoid(np.dot(emb1, emb2)))

                        prob0.sort()
                        prob1.sort()
                        prob2.sort()
                        prob3.sort()

                        prob = [prob0, prob1, prob2, prob3]
                        prob = np.array(list(itertools.chain.from_iterable(prob)))
                        mem.append(prob)
                    # print(mem,np.shape(mem))

                    # print(mem)

                    non_mem = []
                    for path in disconnected_edges_sampled:
                        prob0 = []
                        prob1 = []
                        prob2 = []
                        prob3 = []
                        nd1_ = path[0]
                        nd2_ = path[1]
                        nd3_ = path[2]
                        nd4_ = path[3]
                        emb1 = emb_matrix2[nd1_]
                        emb2 = emb_matrix2[nd2_]
                        emb3 = emb_matrix2[nd3_]
                        emb4 = emb_matrix2[nd4_]

                        embs = [emb1, emb2, emb3, emb4]

                        for ii in range(4):
                            for jj in range(ii + 1, 4):
                                emb1 = embs[ii]
                                emb2 = embs[jj]

                                prob0.append(np.dot(emb1, emb2))
                                prob1.append(
                                    np.dot(emb1, emb2) / (
                                        np.linalg.norm(emb1) * np.linalg.norm(
                                            emb2) + 0.0000000000000000000000000000001))
                                prob2.append(np.linalg.norm(np.array(emb1) - np.array(emb2)))
                                prob3.append(sigmoid(np.dot(emb1, emb2)))

                        prob0.sort()
                        prob1.sort()
                        prob2.sort()
                        prob3.sort()

                        prob = [prob0, prob1, prob2, prob3]
                        prob = np.array(list(itertools.chain.from_iterable(prob)))
                        non_mem.append(prob)

                    # non_mem_class=np.array([0]*int(num_triad/6)+[1]*int(num_triad/6)+[21]*int(num_triad/12)+[22]*int(num_triad/12)+[31]*int(num_triad/18)+[32]*int(num_triad/18)+[33]*int(num_triad/18)+[41]*int(num_triad/12)+[42]*int(num_triad/12)+[51]*int(num_triad/12)+[52]*int(num_triad/12))
                    non_mem_class = np.array(
                        [0] * idx + [1] * idx + [2] * idx + [3] * idx + [4] * idx + [5] * idx + [6] * idx + [
                            7] * idx + [8] * idx + [9] * idx)

                    print('@@@', len(non_mem_class), np.shape(disconnected_edges_sampled)[0])

                    y_label_neg = np.concatenate(
                        (non_mem_class.reshape(-1, 1), np.zeros((np.shape(disconnected_edges_sampled)[0], 2)),
                         disconnected_edges_sampled), axis=1)

                    # print(pos_sampled,neg_sampled)

                    print(np.shape(mem)[0], num_triad)

                    mem_idx = list(range(np.shape(mem)[0]))
                    mem_idx_sample = np.array(random.sample(mem_idx, num_triad))

                    pos_sampled = np.array(mem)[mem_idx_sample]
                    neg_sampled = np.array(non_mem)

                    print(np.shape(pos_sampled), np.shape(neg_sampled))

                    mem_class = np.array([12] * np.shape(pos_sampled)[0])

                    y_label_pos = np.concatenate(
                        (mem_class.reshape(-1, 1), np.ones((len(mem_class), 2)), np.array(cliqs)[mem_idx_sample]),
                        axis=1)

                    # y_label_pos=np.ones((np.shape(pos_sampled)[0],3))
                    # y_label_neg=np.zeros((np.shape(neg_sampled)[0],3))
                    neg_sampled0 = neg_sampled[0:int(num_triad)]
                    neg_sampled1 = neg_sampled[int(num_triad):int(2 * num_triad)]
                    neg_sampled2 = neg_sampled[int(2 * num_triad):int(3 * num_triad)]
                    neg_sampled3 = neg_sampled[int(3 * num_triad):int(4 * num_triad)]
                    neg_sampled4 = neg_sampled[int(4 * num_triad):int(5 * num_triad)]
                    neg_sampled5 = neg_sampled[int(5 * num_triad):int(6 * num_triad)]
                    neg_sampled6 = neg_sampled[int(6 * num_triad):int(7 * num_triad)]
                    neg_sampled7 = neg_sampled[int(7 * num_triad):int(8 * num_triad)]
                    neg_sampled8 = neg_sampled[int(8 * num_triad):int(9 * num_triad)]
                    neg_sampled9 = neg_sampled[int(9 * num_triad):int(10 * num_triad)]

                    y_label_neg0 = y_label_neg[0:int(num_triad)]
                    y_label_neg1 = y_label_neg[int(num_triad):int(2 * num_triad)]
                    y_label_neg2 = y_label_neg[int(2 * num_triad):int(3 * num_triad)]
                    y_label_neg3 = y_label_neg[int(3 * num_triad):int(4 * num_triad)]
                    y_label_neg4 = y_label_neg[int(4 * num_triad):int(5 * num_triad)]
                    y_label_neg5 = y_label_neg[int(5 * num_triad):int(6 * num_triad)]
                    y_label_neg6 = y_label_neg[int(6 * num_triad):int(7 * num_triad)]
                    y_label_neg7 = y_label_neg[int(7 * num_triad):int(8 * num_triad)]
                    y_label_neg8 = y_label_neg[int(8 * num_triad):int(9 * num_triad)]
                    y_label_neg9 = y_label_neg[int(9 * num_triad):int(10 * num_triad)]

                    y_label_pos = np.concatenate(
                        (mem_class.reshape(-1, 1), np.ones((len(mem_class), 2)), np.array(cliqs)[mem_idx_sample]),
                        axis=1)

                    # y_label_pos=np.ones((np.shape(pos_sampled)[0],3))
                    # y_label_neg=np.zeros((np.shape(neg_sampled)[0],3))

                    from sklearn.model_selection import train_test_split

                    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(pos_sampled,
                                                                                                y_label_pos,
                                                                                                test_size=0.3,
                                                                                                random_state=42)

                    X_test_train0, X_test_test0, y_test_train0, y_test_test0 = train_test_split(neg_sampled0,
                                                                                                y_label_neg0,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train1, X_test_test1, y_test_train1, y_test_test1 = train_test_split(neg_sampled1,
                                                                                                y_label_neg1,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train2, X_test_test2, y_test_train2, y_test_test2 = train_test_split(neg_sampled2,
                                                                                                y_label_neg2,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train3, X_test_test3, y_test_train3, y_test_test3 = train_test_split(neg_sampled3,
                                                                                                y_label_neg3,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train4, X_test_test4, y_test_train4, y_test_test4 = train_test_split(neg_sampled4,
                                                                                                y_label_neg4,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train5, X_test_test5, y_test_train5, y_test_test5 = train_test_split(neg_sampled5,
                                                                                                y_label_neg5,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train6, X_test_test6, y_test_train6, y_test_test6 = train_test_split(neg_sampled6,
                                                                                                y_label_neg6,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train7, X_test_test7, y_test_train7, y_test_test7 = train_test_split(neg_sampled7,
                                                                                                y_label_neg7,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train8, X_test_test8, y_test_train8, y_test_test8 = train_test_split(neg_sampled8,
                                                                                                y_label_neg8,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train9, X_test_test9, y_test_train9, y_test_test9 = train_test_split(neg_sampled9,
                                                                                                y_label_neg9,
                                                                                                test_size=0.3,
                                                                                                random_state=42)

                    X_train = np.concatenate((X_train_train, X_test_train0, X_test_train1, X_test_train2, X_test_train3,
                                              X_test_train4, X_test_train5, X_test_train6, X_test_train7, X_test_train8,
                                              X_test_train9), axis=0)
                    X_test = np.concatenate((X_train_test, X_test_test0, X_test_test1, X_test_test2, X_test_test3,
                                             X_test_test4, X_test_test5, X_test_test6, X_test_test7, X_test_test8,
                                             X_test_test9), axis=0)
                    y_train = np.concatenate((y_train_train, y_test_train0, y_test_train1, y_test_train2, y_test_train3,
                                              y_test_train4, y_test_train5, y_test_train6, y_test_train7, y_test_train8,
                                              y_test_train9), axis=0)
                    y_test = np.concatenate((y_train_test, y_test_test0, y_test_test1, y_test_test2, y_test_test3,
                                             y_test_test4, y_test_test5, y_test_test6, y_test_test7, y_test_test8,
                                             y_test_test9), axis=0)
                    # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
                    #
                    # # ######################################################################

                    from sklearn import metrics
                    from sklearn.neural_network import MLPClassifier

                    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
                                        max_iter=500)

                    mlp.fit(X_train, y_train[:, 0])

                    with open("{}/mlp_{}-w-new-629-imp-{}-{}-{}-multi.pkl".format(res_dir,target,sigma,me,ratio), "wb") as f2:
                        pkl.dump(mlp, f2)

                    print("Training set score: %f" % mlp.score(X_train, y_train[:, 0]))
                    print("Test set score: %f" % mlp.score(X_test, y_test[:, 0]))

                    y_score = mlp.predict(X_test)
                    print(metrics.f1_score(y_test[:, 0], y_score, average='micro'))
                    print(metrics.classification_report(y_test[:, 0], y_score, labels=range(3)))

                    proba0 = mlp.predict_proba(X_test)
                    # proba = np.amax(proba, axis=1)
                    proba = proba0[:, 1]

                    acc_mlp_sim = accuracy_score(y_score, y_test[:, 0])

                    y_label_test = y_test

                    print(max(y_label_test[:, 0]),np.shape(proba0))

                    acc = accuracy_score(y_label_test[:, 0], y_score)
                    recall = recall_score(y_score, y_label_test[:, 0], average='macro')
                    precision = precision_score(y_score, y_label_test[:, 0], average='macro')
                    f1 = f1_score(y_score, y_label_test[:, 0], average='macro')
                    auc = roc_auc_score(y_label_test[:, 0], proba0, average='macro', multi_class="ovr")

                    print(args.dataset, acc, recall, precision, f1, auc)
                    results_att.append([args.dataset, acc, recall, precision, f1, auc])
                    results_all.append([args.dataset, b, ratio, acc, recall, precision, f1, auc])
                    tsts = []
                    for i in range(len(y_score)):
                        node1 = y_test[i][0]
                        node2 = y_test[i][1]
                        # dgr1 = g.degree(node1)
                        # dgr2 = g.degree(node2)
                        #
                        # gender1 = g.nodes[node1]['gender']
                        # gender2 = g.nodes[node2]['gender']

                        tst = [y_score[i], proba0[i], y_test[i][2], y_test[i][0], y_test[i][1]]
                        tsts.append(tst)

                    name = ['y_score', 'prob', 'y_test_grd', 'y_test_class', '0']
                    result = pd.DataFrame(columns=name, data=tsts)
                    result.to_csv("{}/mlp_sim-{}-w-new-629-imp-{}-{}-{}-multi.csv".format(res_dir,target,sigma,me,ratio))

                    # y_0 = np.where(y_test[:, 0] == 0)[0]
                    # y_1 = np.where(y_test[:, 0] == 1)[0]
                    # y_2 = np.where(y_test[:, 0] == 2)[0]
                    #
                    # acc = accuracy_score(y_label_test[:, 2][y_0], y_score[y_0])
                    # # recall = recall_score(y_score[y_0], y_label_test[:, 2][y_0])
                    # # precision = precision_score(y_score[y_0], y_label_test[:, 2][y_0])
                    # # f1 = f1_score(y_score[y_0], y_label_test[:, 2][y_0])
                    # # auc = roc_auc_score(y_label_test[:, 2][y_0], proba[y_0])
                    #
                    # print(acc, recall, precision, f1)
                    # results_att.append([args.dataset, b,ratio,acc, recall, precision, f1, auc])
                    #
                    # results_all.append([args.dataset, b, ratio, acc, recall, precision, f1, auc])
                    #
                    # acc = accuracy_score(y_label_test[:, 2][y_1], y_score[y_1])
                    # # recall = recall_score(y_score[y_1], y_label_test[:, 2][y_1])
                    # # precision = precision_score(y_score[y_1], y_label_test[:, 2][y_1])
                    # # f1 = f1_score(y_score[y_1], y_label_test[:, 2][y_1])
                    # # auc = roc_auc_score(y_label_test[:, 2][y_1], proba[y_1])
                    #
                    # print(acc, recall, precision, f1)
                    # results_att.append([args.dataset, b,ratio,acc, recall, precision, f1, auc])
                    #
                    # results_all.append([args.dataset, b, ratio, acc, recall, precision, f1, auc])
                    #
                    # acc = accuracy_score(y_label_test[:, 2][y_2], y_score[y_2])
                    # # recall = recall_score(y_score[y_2], y_label_test[:, 2][y_2])
                    # # precision = precision_score(y_score[y_2], y_label_test[:, 2][y_2])
                    # # f1 = f1_score(y_score[y_2], y_label_test[:, 2][y_2])
                    # # auc = roc_auc_score(y_label_test[:, 2][y_2], proba[y_2])
                    #
                    # print(acc, recall, precision, f1)
                    # results_att.append([args.dataset, b,ratio,acc, recall, precision, f1, auc])
                    #
                    # results_all.append([args.dataset, b, ratio, acc, recall, precision, f1, auc])
                    #
                    # name = ['ego_user','b','ratio', 'acc', 'recall', 'precision', 'f1', 'auc']
                    # result = pd.DataFrame(columns=name, data=results_att)
                    # result.to_csv("{}/pmia-results-triangle-new-629-imp-pert-{}-{}.csv".format(res_dir,sigma,ratio))


                name = ['ego_user','b','ratio', 'acc', 'recall', 'precision', 'f1', 'auc']
                result = pd.DataFrame(columns=name, data=results_all)
                result.to_csv("{}/pmia-results-{}-new-629-imp-{}-{}-multi-w.csv".format(res_dir,target,me,ratio))

for me in mes:
    for target in targets:
        for ratio in ratios:

            results_all=[]
            for ego_user in G_EGO_USERS:
                results_att = []
                args.dataset = ego_user

                dp = 0
                sigma0 = 1
                if dp == 1:
                    Flag = '114-gcn-' + str(ego_user) + '-' + str(dp) + '-' + str(sigma0)

                else:
                    Flag = '131-gcn-' + str(ego_user) + '-' + str(dp)


                res_dir = '%s/'%(ego_user)
                res_dir0 = '/Wang-ds/xwang193/pygcn-popets-triangle/%s/' % (ego_user)

                feat_dir = '/Wang-ds/xwang193/deepwalk-master/data/' + str(ego_user) + '-adj-feat.pkl'
                # feat_dir = '/Users/xiulingwang/Downloads/facebook-data/data/' + str(ego_user) + '-adj-feat.pkl'
                # feat_dir = 'E:/python/banlance/code/google+-raw-data-3/gplus-processed-test1/' + str(ego_user) + '-adj-feat.pkl'
                # feat_dir = 'E:/python/banlance/code/dblp-data/data/' + str(ego_user) + '-adj-feat.pkl'
                f2 = open(feat_dir, 'rb')

                adj_orig, ft = pkl.load(f2, encoding='latin1')

                g = nx.Graph(adj_orig)

                METHOD = 'gcn'
                savepath = res_dir0 + METHOD + '-disconnected_edges_sampled-4clique' + Flag + '-' + str(
                    ego_user) + '-new-629-lap-multi.npy'
                print('***')
                disconnected_edges_sampled = np.load(savepath)

                METHOD = 'gcn'
                savepath = res_dir0 + METHOD + '-pos_sampled-4clique' + Flag + '-' + str(
                    ego_user) + '-new-629-lap-multi.npy'
                print('***')
                cliqs = np.load(savepath)

                num_triad = np.shape(cliqs)[0]

                idx = num_triad

                # savepath = res_dir + 'posterior-' + str(ego_user)+'-layer2.npy'
                # output_test=np.load(savepath)
                #
                # output_test =np.exp(output_test)

                # METHOD = 'gcn'
                # savepath = res_dir + METHOD + '-embed-' + Flag + '-' + str(ego_user) + '-new-629-'+me+'.npy'
                # print('***')
                # emb_matrix0 = np.load(savepath)
                #
                # emb_matrix2=emb_matrix0

                for b in sigmas:

                    sigma = b

                    dp = 7
                    # sigma = 50
                    if dp == 7:
                        Flag = '114-sage-' + str(ego_user) + '-' + str(dp) + '-' + str(b) + '-' + str(ratio)

                    else:
                        Flag = '131-sage-' + str(ego_user) + '-' + str(dp)

                    METHOD = 'sage'
                    savepath = res_dir + METHOD + '-embed-' + Flag + '-' + str(ego_user) + '-new-629-' + me + '.npy'
                    print('***')
                    emb_matrix0 = np.load(savepath)
                    output_test = torch.from_numpy(emb_matrix0)
                    output_test = torch.softmax(output_test, dim=1)

                    output_test = np.array(output_test.detach().numpy())

                    emb_matrix2 = output_test

                    mem = []
                    for path in cliqs:
                        prob0 = []
                        prob1 = []
                        prob2 = []
                        prob3 = []
                        nd1_ = path[0]
                        nd2_ = path[1]
                        nd3_ = path[2]
                        nd4_ = path[3]

                        emb1 = emb_matrix2[nd1_]
                        emb2 = emb_matrix2[nd2_]
                        emb3 = emb_matrix2[nd3_]
                        emb4 = emb_matrix2[nd4_]

                        embs = [emb1, emb2, emb3, emb4]

                        for ii in range(4):
                            for jj in range(ii + 1, 4):
                                emb1 = embs[ii]
                                emb2 = embs[jj]

                                prob0.append(np.dot(emb1, emb2))
                                prob1.append(
                                    np.dot(emb1, emb2) / (
                                        np.linalg.norm(emb1) * np.linalg.norm(
                                            emb2) + 0.0000000000000000000000000000001))
                                prob2.append(np.linalg.norm(np.array(emb1) - np.array(emb2)))
                                prob3.append(sigmoid(np.dot(emb1, emb2)))

                        prob0.sort()
                        prob1.sort()
                        prob2.sort()
                        prob3.sort()

                        prob = [prob0, prob1, prob2, prob3]
                        prob = np.array(list(itertools.chain.from_iterable(prob)))
                        mem.append(prob)
                    # print(mem,np.shape(mem))

                    # print(mem)

                    non_mem = []
                    for path in disconnected_edges_sampled:
                        prob0 = []
                        prob1 = []
                        prob2 = []
                        prob3 = []
                        nd1_ = path[0]
                        nd2_ = path[1]
                        nd3_ = path[2]
                        nd4_ = path[3]
                        emb1 = emb_matrix2[nd1_]
                        emb2 = emb_matrix2[nd2_]
                        emb3 = emb_matrix2[nd3_]
                        emb4 = emb_matrix2[nd4_]

                        embs = [emb1, emb2, emb3, emb4]

                        for ii in range(4):
                            for jj in range(ii + 1, 4):
                                emb1 = embs[ii]
                                emb2 = embs[jj]

                                prob0.append(np.dot(emb1, emb2))
                                prob1.append(
                                    np.dot(emb1, emb2) / (
                                        np.linalg.norm(emb1) * np.linalg.norm(
                                            emb2) + 0.0000000000000000000000000000001))
                                prob2.append(np.linalg.norm(np.array(emb1) - np.array(emb2)))
                                prob3.append(sigmoid(np.dot(emb1, emb2)))

                        prob0.sort()
                        prob1.sort()
                        prob2.sort()
                        prob3.sort()

                        prob = [prob0, prob1, prob2, prob3]
                        prob = np.array(list(itertools.chain.from_iterable(prob)))
                        non_mem.append(prob)

                    # non_mem_class=np.array([0]*int(num_triad/6)+[1]*int(num_triad/6)+[21]*int(num_triad/12)+[22]*int(num_triad/12)+[31]*int(num_triad/18)+[32]*int(num_triad/18)+[33]*int(num_triad/18)+[41]*int(num_triad/12)+[42]*int(num_triad/12)+[51]*int(num_triad/12)+[52]*int(num_triad/12))
                    non_mem_class = np.array(
                        [0] * idx + [1] * idx + [2] * idx + [3] * idx + [4] * idx + [5] * idx + [6] * idx + [
                            7] * idx + [8] * idx + [9] * idx)

                    print('@@@', len(non_mem_class), np.shape(disconnected_edges_sampled)[0])

                    y_label_neg = np.concatenate(
                        (non_mem_class.reshape(-1, 1), np.zeros((np.shape(disconnected_edges_sampled)[0], 2)),
                         disconnected_edges_sampled), axis=1)

                    # print(pos_sampled,neg_sampled)

                    print(np.shape(mem)[0], num_triad)

                    mem_idx = list(range(np.shape(mem)[0]))
                    mem_idx_sample = np.array(random.sample(mem_idx, num_triad))

                    pos_sampled = np.array(mem)[mem_idx_sample]
                    neg_sampled = np.array(non_mem)

                    print(np.shape(pos_sampled), np.shape(neg_sampled))

                    mem_class = np.array([12] * np.shape(pos_sampled)[0])

                    y_label_pos = np.concatenate(
                        (mem_class.reshape(-1, 1), np.ones((len(mem_class), 2)), np.array(cliqs)[mem_idx_sample]),
                        axis=1)

                    # y_label_pos=np.ones((np.shape(pos_sampled)[0],3))
                    # y_label_neg=np.zeros((np.shape(neg_sampled)[0],3))
                    neg_sampled0 = neg_sampled[0:int(num_triad)]
                    neg_sampled1 = neg_sampled[int(num_triad):int(2 * num_triad)]
                    neg_sampled2 = neg_sampled[int(2 * num_triad):int(3 * num_triad)]
                    neg_sampled3 = neg_sampled[int(3 * num_triad):int(4 * num_triad)]
                    neg_sampled4 = neg_sampled[int(4 * num_triad):int(5 * num_triad)]
                    neg_sampled5 = neg_sampled[int(5 * num_triad):int(6 * num_triad)]
                    neg_sampled6 = neg_sampled[int(6 * num_triad):int(7 * num_triad)]
                    neg_sampled7 = neg_sampled[int(7 * num_triad):int(8 * num_triad)]
                    neg_sampled8 = neg_sampled[int(8 * num_triad):int(9 * num_triad)]
                    neg_sampled9 = neg_sampled[int(9 * num_triad):int(10 * num_triad)]

                    y_label_neg0 = y_label_neg[0:int(num_triad)]
                    y_label_neg1 = y_label_neg[int(num_triad):int(2 * num_triad)]
                    y_label_neg2 = y_label_neg[int(2 * num_triad):int(3 * num_triad)]
                    y_label_neg3 = y_label_neg[int(3 * num_triad):int(4 * num_triad)]
                    y_label_neg4 = y_label_neg[int(4 * num_triad):int(5 * num_triad)]
                    y_label_neg5 = y_label_neg[int(5 * num_triad):int(6 * num_triad)]
                    y_label_neg6 = y_label_neg[int(6 * num_triad):int(7 * num_triad)]
                    y_label_neg7 = y_label_neg[int(7 * num_triad):int(8 * num_triad)]
                    y_label_neg8 = y_label_neg[int(8 * num_triad):int(9 * num_triad)]
                    y_label_neg9 = y_label_neg[int(9 * num_triad):int(10 * num_triad)]

                    y_label_pos = np.concatenate(
                        (mem_class.reshape(-1, 1), np.ones((len(mem_class), 2)), np.array(cliqs)[mem_idx_sample]),
                        axis=1)

                    # y_label_pos=np.ones((np.shape(pos_sampled)[0],3))
                    # y_label_neg=np.zeros((np.shape(neg_sampled)[0],3))

                    from sklearn.model_selection import train_test_split

                    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(pos_sampled,
                                                                                                y_label_pos,
                                                                                                test_size=0.3,
                                                                                                random_state=42)

                    X_test_train0, X_test_test0, y_test_train0, y_test_test0 = train_test_split(neg_sampled0,
                                                                                                y_label_neg0,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train1, X_test_test1, y_test_train1, y_test_test1 = train_test_split(neg_sampled1,
                                                                                                y_label_neg1,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train2, X_test_test2, y_test_train2, y_test_test2 = train_test_split(neg_sampled2,
                                                                                                y_label_neg2,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train3, X_test_test3, y_test_train3, y_test_test3 = train_test_split(neg_sampled3,
                                                                                                y_label_neg3,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train4, X_test_test4, y_test_train4, y_test_test4 = train_test_split(neg_sampled4,
                                                                                                y_label_neg4,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train5, X_test_test5, y_test_train5, y_test_test5 = train_test_split(neg_sampled5,
                                                                                                y_label_neg5,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train6, X_test_test6, y_test_train6, y_test_test6 = train_test_split(neg_sampled6,
                                                                                                y_label_neg6,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train7, X_test_test7, y_test_train7, y_test_test7 = train_test_split(neg_sampled7,
                                                                                                y_label_neg7,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train8, X_test_test8, y_test_train8, y_test_test8 = train_test_split(neg_sampled8,
                                                                                                y_label_neg8,
                                                                                                test_size=0.3,
                                                                                                random_state=42)
                    X_test_train9, X_test_test9, y_test_train9, y_test_test9 = train_test_split(neg_sampled9,
                                                                                                y_label_neg9,
                                                                                                test_size=0.3,
                                                                                                random_state=42)

                    X_train = np.concatenate((X_train_train, X_test_train0, X_test_train1, X_test_train2, X_test_train3,
                                              X_test_train4, X_test_train5, X_test_train6, X_test_train7, X_test_train8,
                                              X_test_train9), axis=0)
                    X_test = np.concatenate((X_train_test, X_test_test0, X_test_test1, X_test_test2, X_test_test3,
                                             X_test_test4, X_test_test5, X_test_test6, X_test_test7, X_test_test8,
                                             X_test_test9), axis=0)
                    y_train = np.concatenate((y_train_train, y_test_train0, y_test_train1, y_test_train2, y_test_train3,
                                              y_test_train4, y_test_train5, y_test_train6, y_test_train7, y_test_train8,
                                              y_test_train9), axis=0)
                    y_test = np.concatenate((y_train_test, y_test_test0, y_test_test1, y_test_test2, y_test_test3,
                                             y_test_test4, y_test_test5, y_test_test6, y_test_test7, y_test_test8,
                                             y_test_test9), axis=0)

                    # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
                    #
                    # # ######################################################################

                    from sklearn import metrics
                    from sklearn.neural_network import MLPClassifier

                    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
                                        max_iter=500)

                    mlp.fit(X_train, y_train[:, 0])

                    with open("{}/mlp_{}-b-new-629-imp-{}-{}-{}-multi.pkl".format(res_dir,target,sigma,me,ratio), "wb") as f2:
                        pkl.dump(mlp, f2)

                    print("Training set score: %f" % mlp.score(X_train, y_train[:, 0]))
                    print("Test set score: %f" % mlp.score(X_test, y_test[:, 0]))

                    y_score = mlp.predict(X_test)
                    print(metrics.f1_score(y_test[:, 0], y_score, average='micro'))
                    print(metrics.classification_report(y_test[:, 0], y_score, labels=range(3)))

                    proba0 = mlp.predict_proba(X_test)
                    # proba = np.amax(proba, axis=1)
                    proba = proba0[:, 1]

                    acc_mlp_sim = accuracy_score(y_score, y_test[:, 0])

                    y_label_test = y_test

                    acc = accuracy_score(y_label_test[:, 0], y_score)
                    recall = recall_score(y_score, y_label_test[:, 0], average='macro')
                    precision = precision_score(y_score, y_label_test[:, 0], average='macro')
                    f1 = f1_score(y_score, y_label_test[:, 0], average='macro')
                    auc = roc_auc_score(y_label_test[:, 0], proba0, average='macro', multi_class="ovr")

                    print(args.dataset, acc, recall, precision, f1, auc)
                    results_att.append([args.dataset, acc, recall, precision, f1, auc])
                    results_all.append([args.dataset, b, ratio, acc, recall, precision, f1, auc])
                    tsts = []
                    for i in range(len(y_score)):
                        node1 = y_test[i][0]
                        node2 = y_test[i][1]
                        # dgr1 = g.degree(node1)
                        # dgr2 = g.degree(node2)
                        #
                        # gender1 = g.nodes[node1]['gender']
                        # gender2 = g.nodes[node2]['gender']

                        tst = [y_score[i], proba0[i], y_test[i][2], y_test[i][0], y_test[i][1]]
                        tsts.append(tst)
                    name = ['y_score', 'prob', 'y_test_grd', 'y_test_class', '0']
                    result = pd.DataFrame(columns=name, data=tsts)
                    result.to_csv("{}/mlp_sim-{}-b-new-629-imp-{}-{}-{}-multi.csv".format(res_dir,target,sigma,me,ratio))

                    # y_0 = np.where(y_test[:, 0] == 0)[0]
                    # y_1 = np.where(y_test[:, 0] == 1)[0]
                    # y_2 = np.where(y_test[:, 0] == 2)[0]
                    #
                    # acc = accuracy_score(y_label_test[:, 2][y_0], y_score[y_0])
                    # # recall = recall_score(y_score[y_0], y_label_test[:, 2][y_0])
                    # # precision = precision_score(y_score[y_0], y_label_test[:, 2][y_0])
                    # # f1 = f1_score(y_score[y_0], y_label_test[:, 2][y_0])
                    # # auc = roc_auc_score(y_label_test[:, 2][y_0], proba[y_0])
                    #
                    # print(acc, recall, precision, f1)
                    # results_att.append([args.dataset, b,ratio,acc, recall, precision, f1, auc])
                    #
                    # results_all.append([args.dataset, b, ratio, acc, recall, precision, f1, auc])
                    #
                    # acc = accuracy_score(y_label_test[:, 2][y_1], y_score[y_1])
                    # # recall = recall_score(y_score[y_1], y_label_test[:, 2][y_1])
                    # # precision = precision_score(y_score[y_1], y_label_test[:, 2][y_1])
                    # # f1 = f1_score(y_score[y_1], y_label_test[:, 2][y_1])
                    # # auc = roc_auc_score(y_label_test[:, 2][y_1], proba[y_1])
                    #
                    # print(acc, recall, precision, f1)
                    # results_att.append([args.dataset, b,ratio,acc, recall, precision, f1, auc])
                    #
                    # results_all.append([args.dataset, b, ratio, acc, recall, precision, f1, auc])
                    #
                    # acc = accuracy_score(y_label_test[:, 2][y_2], y_score[y_2])
                    # # recall = recall_score(y_score[y_2], y_label_test[:, 2][y_2])
                    # # precision = precision_score(y_score[y_2], y_label_test[:, 2][y_2])
                    # # f1 = f1_score(y_score[y_2], y_label_test[:, 2][y_2])
                    # # auc = roc_auc_score(y_label_test[:, 2][y_2], proba[y_2])
                    #
                    # print(acc, recall, precision, f1)
                    # results_att.append([args.dataset, b,ratio,acc, recall, precision, f1, auc])
                    #
                    # results_all.append([args.dataset, b, ratio, acc, recall, precision, f1, auc])
                    #
                    # name = ['ego_user','b','ratio', 'acc', 'recall', 'precision', 'f1', 'auc']
                    # result = pd.DataFrame(columns=name, data=results_att)
                    # result.to_csv("{}/pmia-results-triangle-new-629-imp-pert-{}-{}.csv".format(res_dir,sigma,ratio))


                name = ['ego_user','b','ratio', 'acc', 'recall', 'precision', 'f1', 'auc']
                result = pd.DataFrame(columns=name, data=results_all)
                result.to_csv("{}/pmia-results-{}-new-629-imp-{}-{}-multi-b.csv".format(res_dir,target,me,ratio))
