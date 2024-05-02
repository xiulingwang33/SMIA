
import os
import pickle as pkl
import networkx as nx
import numpy as np
import copy
import itertools
import pandas as pd

# G_EGO_USERS = ['citeseer','combined', 'Gplus', 'lastfm']
# num = 6
# m_seed = [11]
# for m_ in m_seed:
#     results_att = []
#
#     for ego_user in G_EGO_USERS:
#
#         dp = 0
#         sigma0 = 1
#         if dp == 1:
#             Flag = '114-gcn-' + str(ego_user) + '-' + str(dp) + '-' + str(sigma0)
#
#         else:
#             Flag = '131-gcn-' + str(ego_user) + '-' + str(dp)
#
#         res_dir = '%s/' % (ego_user)
#
#         feat_dir = '/Wang-ds/xwang193/deepwalk-master/data/' + str(ego_user) + '-adj-feat.pkl'
#         # feat_dir = '/Users/xiulingwang/Downloads/facebook-data/data/' + str(ego_user) + '-adj-feat.pkl'
#         # feat_dir = 'E:/python/banlance/code/google+-raw-data-3/gplus-processed-test1/' + str(ego_user) + '-adj-feat.pkl'
#         # feat_dir = 'E:/python/banlance/code/dblp-data/data/' + str(ego_user) + '-adj-feat.pkl'
#         f2 = open(feat_dir, 'rb')
#
#         adj_orig, ft = pkl.load(f2, encoding='latin1')
#
#         g = nx.Graph(adj_orig)
#
#         if ego_user == 'lastfm' or ego_user == 'Gplus' or ego_user == 'Facebook' or ego_user == 'combined':
#             adj_orig = np.array(adj_orig.todense())
#
#         adj_=np.linalg.matrix_power(adj_orig, 2)
#
#         # print(adj_)
#
#         # result_sum = (np.sum(adj_) - np.trace(adj_)) // 2
#         #
#         # print(result_sum)
#
#         num_nodes=g.number_of_nodes()
#
#         num_2path=0
#
#         for ii in range(num_nodes-1):
#             for jj in range(ii+1,num_nodes):
#
#                 num_2path+=adj_[ii][jj]
#
#         print( num_2path)
#
# exit()

file_path='/Wang-ds/xwang193/pygcn-popets-triangle/gplus/'
fnames = os.listdir(file_path)
fnames = sorted(fnames)

results=[]

for fname in fnames:
    ego_user=fname.split('-')[0]
    print(ego_user)

    if ego_user=='.DS_Store':
        continue

    feat_dir=file_path+fname

    f2 = open(feat_dir, 'rb')

    adj_orig, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj_orig)

    num_nodes=g.number_of_nodes()

    total=int(num_nodes*(num_nodes-1)*(num_nodes-2)/6)

    adj_orig=np.array(adj_orig.todense())


    num_nodes = g.number_of_nodes()

    num_2path=0
    num_3path = 0

    adj_ = np.linalg.matrix_power(adj_orig, 2)
    for ii in range(num_nodes-1):
        for jj in range(ii+1,num_nodes):

            num_2path+=adj_[ii][jj]

    print( num_2path)

    adj_ = np.linalg.matrix_power(adj_orig, 3)
    for ii in range(num_nodes - 1):
        for jj in range(ii + 1, num_nodes):
            num_3path += adj_[ii][jj]

    print(num_3path)

    adj_orig_=copy.deepcopy(adj_orig)
    for i in range(num_nodes):
        adj_orig_[i][i]=1



    set_edges={}
    set_disconnected_edges={}

    for i in range(num_nodes):
        set_edges[i]=np.where(adj_orig[i]==1)[0]
        set_disconnected_edges[i] = np.where(adj_orig_[i] == 0)[0]

    edge_tuples0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in g.edges()]
    # print(edge_tuples0)
    #
    # triad_cliques = []
    # for i,j in edge_tuples0:
    #     n1 = set_edges[i]
    #     n2 = set_edges[j]
    #     nd = np.intersect1d(n1, n2)
    #     triad_clique=np.concatenate((np.array([i]*len(nd)).reshape(-1,1),np.array([j]*len(nd)).reshape(-1,1),nd.reshape(-1,1)),axis=1)
    #     triad_cliques.append(triad_clique)
    #
    # triad_cliques = np.array(list(itertools.chain.from_iterable(triad_cliques)))
    #
    # print(np.shape(triad_cliques))
    #
    # savepath = 'triangle0-'+ str(ego_user)
    # print('***')
    # np.save(savepath, triad_cliques)
    #
    # edge_tuples0=[]
    #
    # for edge in triad_cliques:
    #     # print(edge)
    #     s=np.sort(edge)
    #
    #     edge_tuples0.append((s[0],s[1],s[2]))
    #
    # # triad_cliques= list(set(map(tuple, edge_tuples0)))
    # print(np.shape(edge_tuples0))
    #
    # triad_cliques = list(set(edge_tuples0))
    #
    # # print(np.shape(triad_cliques))
    #
    # print(np.shape(triad_cliques))
    # METHOD='gcn'
    # savepath = 'triangle-'+ str(ego_user)
    # print('***')
    # np.save(savepath, triad_cliques)


    METHOD='gcn'
    savepath = 'triangle-'+ str(ego_user)+'.npy'
    print('***')
    triad_cliques=np.load(savepath)


    cliqs=[]
    for nds in triad_cliques:
        nd1=nds[0]
        nd2=nds[1]
        nd3=nds[2]
        nd1_connected = set_edges[nd1]
        nd2_connected = set_edges[nd2]
        nd3_connected = set_edges[nd3]

        nd_connected=np.intersect1d(nd1_connected,nd2_connected)
        nd_connected=np.intersect1d(nd_connected,nd3_connected)

        cliq=np.concatenate((nd_connected.reshape(-1,1),np.array([nd1]*len(nd_connected)).reshape(-1,1),np.array([nd2]*len(nd_connected)).reshape(-1,1),np.array([nd3]*len(nd_connected)).reshape(-1,1)),axis=1)

        cliqs.append(cliq)

    cliqs=np.array(cliqs)
    cliqs = np.array(list(itertools.chain.from_iterable(cliqs)))

    print(np.shape(cliqs))

    METHOD = 'gcn'
    savepath = '4clique-' + str(ego_user)
    print('***')
    np.save(savepath, cliqs)


    results.append([ego_user,num_nodes,num_2path,num_3path,np.shape(triad_cliques),np.shape(cliqs)])

    result = pd.DataFrame(data=results)

    # result.to_csv("pmia-results-triangle-new-629-multi.csv".format(args.dataset))

    result.to_csv("gplus-preprocess.csv")



