import torch.nn as nn
import torch.nn.functional as F
from layers import *
import torch
import numpy as np


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GCN1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        embed = x
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1),embed

class GCN1_lp(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,cnt_train):
        super(GCN1_lp, self).__init__()

        self.gc1 = GraphConvolution1(nfeat, nhid)
        self.gc2 = GraphConvolution1(nhid, nclass)
        self.sim_train = torch.zeros(cnt_train)
        # self.gc1.requires_grad = True
        # self.gc2.requires_grad = True
        self.sim_train = torch.autograd.Variable(self.sim_train, requires_grad=True)
        # self.gc2 = torch.autograd.Variable(self.gc2, requires_grad=True)
        self.dropout = dropout
        # self.loss_train=

    def forward(self, x, adj,train_edge,train_label, m,criterion):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # sim = []
        embed=x
        # n=torch.from_numpy(np.array(0))
        # # print(n)
        ed=train_edge

        nd1 = ed[0].long()
        nd2 = ed[1].long()

        mul_a = embed[nd1]
        mul_b = embed[nd2]

        sim0 = torch.dot(mul_a, mul_b)
        self.sim_train = (m(sim0))
            # print(n.item())

            # self.sim_train[n]=(m(sim0))
            # n+=1

            # sim.append((mul_a*mul_b).detach().numpy())
            # print(sim)
            # print(sim.type())
            # exit()
        # print(output[idx_train])
        # sim = torch.Tensor(sim)
        # sim_train = torch.Tensor(sim_train)
        # sim_train= torch.autograd.Variable(sim_train, requires_grad=True)
        # loss_train = criterion(self.sim_train, train_labels.float())


        return x,self.sim_train


class GCN_pia(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_pia, self).__init__()

        self.gc1 = GraphConvolution_pia(nfeat, nhid)
        self.gc2= GraphConvolution_pia(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        embed1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        embed2 = x
        return F.log_softmax(x, dim=1),embed1,embed2


class GCN_pia1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_pia1, self).__init__()

        self.gc1 = GraphConvolution_pia(nfeat, nhid)
        # self.gc2= GraphConvolution_pia(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        embed1 = x
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        # embed2 = x
        return F.log_softmax(x, dim=1),embed1



class GCN_pia_unlearn_baseline(nn.Module):
    def __init__(self,npara, nfeat, nhid, nclass, dropout):
        super(GCN_pia_unlearn, self).__init__()

        npara1=np.array(npara[0])
        npara2 = np.array(npara[1])

        self.gc1 = GraphConvolution_pia_unlearn1(npara1,npara2,nfeat, nhid)

        npara1 = np.array(npara[2])
        npara2 = np.array(npara[3])

        self.gc2= GraphConvolution_pia_unlearn1(npara1,npara2,nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        embed1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        embed2 = x
        return F.log_softmax(x, dim=1),embed1,embed2




class GCN_feature_selection(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,featuresSelected,):
        super(GCN_feature_selection, self).__init__()

        self.gc1 = GraphConvolution_feature_selection(nfeat, nhid,featuresSelected, True)
        self.gc2= GraphConvolution_feature_selection(nhid, nclass, featuresSelected,False)
        self.dropout = dropout

    def forward(self, x, adj,temp):
        x,featureSelector=self.gc1(x, adj,True,temp)
        embed1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj,False,temp)
        embed2 = x
        return F.log_softmax(x, dim=1),embed1,embed2,featureSelector


class GCN_pia2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_pia2, self).__init__()

        self.gc1 = GraphConvolution_pia(nfeat, nhid)
        self.gc2 = GraphConvolution_pia(nhid, nhid)
        self.gc3= GraphConvolution_pia(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        embed1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        embed2 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        embed3 = x
        return F.log_softmax(x, dim=1),embed1,embed2,embed3


class GCN_pia3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_pia3, self).__init__()

        self.gc1 = GraphConvolution_pia(nfeat, nhid)
        self.gc2 = GraphConvolution_pia(nhid, nhid)
        self.gc3 = GraphConvolution_pia(nhid, nhid)
        self.gc4= GraphConvolution_pia(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        embed1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        embed2 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        embed3 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        embed4 = x
        return F.log_softmax(x, dim=1),embed1,embed2,embed3,embed4

class GCN_pia4(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_pia4, self).__init__()

        self.gc1 = GraphConvolution_pia(nfeat, nhid)
        self.gc2 = GraphConvolution_pia(nhid, nhid)
        self.gc3 = GraphConvolution_pia(nhid, nhid)
        self.gc4 = GraphConvolution_pia(nhid, nhid)
        self.gc5= GraphConvolution_pia(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        embed1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        embed2 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        embed3 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        embed4 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc5(x, adj)
        embed5 = x
        return F.log_softmax(x, dim=1),embed1,embed2,embed3,embed4,embed5

class GCN_pia6(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_pia6, self).__init__()

        self.gc1 = GraphConvolution_pia(nfeat, nhid)
        self.gc2 = GraphConvolution_pia(nhid, nhid)
        self.gc3 = GraphConvolution_pia(nhid, nhid)
        self.gc4 = GraphConvolution_pia(nhid, nhid)
        self.gc5 = GraphConvolution_pia(nhid, nhid)
        self.gc6 = GraphConvolution_pia(nhid, nhid)
        self.gc7 = GraphConvolution_pia(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        embed1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        embed2 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        embed3 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        embed4 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc5(x, adj)
        embed5 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc6(x, adj)
        embed6 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc7(x, adj)
        embed7 = x
        return F.log_softmax(x, dim=1), embed1, embed2, embed3, embed4, embed5, embed6, embed7


class GCN_pia8(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_pia8, self).__init__()

        self.gc1 = GraphConvolution_pia(nfeat, nhid)
        self.gc2 = GraphConvolution_pia(nhid, nhid)
        self.gc3 = GraphConvolution_pia(nhid, nhid)
        self.gc4 = GraphConvolution_pia(nhid, nhid)
        self.gc5 = GraphConvolution_pia(nhid, nhid)
        self.gc6 = GraphConvolution_pia(nhid, nhid)
        self.gc7 = GraphConvolution_pia(nhid, nhid)
        self.gc8 = GraphConvolution_pia(nhid, nhid)
        self.gc9= GraphConvolution_pia(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        embed1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        embed2 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        embed3 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        embed4 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc5(x, adj)
        embed5 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc6(x, adj)
        embed6 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc7(x, adj)
        embed7 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc8(x, adj)
        embed8 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc9(x, adj)
        embed9 = x
        return F.log_softmax(x, dim=1),embed1,embed2,embed3,embed4,embed5,embed6,embed7,embed8,embed9




class GCN_pia2_unlearn(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_pia2_unlearn, self).__init__()

        self.gc1 = GraphConvolution_pia_unlearn(nfeat, nhid)
        self.gc2 = GraphConvolution_pia_unlearn(nhid, nhid)
        self.gc3= GraphConvolution_pia_unlearn(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        embed1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        embed2 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        embed3 = x
        return F.log_softmax(x, dim=1),embed1,embed2,embed3


class GCN_pia_unlearn(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_pia2_unlearn, self).__init__()

        self.gc1 = GraphConvolution_pia_unlearn(nfeat, nhid)
        self.gc2= GraphConvolution_pia_unlearn(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        embed1 = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        embed2 = x
        return F.log_softmax(x, dim=1),embed1,embed2