import math

import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        print(self.weight)
        # self.weight.requires_grad = True
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            # self.bias.requires_grad = True
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution1(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # print(self.weight)
        # self.weight.requires_grad = True
        # self.weight = torch.autograd.Variable(self.weight, requires_grad=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            # self.bias.requires_grad = True
            # self.bias = torch.autograd.Variable(self.bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolution_pia(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_pia, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # print(self.weight)
        # self.weight.requires_grad = True
        # self.weight = torch.autograd.Variable(self.weight, requires_grad=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            # self.bias.requires_grad = True
            # self.bias = torch.autograd.Variable(self.bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GraphConvolution_feature_selection(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, featuresSelected,flag2,bias=True):
        super(GraphConvolution_feature_selection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = featuresSelected

        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if flag2:
            self.WF = Parameter(torch.FloatTensor(in_features,self.k))
            self.weight = Parameter(torch.FloatTensor(self.k,out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        # print(self.weight)
        # self.weight.requires_grad = True
        # self.weight = torch.autograd.Variable(self.weight, requires_grad=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            # self.bias.requires_grad = True
            # self.bias = torch.autograd.Variable(self.bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj,flag,temp):
        if flag:
            global featureSelector
            featureSelector = self.WF
            support = torch.mm(input, F.gumbel_softmax(self.WF,hard=False,tau=temp))
            output = torch.mm(adj, support)
            op = torch.mm(output, self.weight)

            if self.bias is not None:
                return op + self.bias, featureSelector
            else:
                return op, featureSelector

        else:
            output = torch.mm(adj, input)
            op = torch.mm(output, self.weight)

            if self.bias is not None:
                return op + self.bias
            else:
                return op



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution_pia_unlearn1(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, para1,para2,in_features, out_features, bias=True):
        super(GraphConvolution_pia_unlearn1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.paras1=para1
        self.paras2 = para2
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        print(self.weight)
        print(self.weight.size())
        # self.weight.requires_grad = True
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            # self.bias.requires_grad = True
        else:
            self.register_parameter('bias', None)
        self.reset_parameters1()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def reset_parameters1(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

        self.weight.data.copy_(torch.from_numpy(np.array(self.paras1)))
        if self.bias is not None:
            self.bias.data.copy_(torch.from_numpy(np.array(self.paras2)))




    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GraphConvolution_pia_unlearn(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_pia_unlearn, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # print(self.weight)
        # self.weight.requires_grad = True
        # self.weight = torch.autograd.Variable(self.weight, requires_grad=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            # self.bias.requires_grad = True
            # self.bias = torch.autograd.Variable(self.bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'