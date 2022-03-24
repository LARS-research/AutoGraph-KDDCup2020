"""the simple baseline for autograph"""

# python run_local_test.py --dataset_dir=../public/c --code_dir=./code_submission

import numpy as np
import pandas as pd


import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge
from torch_geometric.data import Data
from torch_geometric.utils import scatter_
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
import os
import time
from sklearn.model_selection import StratifiedKFold,KFold



import math
import random
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(1234)

from torch.optim import Optimizer

from torch.autograd import Variable

class FocalLoss(torch.nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
 
 
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
 
        probs = (P*class_mask).sum(1).view(-1,1)
 
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
 
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)
 
 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

def calc_ent(x):
    if np.sum(x) <= 5:
        ent = 50.0
        return ent
    else:
        ent = 0.0
    for i in range(x.shape[0]):
        p = float(x[i]) / np.sum(x)
        if p != 0:
            logp = np.log2(p)
            ent -= p * logp

    # if ent == 0:
    #     ent += (- min(100,np.sum(x))) * 0.0001 + 0.01
    return ent

class Linear2(Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear2, self).__init__(in_features, out_features, bias=True)


    def forward(self, input):
        x = torch.matmul(input, self.weight.permute(1,0))
        return x + self.bias


class GCNConv2(GCNConv):
    def __init__(self, in_channels, out_channels, isconvert = False, **kwargs):
        super(GCNConv2, self).__init__(in_channels, out_channels, **kwargs)
        self.first_lin = Linear(in_channels, out_channels)
        self.converter = Linear(in_channels, out_channels)
        self.first_lin2 = Linear(out_channels, out_channels)
        self.converter2 = Linear(out_channels, out_channels)
        self.isconverter = isconvert
        self.norm_weights = Parameter(torch.Tensor(1))
        if random.random() > 0.5:
            torch.nn.init.constant(self.norm_weights, 1.0)
        else:
            torch.nn.init.constant(self.norm_weights, 5.0)
        self.cached_result2 = None
        
        self.norm_w1 = Linear(out_channels * 2 + 1, 1)
        self.norm_w2 = Linear(16, 1)
        # self.aggr = 'mean'
        self.first_lin.reset_parameters()
        self.converter.reset_parameters()
        self.first_lin2.reset_parameters()
        self.converter2.reset_parameters()
        self.norm_w1.reset_parameters()   
        self.norm_w2.reset_parameters()  


        
        
    def norm2(self,edge_index, num_nodes, norm_weights, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        if self.cached_result2 is None:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            self.cached_result2 = edge_index, edge_weight
        
        else:
            edge_index, edge_weight = self.cached_result2

        # print(edge_weight)
        if self.training: 
            rand_tensor = torch.rand(edge_weight[:-num_nodes].shape).to(edge_weight.device)
            rand_tensor = (rand_tensor > 0.15).float()
            edge_weight = torch.cat([edge_weight[:-num_nodes] * rand_tensor,edge_weight[-num_nodes:]],0)
        # print(edge_weight)    
        edge_weight = torch.cat([edge_weight[:-num_nodes],edge_weight[-num_nodes:] * norm_weights],0)
        
        # edge_weight = torch.cat([torch.clamp(edge_weight[:-num_nodes],0.5,10.0),edge_weight[-num_nodes:] * norm_weights],0)
        # edge_weight = norm_weights
        # print(norm_weights)
        # print(edge_index.shape)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_tensor = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[row]
        # norm_tensor = torch.clamp(norm_tensor,0.02,10.0)
        return edge_index, norm_tensor
    
    
    def propagate(self, edge_index, size=None, dim=0, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferred and assumed to be symmetric.
                (default: :obj:`None`)
            dim (int, optional): The axis along which to aggregate.
                (default: :obj:`0`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """

        dim = 0
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}
#         print([y.shape for x,y in kwargs.items()])
#         print(kwargs)
        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)[0]
                tmp2 = kwargs.get(arg[:-2], None)[1] 
#                 print(arg,tmp.shape,tmp2.shape)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
#                     if isinstance(tmp, tuple) or isinstance(tmp, list):
#                         assert len(tmp) == 2
#                         if tmp[1 - idx] is not None:
#                             if size[1 - idx] is None:
#                                 size[1 - idx] = tmp[1 - idx].size(dim)
#                             if size[1 - idx] != tmp[1 - idx].size(dim):
#                                 raise ValueError(__size_error_msg__)
#                         tmp = tmp[idx]

                    if tmp is None:
                        message_args.append(tmp)
                    else:
                        if size[idx] is None:
                            size[idx] = tmp.size(dim)
                        if size[idx] != tmp.size(dim):
                            raise ValueError(__size_error_msg__)
                        tmp = torch.index_select(tmp, dim, edge_index[idx][:-size[0]])
                        tmp2 = torch.index_select(tmp2, dim, edge_index[idx][-size[0]:])
                        message_args.append(torch.cat([tmp,tmp2],0))
            else:
                message_args.append(kwargs.get(arg, None))
                
#         print([x.shape for x in message_args])
#         message_args = [torch.cat([message_args[0][:-size[0]],message_args[1][-size[0]:]],0)] + message_args[2:]
#         print(message_args)

        
        
        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        
        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])
                
        update_args = [kwargs[arg] for arg in self.__update_args__]
#         print(size)
        out = self.message(*message_args)

        out = scatter_(self.aggr, out, edge_index[i], dim, dim_size=size[i])
#         print(out.shape,edge_index[i].shape,size)
#         print(out,out.shape)
        out = self.update(out, *update_args)

        return out
    
    def message(self, x_j, norm):
#         print(x_j.shape)

        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
#         print(aggr_out.shape)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out    
    
    def forward(self, x, edge_index, edge_weight=None):
        """"""
        
        if self.isconverter:        
            
            x1 = F.relu(self.converter(x))
            x1 = F.dropout(x1, p=0.3, training=self.training)
            x1 = (self.converter2(x1))
            x2 = (self.first_lin(x))
            x2 = F.dropout(x2, p=0.3, training=self.training)
#             x2 = F.relu(self.first_lin2(x2))
            x = [x1,x2]
            
        else:
            tmp = x
            tmp = (self.first_lin(tmp))
            tmp = F.dropout(tmp, p=0.3, training=self.training)
#             tmp = (self.first_lin2(tmp))
            x = [tmp,tmp]
#         x = torch.matmul(x, self.weight)
#         edge_weight = F.dropout(edge_weight, p=0.3, training=self.training)
#         if self.cached and self.cached_result is not None:
#             if edge_index.size(1) != self.cached_num_edges:
#                 raise RuntimeError(
#                     'Cached {} number of edges, but found {}. Please '
#                     'disable the caching behavior of this layer by removing '
#                     'the `cached=True` argument in its constructor.'.format(
#                         self.cached_num_edges, edge_index.size(1)))
        
        self.num_nodes = x[0].size(0)
        
#         if not self.cached or self.cached_result is None:
        if True:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm2(edge_index, x[0].size(0), self.norm_weights, edge_weight,
                                         self.improved, x[0].dtype)
#             self.cached_result = edge_index, norm

#         edge_index, norm = self.cached_result
#         print(norm.shape,norm)
        return self.propagate(edge_index, x=x, norm=norm)
    




class GCN3(torch.nn.Module):

    def __init__(self, num_layers=2, hidden=48,  features_num=16, num_class=2,  features_num2=16,  features_num3=16,  features_num4=16, features_num5=16, features_num6 = 16, isconvert = False, isx4 = False, isx6 = False, norm_prob = False, isembed = False):
        super(GCN3, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.isx4 = isx4
        self.isx6 = isx6
        self.isconvert = isconvert
        self.norm_prob = norm_prob
        self.norm_prob = False
        self.num_class = num_class
        self.isembed = isembed
        self.hidden = hidden
        
        self.lin_raw = Linear(features_num, hidden)
        if self.isconvert:
            first_hidden = features_num + features_num5
        else:
            first_hidden = features_num + features_num5 + features_num2
            
        for i in range(num_layers - 1):
            
            if i > 0:
                self.convs.append(GCNConv2(hidden, hidden, isconvert = False))
            else:
                self.convs.append(GCNConv2(first_hidden, hidden, isconvert = isconvert))
            
#         self.convs2 = GCNConv2(hidden, hidden, isconvert = False)
        
        if self.isx4:
            self.lin2 = Linear(hidden + features_num3 + features_num4, hidden)
            self.lin3 = Linear(2, 1)
        else:
            self.lin2 = Linear(hidden + features_num3, hidden)
            self.lin3 = Linear(2, 1)

        if not isembed:
            temp_c = hidden + features_num3 
            if self.isx4:
                temp_c += features_num4
            if self.isx6:
                temp_c += features_num6
        self.lin2 = Linear(temp_c, num_class)
            
        self.first_lin = Linear2(first_hidden, hidden)
        
        self.bn1 = torch.nn.BatchNorm1d(features_num)
        self.bn2 = torch.nn.BatchNorm1d(features_num3)
        self.bn4 = torch.nn.BatchNorm1d(features_num4)
        self.bn6 = torch.nn.BatchNorm1d(features_num6)
        self.bn5 = torch.nn.BatchNorm1d(features_num5)
        
        self.embed = torch.nn.Embedding(num_class, hidden)
        
        
        
        
        
#         self.lin3 = torch.nn.Linear(features_num2, features_num2)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.bn4.reset_parameters()
        self.bn6.reset_parameters()
        self.bn5.reset_parameters()
        self.embed.reset_parameters()
        
    def forward(self, data, node_mask = None):
        x, x2, x3, x4, edge_index, edge_weight,  = data.x, data.x2, data.x3, data.x4, data.edge_index, data.edge_weight
        x6 = data.x6
        x5 = data.x5

#         if self.training: 
#             ylim = edge_weight.shape[0]
#             rand_tensor = torch.rand(edge_weight.shape).to(edge_weight.device)
#             rand_tensor = (rand_tensor > 0.15).float()

#             edge_weight = edge_weight * rand_tensor
#         print(time.ctime())
        


        x2 = torch.cat([x2[:,:-1],1-x2[:,:-1].sum(1).view(-1,1)],1)
#         print(torch.index_select(x2[:,:-1], 0, edge_index[0,:]).shape)
#         print(x2_norm)
        
        if node_mask is not None:
            x2 = x2 * node_mask.view(-1,1)
            
#             print(x2)
#         x5 = self.bn5(x5)
#         raw_x = self.bn1(x)
        if not self.isconvert:  # 如果没有convert操作，则加入x2真标签
            x = torch.cat([x,x5,x2],1)
        else:
            x = torch.cat([x,x5],1)
        
            
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
#             x = torch.cat([F.relu(x[:,:self.hidden//2]),F.sigmoid(x[:,self.hidden//2:])],1)
#             x[:,self.hidden//2:] = F.sigmoid(x[:,self.hidden//2:])
        
#         print(x.shape,F.relu(self.convs2(x, edge_index, edge_weight=edge_weight)).shape)
#         x2 = F.relu(self.convs2(x, edge_index, edge_weight=edge_weight))  
#         x = x + x2
        x = F.dropout(x, p=0.2, training=self.training)
        middle = x
        x3 = self.bn2(x3)
        x3 = F.dropout(x3, p=0.3, training=self.training)
        x4 = self.bn4(x4)
        x4 = F.dropout(x4, p=0.3, training=self.training)
        x6 = self.bn6(x6)
        x6 = F.dropout(x6, p=0.2, training=self.training)
        if self.isx4:
            x = torch.cat([x, x3, x4],1)
        else:
            x = torch.cat([x, x3],1)
            
        if self.isx6:
            x = torch.cat([x, x6],1)
        x = self.lin2(x) # + x2_norm
        return x,middle

    def __repr__(self):
        return self.__class__.__name__     
    
class Model:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.isconvert = False
        self.isx4 = False


    def generate_pyg_data(self, data, isvalid = False):

        x = data['fea_table']

        

        df = data['edge_file']
        cache = df.values
        if cache.shape[0] < 2000000 and x.shape[1] > 1:
            sparse_dict2 = {}
            cache2 = x.values
            for i in range(cache2.shape[0]):
                sparse_dict2[int(cache2[i,0])] = cache2[i,1:]
            len_fenwei = 8
            cosine_dict = {}
            w_mat = []
            for i in range(cache.shape[0]):
                a = int(cache[i,0])
                b = int(cache[i,1])
                w_mat.append(np.sum(sparse_dict2[a]*sparse_dict2[b]))
            fenweishu = []
            w_mat = np.array(w_mat)
            w_fenwei =  [0.9,0.9,0.95,0.95,1.0,1.0,1.0,1.0]
            for i in range(len_fenwei):
                fenweishu.append(np.percentile(w_mat,i * 100//len_fenwei))
            print(fenweishu)
        
        node_num = x.shape[0]
        if x.shape[1] == 1:
            x = x.to_numpy()
            x = x.reshape(x.shape[0])
#             x = np.array(pd.get_dummies(x))
            x = np.zeros((x.shape[0],1))
        else:
            x = x.drop('node_index', axis=1).to_numpy()
#             x = np.zeros((x.shape[0],1))

        # 普通特征，并且并入中心点embedding
        x = torch.tensor(x, dtype=torch.float)

        df = data['edge_file']
        edge_index = df[['src_idx', 'dst_idx']].to_numpy()
        edge_index = sorted(edge_index, key=lambda d: d[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)

        edge_weight = df['edge_weight'].to_numpy()
        if edge_weight.shape[0] < 2000000 and x.shape[1] > 1:
            for i in range(w_mat.shape[0]):
                res = 1.0
                for j in range(len_fenwei):
                    if fenweishu[len_fenwei-1-j] < w_mat[i]:
                        res = w_fenwei[len_fenwei-1-j]
                edge_weight[i] *= res
        
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        num_nodes = x.size(0)
        y = torch.zeros(num_nodes, dtype=torch.long)
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        y[inds] = torch.tensor(train_y, dtype=torch.long)

        if isvalid:
            inds = data['test_label'][['node_index']].to_numpy()
            train_y = data['test_label'][['label']].to_numpy()
            y[inds] = torch.tensor(train_y, dtype=torch.long)         
        
        train_indices = data['train_indices']
        test_indices = data['test_indices']

        self.feature_num2 = 0
        self.feature_num2 += int(max(y)) + 1
        
        # x2 自己本身的label
        x2 = np.array(pd.get_dummies(y))
        
        x2[test_indices,:] *= 0
        x2 = np.concatenate([x2,np.zeros((x2.shape[0],1))],1)
        x2 = torch.tensor(x2, dtype=torch.float)
        self.feature_num2 = x2.shape[1]
#         x2 = np.zeros((num_nodes,self.feature_num2))
#         labeldict = {}
#         cache = data['train_label'].values
#         for i in range(cache.shape[0]):
#             labeldict[cache[i,0]] = cache[i,1]

#         cache = df[['src_idx', 'dst_idx']].values
#         for i in range(cache.shape[0]):
#             if labeldict.get(cache[i,1],-1) != -1:
#                 x2[cache[i,0],labeldict[cache[i,1]]] += 1
#             if labeldict.get(cache[i,0],-1) != -1:
#                 x2[cache[i,1],labeldict[cache[i,0]]] += 1
        
        edge_list = data['edge_file']
        train_label = data['train_label']
        n_class = int(max(y)) + 1
        crosstable = np.zeros((n_class,n_class))

        labeldict = {}
        node_d_dict = {}
        node_d_dict2 = {}
        node_d_dict3 = {}
        node_ent = {}
        node_w_dict = {}
        edge_count = {}
        special_node = []

        
        for i in range(node_num):
            node_d_dict[i] = [set(),set()]
            node_d_dict2[i] = np.zeros(n_class + 1)
            node_d_dict3[i] = np.zeros(n_class + 1)
            node_w_dict[i] = []
            labeldict[i] = -1
#             edge_count[i] = 0

        cache = train_label.values
        label_vc = {}
        for i in range(cache.shape[0]):
            labeldict[cache[i,0]] = cache[i,1]
            label_vc[cache[i,1]] = label_vc.get(cache[i,1],0) + 1

        print(time.ctime())    
        edge_list['src_label'] = edge_list['src_idx'].map(lambda x:labeldict.get(x,-1))    
        edge_list['dst_label'] = edge_list['dst_idx'].map(lambda x:labeldict.get(x,-1)) 
        cache = edge_list.values
        
#         print(edge_list['edge_weight'].value_counts().iloc[:20])
        for i in range(cache.shape[0]):
            node_d_dict[int(cache[i,0])][0].add(int(cache[i,1]))
            node_d_dict[int(cache[i,1])][1].add(int(cache[i,0]))
            
            for e in [cache[i,0],cache[i,1]]:
                edge_count[e] = edge_count.get(e,0) + 1
            la = int(cache[i,3])
            lb = int(cache[i,4])
#             node_d_dict2[int(cache[i,0])][lb] += 1
#             node_d_dict2[int(cache[i,1])][la] += 1
            if la != -1 and lb != -1:
                crosstable[la,lb] += 1

        print(time.ctime()) 
        x_class_mean_node = np.zeros((n_class + 1,n_class + 1))
        x_class_count = np.zeros((n_class + 1))
        aset_d = {}
        bset_d = {}
        for i in range(node_num):
            aset_d[i] = node_d_dict[i][1] | node_d_dict[i][0]
            bset_d[i] = node_d_dict[i][1] & node_d_dict[i][0]
            for e in aset_d[i]:
                node_d_dict2[i][labeldict[e]] += 1 
                x_class_mean_node[labeldict[i],labeldict[e]] += 1
            
            x_class_count[labeldict[i]] += 1
            temp = node_d_dict2[i][:-1].copy()
            temp[labeldict[i]] = 0
            node_ent[i] = calc_ent(node_d_dict2[i][:-1])
#             if node_ent[i] < 1.0:
#                 print(i,node_d_dict2[i][:-1],calc_ent(node_d_dict2[i][:-1]))
            

                
        # x3 一些简单特征 fc特征，连接关键节点
        # x5 关键节点连接特征 点特征
        print(time.ctime())
        x3 = []
        edge_index2 = []
        edge_weight2 = []
        for i in range(node_num):
            res = []
            aset = aset_d[i]
            bset = bset_d[i]
            res.append(len(bset) * 1.0 / (len(aset) + 0.01))
            res.append(len(node_d_dict[i][0]) * 1.0 / (len(aset) + 0.01))
            res.append(len(node_d_dict[i][1]) * 1.0 / (len(aset) + 0.01))
            x3.append(res)
            
            if len(node_d_dict[i][0]) != len(node_d_dict[i][1]):
                for node in node_d_dict[i][0] - node_d_dict[i][1]:
                    edge_index2.append([node,i])
                    edge_weight2.append(1.0)

                    

        x3 = torch.tensor(np.array(x3), dtype=torch.float)
        
        
        sum_eye = 0
        sum_eye2 = []
        for i in range(n_class):
            sum_eye += crosstable[i,i]
            sum_eye2.append(crosstable[i,i] / (np.sum(crosstable[i,:]) + 0.01))
            
        print(sum(sum_eye2)/len(sum_eye2),sum_eye2)   
#         print(train_label['label'].value_counts())
        print(crosstable)
        print(time.ctime())
        print("isconvert",sum_eye / (np.sum(crosstable) + 0.01))
        if sum_eye / (np.sum(crosstable) + 0.01) < 0.7 or sum(sum_eye2)/len(sum_eye2) < 0.2 :
            self.isconvert = True
            self.isx4 = True
            print("isconvert",sum_eye / np.sum(crosstable))
            
                                        
                    
        edge_index2 = np.array(edge_index2).T
        edge_weight2 = np.array(edge_weight2)
        edge_index2 = torch.tensor(edge_index2, dtype=torch.long)
        edge_weight2 = torch.tensor(edge_weight2, dtype=torch.float)
        edge_index = torch.cat([edge_index,edge_index2],1)
        edge_weight = torch.cat([edge_weight,edge_weight2],0)
        print(edge_index2.shape)
        
        # x6 二度信息点  fc特征
        bad_point = []
        x7 = np.zeros((node_num,n_class + 1 + n_class + 1))
        if self.isconvert and False:
            x6 = np.zeros((node_num,n_class + 1 + n_class + 1))
        else:
            x6 = np.zeros((node_num,n_class + 1 + n_class + 1 + n_class + 1 + n_class + 1))
        for i in range(node_num):
            for e in aset_d[i]:
                node_d_dict3[i] += node_d_dict2[e]
            node_d_dict3[i][labeldict[i]] -= len(aset_d[i])
            x7[i] = np.concatenate([np.log(np.clip(node_d_dict2[i] - 1,0,1) + 1),np.clip(node_d_dict2[i] - 1,0,1)/(sum(np.clip(node_d_dict2[i] - 1,0,1)) + 0.001)],axis = 0)
            if self.isconvert and False:
                x6[i] = np.concatenate([np.log(node_d_dict3[i] + 1),node_d_dict3[i]/(sum(node_d_dict3[i]) + 0.001)],axis = 0)
            else:
                x6[i] = np.concatenate([np.log(node_d_dict3[i] + 1),node_d_dict3[i]/(sum(node_d_dict3[i]) + 0.001),np.log(node_d_dict2[i] + 1),node_d_dict2[i]/(sum(node_d_dict2[i]) + 0.001)],axis = 0)
            
        x6 = torch.tensor(x6, dtype=torch.float) 
#         x8 = torch.tensor(data['fea_table'].drop('node_index', axis=1).to_numpy(), dtype=torch.float)
#         x6 = torch.cat([x6,x8],1)
        self.feature_num6 = x6.shape[1]      
        x7 = torch.tensor(x7, dtype=torch.float)
        
        # x4 fc特征，连接关键节点
        if len(edge_count.items()) > 10:
            mean_count = sorted(edge_count.items(),key = lambda x:-x[1])[int(len(edge_count.items()) * 0.2)][1]
            mean_count3 = sorted(edge_count.items(),key = lambda x:-x[1])[min(len(edge_count.items()),150)][1]
            mean_count2 = sorted(edge_count.items(),key = lambda x:-x[1])[min(30,int(len(edge_count.items()) * 0.02))][1]
            mean_count4 = sorted(node_ent.items(),key = lambda x:x[1])[node_num//10][1]
            
            
            print("mean_count",mean_count,mean_count2,mean_count3,mean_count4)
            special_node = []
            if mean_count4 != 0:
                special_node = [int(x[0]) for x in edge_count.items() if x[1] > mean_count * 2 or node_ent[int(x[0])] < min(1.0,mean_count4)]
            if len(special_node) == 0:
                special_node = [int(x[0]) for x in edge_count.items() if x[1] > mean_count * 2]
#             special_node = [int(i) for i in range(node_num) if node_ent[i] < 1.0]
            if len(special_node) > 0:
                print(len(special_node))
                x_dummy = np.zeros((x.shape[0],len(special_node)))
#                 for i in range(len(special_node)):
#                     x_dummy[int(special_node[i]),i] = 1.0
                for i in range(node_num):          # 统计每个点是否连接special
                    for node in set(special_node) & node_d_dict[i][0]:
                        x_dummy[i,special_node.index(node)] += 1.0
                    for node in set(special_node) & node_d_dict[i][1]:
                        x_dummy[i,special_node.index(node)] += 1.0
#                 x_dummy = np.log(x_dummy + 1)
                x_dummy = torch.tensor(x_dummy, dtype=torch.float)
                
                x5 = x_dummy
#                 x = torch.cat([x,x_dummy],axis = 1)
                
                special_node2 = []
                for i in range(len(special_node)):
                    node = special_node[i]
                    if edge_count[node] > mean_count2:
                        special_node2.append(i)
                print(len(special_node2))
                if len(special_node2) == 0:
                    x_dummy = torch.tensor(np.zeros((x.shape[0],1)), dtype=torch.float)
                else:
                    x_dummy = x_dummy[:,special_node2]
            else:
                x_dummy = torch.tensor(np.zeros((x.shape[0],1)), dtype=torch.float)
                x5 = torch.tensor(np.zeros((x.shape[0],1)), dtype=torch.float)
                
        else:
            x_dummy = torch.tensor(np.zeros((x.shape[0],1)), dtype=torch.float)
        
        print(x5.shape,x7.shape)
#         x5 = torch.cat([x5,x7],1)
        print(time.ctime())
        data = Data(x=x, x2=x2, x3=x3, x4 = x_dummy, edge_index=edge_index, y=y, edge_weight=edge_weight
#                     ,train_indices = train_indices,test_indices = test_indices
                    ,x5 = x5,x6 = x6
                   )
#         print(x[0],x5[0],x3[0])
        self.labeldict = labeldict
        self.label_vc = label_vc
        self.node_d_dict = node_d_dict
        self.node_d_dict2 = node_d_dict2
        self.node_d_dict3 = node_d_dict3
        self.aset_d = aset_d
        return data

    def train(self, data, isvalid = False, test_indices = None, isembed = False, isx6 = False):
        model = GCN3(features_num=data.x.size()[1], num_class=int(max(data.y)) + 1, features_num2 = data.x2.size()[1], features_num3 = data.x3.size()[1], features_num4 = data.x4.size()[1], features_num5 = data.x5.size()[1], features_num6 = data.x6.size()[1], isconvert = self.isconvert, isx4 = self.isx4, isx6 = isx6, isembed = isembed)
        print(model,int(max(data.y)) + 1,data.x.size(),self.isconvert,self.isx4,isx6)
        model = model.to(self.device)
        data = data.to(self.device)
        print(data)
        self.stop = False
#         print(data.x)
        lr = 0.005
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        #optimizer = torch.optim.Adagrad(model.parameters(), lr=0.005, weight_decay=5e-4)

        
        min_loss = float('inf')
        model.train()
        bestscore = 0.0
        bestloss = 1000.0
        bestiter = -100
        mask_alpha_list = [0.7,0.4,0.2]
        mask_alpha = 0
        val_step = 25
        ylim_list = list(range(data.edge_weight.shape[0]))
        gradient_accumulation_steps = 1
        optimizer.zero_grad()
        for epoch in range(1,1000):
#             print(epoch)
            rand_tensor = torch.rand(data.train_mask.shape).to(self.device)
            if not self.isconvert:
                node_mask = data.train_mask.float() * (rand_tensor > mask_alpha_list[mask_alpha]).float()
            else:
                node_mask = data.train_mask.float() * (rand_tensor > 0.7).float()
            node_mask2 = node_mask
#             print(node_mask,node_mask.shape)
#             print(node_mask)
            
            temp,temp2 = model(data,node_mask2)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct((temp[data.train_mask * (1 - node_mask).bool()]),data.y[data.train_mask * (1 - node_mask).bool()])
#             for i in range(int(max(data.y)) + 1):
#                 loss *= (data.y[data.train_mask * (1 - node_mask).bool()] == i) * weights[i] + 1.0
            loss = loss.mean(0)
#             print(loss.item())
            loss.backward()
            if (epoch + 1) % gradient_accumulation_steps == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.zero_grad()
            

            
            if epoch % val_step == val_step - 1:
                print("epoch:",epoch,mask_alpha)
                print(time.ctime())
                model.eval()
                with torch.no_grad():
                    node_mask = 1 - data.valid_mask.float()
                    temp = model(data,node_mask)[0]
                    temp = F.log_softmax(temp, dim=-1)
                    lossval = F.nll_loss(temp[data.valid_mask * (1 - node_mask).bool()], data.y[data.valid_mask * (1 - node_mask).bool()])
                    print(loss.item(),lossval.item())
                    pred = temp[data.train_mask].max(1)[1]
                    tp = np.sum((data.y[data.train_mask] == pred).tolist())
                    print("tp:",tp,"all:",data.y[data.train_mask].shape[0],tp/data.y[data.train_mask].shape[0])
                    pred = temp[data.valid_mask].max(1)[1]
                    tp = np.sum((data.y[data.valid_mask] == pred).tolist())
                    print("tp:",tp,"all:",data.y[data.valid_mask].shape[0],tp/data.y[data.valid_mask].shape[0])
#                     crosstable = np.zeros((int(max(data.y)) + 1,int(max(data.y)) + 1))
#                     pred = pred.tolist()
#                     label = data.y[data.valid_mask].tolist()
#                     for i in range(len(pred)):
#                         crosstable[pred[i],label[i]] += 1
#                     print((crosstable).astype(int))
                    if tp/data.y[data.valid_mask].shape[0] >= bestscore:
                        print('save')
                        self.maxepoch = max(epoch,self.maxepoch)
                        torch.save(model.state_dict(), './model.pt')
                    if lossval.item() < bestloss:
                        print('save2')
                        self.maxepoch = max(epoch,self.maxepoch)
                        torch.save(model.state_dict(), './model2.pt')
                        bestloss = lossval.item()
                    if tp/data.y[data.valid_mask].shape[0] > bestscore:
                        if tp/data.y[data.valid_mask].shape[0] > 0.7:
                            bestiter = 0
                            if self.isconvert:
                                bestiter = 1
                        bestscore = tp/data.y[data.valid_mask].shape[0]
                    else:
                        bestiter += 1
                    if tp/data.y[data.valid_mask].shape[0] < bestscore - 0.10:
                        bestiter += 100
#                     if lossval.item() > bestloss + 0.10:
#                         bestiter += 100
                model.train()
                
            if isvalid and epoch %  val_step == val_step - 1:    
                model.eval()
                with torch.no_grad():
                    pred = model(data)[0][data.test_mask].max(1)[1]
                    tp = np.sum((data.y[data.test_mask] == pred).tolist())
                    print("tp:",tp,"all:",data.y[data.test_mask].shape[0],tp/data.y[data.test_mask].shape[0])
                    
                    crosstable = np.zeros((int(max(data.y)) + 1,int(max(data.y)) + 1))
                    pred = pred.tolist()
                    label = data.y[data.test_mask].tolist()
                    badcase = 10
                    for i in range(len(pred)):
                        crosstable[pred[i],label[i]] += 1
                        if badcase > 0 and pred[i] != label[i]:
#                             print(test_indices[i],pred[i],label[i])
                            badcase -= 1
#                     print((crosstable).astype(int))
                model.train()
            
            if bestiter > 4:
                mask_alpha += 1
                bestiter = 0
#                 lr *= 0.5
#                 optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
#                 model.load_state_dict(torch.load('./model.pt',map_location=self.device))
            if mask_alpha > 2:
                break
            endTime = int(round(time.time()))
            if (endTime - self.startTime) > self.time_budget - 5:
                print('time up!',epoch,self.maxepoch)
                self.beta = epoch / self.maxepoch
                self.stop = True
                    
                break
            
        
        return model

    def pred(self, model, data):
        model.eval()
        model.load_state_dict(torch.load('./model.pt',map_location=self.device))
        data = data.to(self.device)
        with torch.no_grad():
            pred = model(data)[0][data.test_mask]
            pred = F.log_softmax(pred, dim=-1)
        model.load_state_dict(torch.load('./model2.pt',map_location=self.device))
        with torch.no_grad():
            pred2 = model(data)[0][data.test_mask]
            pred2 = F.log_softmax(pred2, dim=-1)
#             print(model(data)[data.test_mask])
#             print(pred[:10])
        if self.stop:
            pred *= self.beta
            pred2 *= self.beta
        return pred + pred2

    def train_predict(self, data, time_budget,n_class,schema, isvalid = False):
        self.startTime = int(round(time.time()))

        self.time_budget = time_budget
        preds = []
        self.maxepoch = 100.0
        datat = self.generate_pyg_data(data, isvalid = isvalid)
        for i in range(5):
            print(i)
            print(time.ctime())
            validkey = i
            train_indices = data['train_indices']
            test_indices = data['test_indices']
            num_nodes = datat.num_nodes
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_indices] = 1
            datat.train_mask = train_mask

            if True:
                valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
                valid_mask[train_indices] = 1
                train_mask[[x for x in range(num_nodes) if x % 10 == validkey]] = 0
                valid_mask[[x for x in range(num_nodes) if x % 10 != validkey]] = 0
#                 train_mask[kfx[test_x]] = 0
#                 valid_mask[kfx[train_x]] = 0
                
                for ii in range(num_nodes):
                    if self.labeldict[ii] != -1 and self.label_vc[self.labeldict[ii]] < 30 and random.random() > 0.15 and valid_mask[ii] == 1:
                        train_mask[ii] = 1
                        valid_mask[ii] = 1
                        
                datat.train_mask = train_mask
                datat.valid_mask = valid_mask


            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[test_indices] = 1
            datat.test_mask = test_mask
            
            if not self.isconvert:
                isx6 = False
            else:
                isx6 = True
            model = self.train(datat, isvalid = isvalid, test_indices = test_indices, isx6 = isx6)
            pred = self.pred(model, datat)
            preds.append(pred)
            
            if self.stop:
                break
            
        preds = sum(preds).max(1)[1].cpu().numpy().flatten()
        endTime = int(round(time.time()))
        print("end",endTime - self.startTime)
        return preds
