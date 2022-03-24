# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.nn import SAGEConv, GATConv, JumpingKnowledge, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data, DataLoader, DataListLoader, NeighborSampler
from util import timeclass
import pandas as pd
import numpy as np
import random
import time
import copy
from process_data import ModelData


class SAGE(torch.nn.Module):

    def __init__(self, num_layers=2, hidden=64,  features_num=16, num_class=2, node_num=10000):
        super(SAGE, self).__init__()
        embed_size = 8
        self.node_emb = Embedding(node_num, embed_size)
        if features_num==0:
            self.lin1 = Linear(embed_size+features_num+node_num, hidden)
            self.conv1 = SAGEConv(embed_size+features_num+node_num, hidden)
        else:
            self.lin1 = Linear(embed_size+features_num, hidden)
            self.conv1 = SAGEConv(embed_size+features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        self.lin1.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    
    def forward(self, data_x, data_node_index, data_node_one_hot, data_flow):
        x, node_index, node_one_hot = data_x, data_node_index, data_node_one_hot 
        block = data_flow[0]
        x = x[block.n_id]
        node_embedding = self.node_emb( node_index[block.n_id] )
        node_one_hot = node_one_hot[block.n_id]
        if x.shape[1]==0:
            x = torch.cat( [node_embedding,x,node_one_hot], axis=1 )
        else:
            x = torch.cat( [node_embedding,x], axis=1 )
        x = self.conv1( (x, None), block.edge_index, size=block.size,)
#                       res_n_id=block.res_n_id)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        for idx,conv in enumerate(self.convs):
            block = data_flow[1+idx]
            x = conv( (x, None), block.edge_index, size=block.size,)
#                       res_n_id=block.res_n_id)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class SAGEModel:
    def __init__(self,num_boost_round=1001,best_iteration=100):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = 50
        self.best_iteration = best_iteration
        self.learning_rate = 0.005
        self.echo_epochs = 20
        self.batch_size = 500
        self.sample_size = [0.8,0.2]
        self.num_hops = 2
        
    @timeclass('SAGEModel') 
    def init_model(self,model_data,table):
        data = model_data.sage_data
        model = SAGE(features_num=data.x.size()[1], num_class=table.n_class, node_num=data.num_nodes)
        
        model = model.to(self.device)
#        data = data.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        
        return model,data,optimizer
    
    @timeclass('SAGEModel')
    def train_and_valid(self, model_data, table):
        model,data,optimizer = self.init_model(model_data,table)
        
        loader_1 = NeighborSampler(data, size=self.sample_size, num_hops=self.num_hops,
                                     batch_size=self.batch_size, add_self_loops=True)
        loader_2 = NeighborSampler(data, size=self.sample_size, num_hops=self.num_hops,
                             batch_size=len(data.valid_test_mask), add_self_loops=True)
        pre_time = time.time()
        best_acc = 0
        best_epoch = 0
        best_model = copy.deepcopy(model)
        best_pred_matrix = None
        keep_epoch = 0
        model.train()
        
        for epoch in range(self.num_boost_round):
            for data_flow in loader_1(data.valid_train_mask):
                optimizer.zero_grad()
                output = model(data.x.to(self.device), data.node_index.to(self.device),
                               data.node_one_hot.to(self.device), data_flow.to(self.device))
                loss = F.nll_loss( output, data.y[data_flow.n_id].to(self.device) )
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                for data_flow in loader_2(data.valid_test_mask):
                    output = model(data.x.to(self.device), data.node_index.to(self.device),
                                   data.node_one_hot.to(self.device), data_flow.to(self.device))
            acc = (output.max(1)[1]==data.y[data.valid_test_mask].to(self.device)).sum().float() \
                   / len(data.y[data.valid_test_mask].to(self.device))
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                best_pred_matrix = output
                keep_epoch = 0
            else:
                keep_epoch += 1
                if keep_epoch > self.early_stopping_rounds:
                    break
            
            now_time = time.time()
            if epoch % self.echo_epochs == 0:
                print(f'epoch:{epoch} [train loss]: {loss.data} [valid acc]: {acc} [each epoch use time]: {now_time-pre_time} s')
            pre_time = now_time
        
        print(f'best_epoch: {best_epoch} , best_acc: {best_acc}')
        self.model = best_model
        return best_epoch, float(best_acc.cpu().numpy()), best_pred_matrix.cpu().detach().numpy()

        
    @timeclass('SAGEModel')
    def train(self, model_data, table):
        model,data,optimizer = self.init_model(model_data,table)
        
        loader = NeighborSampler(data, size=self.sample_size, num_hops=self.num_hops,
                                     batch_size=self.batch_size, add_self_loops=True)
        
        pre_time = time.time()
        model.train()
        for epoch in range(self.best_iteration):
            for data_flow in loader(data.train_mask):
                optimizer.zero_grad()
                output = model(data.x.to(self.device), data.node_index.to(self.device),
                               data.node_one_hot.to(self.device), data_flow.to(self.device))
                loss = F.nll_loss( output, data.y[data_flow.n_id].to(self.device) )
                loss.backward()
                optimizer.step()
            if epoch%self.echo_epochs==0:
                now_time = time.time()
                print(f'epoch:{epoch} [last batch train loss]: ', loss.data,' use time: ',now_time-pre_time,'s')
                pre_time = now_time
        self.model = model
            
    @timeclass('SAGEModel')
    def predict(self, model_data):
        data = model_data.sage_data
        loader = NeighborSampler(data, size=self.sample_size, num_hops=self.num_hops,
                                     batch_size=len(data.test_mask), add_self_loops=True)
        self.model.eval()
        with torch.no_grad():
            for data_flow in loader(data.test_mask):
                preds_matrix = self.model(data.x.to(self.device), data.node_index.to(self.device),
                               data.node_one_hot.to(self.device), data_flow.to(self.device)).cpu().numpy()
                preds = preds_matrix.argmax(axis=1).flatten()
        return preds, preds_matrix

    @timeclass('SAGEModel')
    def get_run_time(self, model_data, table):
        t1 = time.time()
        model,data,optimizer = self.init_model(model_data,table)
        
        loader = NeighborSampler(data, size=self.sample_size, num_hops=self.num_hops,
                                     batch_size=self.batch_size, add_self_loops=True)
        
        t2 = time.time()
        model.train()
        for epoch in range(1):
            for data_flow in loader(data.train_mask):
                optimizer.zero_grad()
                output = model(data.x.to(self.device), data.node_index.to(self.device),
                               data.node_one_hot.to(self.device), data_flow.to(self.device))
                loss = F.nll_loss( output, data.y[data_flow.n_id].to(self.device) )
                loss.backward()
                optimizer.step()
        t3 = time.time()
        init_time = t2-t1
        one_epoch_time = t3-t2
        return init_time, one_epoch_time
    
    @timeclass('SAGEModel')
    def get_train_and_valid(self,table,train_rate=0.7,seed=None):
        #划分训练集和验证集
        valid_model_data = ModelData()
        train_idx,valid_idx = valid_model_data.split_train_and_valid(table,train_rate=train_rate,seed=seed)

        #获取gat数据
        data = table.sage_data.clone()
        num_nodes = data.y.shape[0]
        
        valid_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_train_mask[train_idx] = 1
        data.valid_train_mask = valid_train_mask
        
        valid_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_test_mask[valid_idx] = 1
        data.valid_test_mask = valid_test_mask
        
        valid_model_data.sage_data = data
        
        return valid_model_data
    
    @timeclass('SAGEModel')
    def get_train(self,table,seed=None):
        #划分训练集和测试集
        all_train = table.df.loc[table.df['is_test']==0]
        all_test = table.df.loc[table.df['is_test']==1]
        train_idx = all_train.index
        test_idx = all_test.index
        
        #训练数据类
        model_data = ModelData()
        
        #获取gat数据
        data = table.sage_data.clone()
        num_nodes = data.y.shape[0]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = 1
        data.train_mask = train_mask
        
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_idx] = 1
        data.test_mask = test_mask
        
        model_data.sage_data = data
        
        return model_data
                    
   