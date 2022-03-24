# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_adj
from torch_geometric.data import Data
from util import timeclass
import pandas as pd
import numpy as np
import random
import time
import copy
from process_data import ModelData

class GUNET(torch.nn.Module):

    def __init__(self, features_num=16, num_class=2, node_num=10000, sparse=False):
        super(GUNET, self).__init__()
        hidden = 16
        embed_size = 8
        dropout = 0.25
        self.dropout_p = dropout
        
        self.node_emb = Embedding(node_num, embed_size)
        
        pool_ratios = [2000 / node_num, 0.5]
        self.unet = GraphUNet(embed_size+features_num, hidden, num_class,
                              depth=2, pool_ratios=pool_ratios)

    def forward(self, data):
        x, node_index, node_one_hot, edge_index, edge_weight = data.x, data.node_index, data.node_one_hot, data.edge_index, data.edge_weight
        
        node_embedding = self.node_emb(node_index)
        
        x = torch.cat( [node_embedding,x], axis=1 )
        
        edge_index, _ = dropout_adj(edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        x = F.dropout(x, p=0.92, training=self.training)

        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
    

class GUNETModel:
    def __init__(self,num_boost_round=1001,best_iteration=100):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        
        self.num_boost_round = 1000
        self.early_stopping_rounds = 50
        self.best_iteration = best_iteration
        self.learning_rate = 0.01
        self.echo_epochs = 20
    
    @timeclass('GUNETModel')
    def init_model(self,model_data,table):
        data = model_data.gunet_data
        sparse = True if data.num_nodes**2*0.01>data.num_edges else False
        model = GUNET(features_num=data.x.size()[1], num_class=table.n_class, node_num=data.num_nodes,sparse=sparse)
        
        model = model.to(self.device)
        data = data.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        
        return model,data,optimizer
        
    @timeclass('GUNETModel')
    def train_and_valid(self, model_data, table):
        model,data,optimizer = self.init_model(model_data,table)
        pre_time = time.time()
        
        best_acc = 0
        best_epoch = 0
        best_model = copy.deepcopy(model)
        best_pred_matrix = None
        keep_epoch = 0
        model.train()
        for epoch in range(1,self.num_boost_round+1):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss( output[data.valid_train_mask], data.y[data.valid_train_mask] )
            acc = (output[data.valid_test_mask].max(1)[1]==data.y[data.valid_test_mask]).sum().float() / len(data.y[data.valid_test_mask])
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                best_pred_matrix = output[data.valid_test_mask]
                keep_epoch = 0
            else:
                keep_epoch += 1
                if keep_epoch > self.early_stopping_rounds:
                    break
            if epoch%self.echo_epochs==0:
              now_time = time.time()
              print(f'epoch:{epoch} [train loss]: {loss.data} [valid acc]: {acc} [use time]: {now_time-pre_time} s')
              pre_time = now_time
            loss.backward()
            optimizer.step()
        self.model = best_model
        return best_epoch, float(best_acc.cpu().numpy()), best_pred_matrix.cpu().detach().numpy()

    @timeclass('GUNETModel')
    def train(self, model_data,table):
        model,data,optimizer = self.init_model(model_data,table)
        pre_time = time.time()
        model.train()
        for epoch in range(1,self.best_iteration):
            optimizer.zero_grad()
            loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])  
            if epoch%self.echo_epochs==0:
                now_time = time.time()
                print(f'epoch:{epoch} [train loss]: ', loss.data,' use time: ',now_time-pre_time,'s')
                pre_time = now_time
            loss.backward()
            optimizer.step()
        
        self.model = model

    @timeclass('GUNETModel')
    def predict(self, model_data):
        data = model_data.gunet_data
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            preds_matrix = self.model(data)[data.test_mask].cpu().exp().numpy()
            preds = preds_matrix.argmax(axis=1).flatten()
        return preds,preds_matrix

    @timeclass('GUNETModel')
    def get_run_time(self, model_data, table):
        t1 = time.time()
        model,data,optimizer = self.init_model(model_data,table)
        t2 = time.time()
        for epoch in range(1,2):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss( output[data.train_mask], data.y[data.train_mask] )
            loss.backward()
            optimizer.step()
        t3 = time.time()
        init_time = t2-t1
        one_epoch_time = t3-t2
        return init_time, one_epoch_time
    
    @timeclass('GUNETModel')
    def get_train_and_valid(self,table,train_rate=0.7,seed=None):
        #划分训练集和验证集
        valid_model_data = ModelData()
        train_idx,valid_idx = valid_model_data.split_train_and_valid(table,train_rate=train_rate,seed=seed)

        #获取gat数据
        data = table.gunet_data.clone()
        num_nodes = data.y.shape[0]
        
        valid_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_train_mask[train_idx] = 1
        data.valid_train_mask = valid_train_mask
        
        valid_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_test_mask[valid_idx] = 1
        data.valid_test_mask = valid_test_mask
        
        valid_model_data.gunet_data = data
        
        return valid_model_data
    
    @timeclass('GUNETModel')
    def get_train(self,table,seed=None):
        #划分训练集和测试集
        all_train = table.df.loc[table.df['is_test']==0]
        all_test = table.df.loc[table.df['is_test']==1]
        train_idx = all_train.index
        test_idx = all_test.index
        
        #训练数据类
        model_data = ModelData()
        
        #获取gat数据
        data = table.gunet_data.clone()
        num_nodes = data.y.shape[0]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = 1
        data.train_mask = train_mask
        
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_idx] = 1
        data.test_mask = test_mask
        
        model_data.gunet_data = data
        
        return model_data
