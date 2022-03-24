# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.nn import ARMAConv
from torch_geometric.data import Data

from util import timeclass,get_logger
import pandas as pd
import numpy as np
import random
import time
import copy
from process_data import ModelData

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

class ARMA(torch.nn.Module):

    def __init__(self,categories_nums, features_num=16, num_class=2, sparse=False):
        super(ARMA, self).__init__()
        hidden = 16
        embed_size = 8
        dropout = 0.25
        self.dropout_p = dropout
        
        self.embeddings = torch.nn.ModuleList()
        for max_nums in categories_nums:
            self.embeddings.append(Embedding(max_nums, embed_size))

        self.conv1 = ARMAConv(embed_size*len(categories_nums)+features_num, hidden, num_stacks=3,
                              num_layers=2,shared_weights=True,dropout=dropout)
        self.conv2 = ARMAConv(hidden, num_class, num_stacks=3,
                              num_layers=2,shared_weights=True,dropout=dropout)
        

    def forward(self, data):
        x, node_one_hot, edge_index, edge_weight = data.x, data.node_one_hot, data.edge_index, data.edge_weight
        
        emb_res = []
        for f,emb in zip(data.categories_value.T,self.embeddings):
            emb_res.append(emb(f))
        
        x = torch.cat( [x]+emb_res, axis=1 )
        
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x,edge_index)
        
        return F.log_softmax(x, dim=-1)
    
    def reset_parameters(self):
        self.node_emb.reset_parameters()
        self.lin1.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__
    

class ARMAModel:
    def __init__(self,num_boost_round=1001,best_iteration=100):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = 50
        self.best_iteration = best_iteration
        self.learning_rate = 0.005
        self.echo_epochs = 20
        
    def init_model(self,model_data,table):
        data = model_data.arma_data
        sparse = table.sparse
        categories_nums = data.categories_nums
        model = ARMA(categories_nums,features_num=data.x.size()[1], num_class=table.n_class,sparse=sparse)
        
        model = model.to(self.device)
        data = data.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        
        return model,data,optimizer
        
    @timeclass('ARMAModel')
    def train_and_valid(self, model_data, table, seed=None):
        model,data,optimizer = self.init_model(model_data,table)
        
        pre_time = time.time()
        best_acc = 0
        best_epoch = 0
        best_model = copy.deepcopy(model)
        best_pred_matrix = None
        keep_epoch = 0
        model.train()
        for epoch in range(1,self.num_boost_round+1):
            model.train()
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
        return best_epoch, float(best_acc.cpu().numpy()), best_pred_matrix.cpu().detach().exp().numpy()

        
    @timeclass('ARMAModel')
    def train(self, model_data,table, seed=None):
        model,data,optimizer = self.init_model(model_data,table)
        model.train()
        pre_time = time.time()
        for epoch in range(1,self.best_iteration+1):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])  
            if epoch%self.echo_epochs==0:
                now_time = time.time()
                print(f'epoch:{epoch} [train loss]: , {loss.data}, use time: {now_time-pre_time}s')
                pre_time = now_time
            loss.backward()
            optimizer.step()
        
        self.model = model
        return output[data.test_mask].cpu().detach().exp().numpy()

    @timeclass('ARMAModel')
    def predict(self, model_data):
        data = model_data.arma_data
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            preds_matrix = self.model(data)[data.test_mask].cpu().detach().exp().numpy()
            preds = preds_matrix.argmax(axis=1).flatten()
        return preds,preds_matrix

    @timeclass('ARMAModel')
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
        LOGGER.info(f'init_time:{init_time},one_epoch_time:{one_epoch_time}')
        return init_time, one_epoch_time

    @timeclass('ARMAModel')
    def get_train_and_valid(self,table,train_valid_idx,valid_idx,seed=None):
        #划分训练集和验证集
        valid_model_data = ModelData()

        #获取gcn数据
        data = table.arma_data.clone()
        num_nodes = data.y.shape[0]
        
        valid_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_train_mask[train_valid_idx] = 1
        data.valid_train_mask = valid_train_mask
        
        valid_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_test_mask[valid_idx] = 1
        data.valid_test_mask = valid_test_mask
        
        valid_model_data.arma_data = data
        
        return valid_model_data
    
    @timeclass('ARMAModel')
    def get_train(self,table,train_idx,test_idx,seed=None):
        #划分训练集和测试集
        model_data = ModelData()
        
        #获取gcn数据
        data = table.arma_data.clone()
        num_nodes = data.y.shape[0]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = 1
        data.train_mask = train_mask
        
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_idx] = 1
        data.test_mask = test_mask
        
        model_data.arma_data = data
        
        return model_data
