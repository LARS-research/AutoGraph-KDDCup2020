# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.nn import DNAConv
from torch_geometric.data import Data
from util import timeclass
import pandas as pd
import numpy as np
import random
import time
import copy

class DNA(torch.nn.Module):

    def __init__(self, features_num=16, num_class=2, node_num=10000, sparse=False, num_layers=2,groups=1):
        super(DNA, self).__init__()
        embed_size = 8
        dropout = 0.2
        
        if sparse:
            print('---sparse---')
            hidden = 16
            heads = 4
            num_layers = 5
        else:
            print('---no sparse---')
            hidden = 8
            groups = 8
            heads = 2
            num_layers = 3
        
        self.node_emb = Embedding(node_num, embed_size)
        self.dropout_p = dropout
        
        self.hidden = hidden
        self.lin1 = torch.nn.Linear(embed_size+features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                DNAConv(
                    hidden, heads, groups, dropout=dropout, cached=True))
        self.lin2 = torch.nn.Linear(hidden, num_class)

    def forward(self, data):
        x, node_index, node_one_hot, edge_index, edge_weight = data.x, data.node_index, data.node_one_hot, data.edge_index, data.edge_weight
        
        node_embedding = self.node_emb(node_index)
        
        x = torch.cat( [node_embedding,x], axis=1 )
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x_all = x.view(-1, 1, self.hidden)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
    

class DNAModel:
    def __init__(self,best_iteration=100):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        
        self.num_boost_round = 1000
        self.early_stopping_rounds = 100
        self.best_iteration = best_iteration
        self.learning_rate = 0.005
        self.echo_epochs = 20
        
    @timeclass('DNAModel')
    def train_and_valid(self, model_data, table):
        data = model_data.dna_data
        sparse = True if data.num_nodes**2*0.01>data.num_edges else False
        if not sparse:
            self.learning_rate = 0.05
        model = DNA(features_num=data.x.size()[1], num_class=table.n_class, node_num=data.num_nodes, sparse=sparse, num_layers=5,groups=16)
        model = model.to(self.device)
        data = data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
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

    @timeclass('DNAModel')
    def train(self, model_data,table):
        data = model_data.dna_data
        sparse = True if data.num_nodes**2*0.01>data.num_edges else False
        if not sparse:
            self.learning_rate = 0.05
        model = DNA(features_num=data.x.size()[1], num_class=table.n_class, node_num=data.num_nodes, sparse=sparse, num_layers=5,groups=16)
        model = model.to(self.device)
        data = data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
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

    @timeclass('DNAModel')
    def predict(self, model_data):
        data = model_data.dna_data
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            preds_matrix = self.model(data)[data.test_mask].cpu().exp().numpy()
            preds = preds_matrix.argmax(axis=1).flatten()
        return preds,preds_matrix

    @timeclass('DNAModel')
    def get_run_time(self, model_data, table):
        t1 = time.time()
        data = model_data.dna_data
        data.x.requires_grad = True
        model = DNA(features_num=data.x.size()[1], num_class=table.n_class, node_num=data.num_nodes, sparse=sparse, num_layers=5,groups=16)
        model = model.to(self.device)
        data = data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
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
