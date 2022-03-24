import lightgbm as lgb
import numpy as np
import pandas as pd
from util import timeclass, get_logger
import time
from process_data import ModelData,get_model_input

import warnings
warnings.filterwarnings('ignore')


VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

class LGBModel:
    def __init__(self,num_boost_round=1001,best_iteration=100):
        self.params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "metric": "multi_error",
            "verbosity": -1,
            "seed": 2020,
            "num_threads": 4
        }

        self.hyperparams = {
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin':255,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'subsample_for_bin': 200000,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = 50
        self.best_iteration = best_iteration
        self.learning_rate = 0.1
    
    @timeclass('LGBModel')
    def train_and_valid(self,model_data,table,seed=None):
        self.params["num_class"] = table.n_class
        df_train,df_valid,y_train,y_valid = model_data.lgb_data

        categories = table.categories
        
        self.columns = list(df_train.columns)
        train_data = lgb.Dataset(df_train, label=y_train.values,free_raw_data=False)
        valid_data =  lgb.Dataset(df_valid, label=y_valid.values,free_raw_data=False)
        
        params = self.params
        hyperparams = self.hyperparams
        params['seed'] = seed
        
        model = lgb.train({**params, **hyperparams}, train_data,
                          num_boost_round=self.num_boost_round,
                          learning_rate = self.learning_rate,
                          valid_sets=[train_data,valid_data],
                          early_stopping_rounds=self.early_stopping_rounds,
                          feature_name=self.columns,
                          categorical_feature=categories,
                          verbose_eval=10,
                          )
        
        importances = pd.DataFrame({'features':model.feature_name(),
                                'importances':model.feature_importance()})
            
        importances.sort_values('importances',ascending=False,inplace=True)
        print(importances.shape[0])
        print(importances)
        self.model = model
        return model.best_iteration,1-model.best_score['valid_1']['multi_error'],model.predict(df_valid)
    
    @timeclass('LGBModel')
    def train(self,model_data,table,seed=None):
        self.params["num_class"] = table.n_class
        df_train,all_test,y_train = model_data.lgb_data
        categories = table.categories
        
        self.columns = list(df_train.columns)
        train_data = lgb.Dataset(df_train, label=y_train.values,free_raw_data=False)
        
        params = self.params
        hyperparams = self.hyperparams
        params['seed'] = seed
        
        model = lgb.train({**params, **hyperparams}, train_data,
                          num_boost_round=self.best_iteration,
                          valid_sets=[train_data],
                          feature_name=self.columns,
                          categorical_feature=categories,
                          verbose_eval=10,
                          )
        
        importances = pd.DataFrame({'features':model.feature_name(),
                                'importances':model.feature_importance()})
            
        importances.sort_values('importances',ascending=False,inplace=True)
        print(importances.shape[0])
        print(importances)
        self.model = model
        
        return model.predict(all_test)
    
    @timeclass('LGBModel')
    def predict(self,model_data):
        _,all_test,_ = model_data.lgb_data
        preds_matrix = self.model.predict(all_test)
        preds = preds_matrix.argmax(axis=1).flatten()
        return preds,preds_matrix
    
    @timeclass('LGBModel')
    def get_run_time(self, model_data, table):
        t1 = time.time()
        self.params["num_class"] = table.n_class
        df_train,all_test,y_train = model_data.lgb_data
        categories = table.categories
        
        self.columns = list(df_train.columns)
        train_data = lgb.Dataset(df_train, label=y_train.values,free_raw_data=False)
        
        params = self.params
        hyperparams = self.hyperparams
        
        t2 = time.time()
        
        model = lgb.train({**params, **hyperparams}, train_data,
                          num_boost_round=1,
                          valid_sets=[train_data],
                          feature_name=self.columns,
                          categorical_feature=categories,
                          verbose_eval=5,
                          )
        
        t3 = time.time()
        
        init_time = t2-t1
        one_epoch_time = (t3-t2)/5/2
        
        LOGGER.info(f'init_time:{init_time},one_epoch_time:{one_epoch_time}')
        return init_time, one_epoch_time
    
    @timeclass('LGBModel')
    def get_train_and_valid(self,table,train_valid_idx,valid_idx,seed=None):
        #划分训练集和验证集
        valid_model_data = ModelData()
        #获取lgb数据
        df_train = table.df.iloc[train_valid_idx][table.lgb_columns]
        df_valid = table.df.iloc[valid_idx][table.lgb_columns]
        
        df_train = get_model_input(df_train)
        df_valid = get_model_input(df_valid)
        
        y_train = df_train.pop('label')
        y_valid = df_valid.pop('label')
        
        valid_model_data.lgb_data = (df_train,df_valid,y_train,y_valid)
        
        return valid_model_data
    
    @timeclass('LGBModel')
    def get_train(self,table,train_idx,test_idx,seed=None):
        #训练数据类
        model_data = ModelData()
        all_train = table.df.loc[train_idx]
        all_test = table.df.loc[test_idx]
        
        #获取lgb数据
        all_train = all_train[table.lgb_columns].reset_index(drop=True)
        all_test = all_test[table.lgb_columns].reset_index(drop=True)
        
        all_train = all_train.sample(frac=1,random_state=seed,axis=0)
        
        all_train = get_model_input(all_train)
        all_test = get_model_input(all_test)
        
        y_train = all_train.pop('label')
        all_test.pop('label')
        
        model_data.lgb_data = (all_train,all_test,y_train)
        
        return model_data
    
    @timeclass('LGBModel')
    def get_lr(self,lr_one,model_data,table,seed=None):
        t1 = time.time()
        self.params["num_class"] = table.n_class
        df_train,all_test,y_train = model_data.lgb_data
        
        
        
        self.learning_rate = lr_one
        model,data,optimizer,scheduler = self.init_model(model_data,table)
        loss_list = np.zeros(table.lr_epoch)
        valid_loss_list = np.zeros(table.lr_epoch)
        acc_list = np.zeros(table.lr_epoch)
        
        LOGGER.info(f'learning rate:{lr_one}')
        model.train()
        for epoch in range(table.lr_epoch):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss( output[data.valid_train_mask], data.y[data.valid_train_mask] )
            valid_loss = F.nll_loss( output[data.valid_test_mask], data.y[data.valid_test_mask] )
            acc = (output[data.valid_test_mask].max(1)[1]==data.y[data.valid_test_mask]).sum().float() / len(data.y[data.valid_test_mask])
            
            loss_list[epoch] = loss.item()
            valid_loss_list[epoch] = valid_loss.item()
            acc_list[epoch] = acc.item()
            
            print(f'[{epoch+1}/{table.lr_epoch}] train loss:{loss.data}, valid loss:{valid_loss.data}, valid acc:{acc.data}')
            loss.backward()
            optimizer.step()
        k = 3
#        return loss.item(),valid_loss.item(),acc.item()
        return loss_list[-k:].mean(),valid_loss_list[-k:].mean(), acc_list[-k:].mean()

#    @timeclass('LGBModel')
#    def get_run_time(self, model_data, table):
#        t1 = time.time()
#        self.params["num_class"] = table.n_class
#        df_train,all_test,y_train = model_data.lgb_data
#        categories = table.categories
#        
#        self.columns = list(df_train.columns)
#        train_data = lgb.Dataset(df_train, label=y_train.values,free_raw_data=False)
#        
#        params = self.params
#        hyperparams = self.hyperparams
#        
#        t2 = time.time()
#        
#        model = lgb.train({**params, **hyperparams}, train_data,
#                          num_boost_round=1,
#                          valid_sets=[train_data],
#                          feature_name=self.columns,
#                          categorical_feature=categories,
#                          verbose_eval=5,
#                          )
#        
#        t3 = time.time()
#        
#        init_time = t2-t1
#        one_epoch_time = (t3-t2)/5/2
#        
#        LOGGER.info(f'init_time:{init_time},one_epoch_time:{one_epoch_time}')
#        return init_time, one_epoch_time