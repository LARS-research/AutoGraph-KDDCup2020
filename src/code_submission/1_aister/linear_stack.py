#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:03:22 2020

@author: tang
"""
import numpy as np
#import copy

class linear_stack:
    
    def __init__(self, feature_size, W, iters = 100, rate_max = 0.003, rate_min = 0.0001, verbose = 0, col_rate = 0.75):
        self.W = W#[1.0/feature_size]*feature_size
        self.iters = iters
        self.rate_max = rate_max
        self.rate_min = rate_min
        self.rate = rate_max
        
        self.verbose = verbose
        self.coef_ = None
        self.col_rate = col_rate

        
    def fit(self, X, Y):
        #X=(models,n,class),Y=(n)
        rate_list = [self.rate_max-(self.rate_max-self.rate_min)/(self.iters-1)*x for x in range(self.iters)]
        print(f'iter:0, acc:{self.cal_acc(X, Y, self.W)}, rate:0, weight:{self.W}')
        for it in range(self.iters):
            flg =False
#            self.rate = rate_list[it]
            self.rate = max(np.random.rand()*self.rate_max,self.rate_min)
            self.rate_max *=0.99
            
#            print(f'rate {self.rate:.5f}')
            
            arr = np.arange(len(self.W))
            np.random.shuffle(arr)
            
#            t = np.random.choice(len(X1), len(X1)*self.col_rate)
#            X = X1[t]
#            Y = Y1[t]

            #for i in range(len(self.W)):
            for i in arr:
                w1 = self.W[:]
                w2 = self.W[:]
    
                s0 = self.cal_acc(X, Y, self.W)
                
#                sd = np.random.randint(0,2)
#                print (sd)
#                if sd == 0:
                
                w1 = self.sub_wei(w1, i, min(self.rate,w1[i]))
                s1 = self.cal_acc(X, Y, w1)
                if s1 > s0:
                    self.W = w1[:]
                    s0 = s1
                    flg = True
                
                w2 = self.W[:]
                w2 = self.add_wei(w2, i, self.rate)
                s2 = self.cal_acc(X, Y, w2)
                if s2 > s0:
                    self.W = w2[:]
                    s0 = s2
                    flg = True
                    
            if self.verbose == 1:
                if flg:
                    print(f'iter:{it}, 【acc】:{s0}, rate:{self.rate:.5f}, weight:{self.W}')
                else:
                    print(f'iter:{it}, acc:{s0}, rate:{self.rate:.5f}, weight:{self.W}')
        #t = np.array(self.W)
        #self.W = (t - t.min())/(t.max() - t.min())    
        self.coef_  = self.W        
            
    def my_score(self, ps, y, coef):
        coef = np.array(coef)
        coef = coef/coef.sum()
        preds = ps[:,0]*coef[0]
        for i in range(1, len(coef)):
            preds = preds + ps[:,i]*coef[i]
        return abs(y-preds).sum()/len(y)
        
        
        
    def cal_acc(self, ps, y, coef):
        coef = np.array(coef)
        coef = coef/coef.sum()
        preds = ps[0,:,:]*coef[0]
        for i in range(1, len(coef)):
            preds = preds + ps[i,:,:]*coef[i]
        
        return (preds.argmax(axis=1).flatten()==y).mean()
        
        
    def sub_wei(self, w, i, rate = 0.003):
        w[i] -= rate
        sum_w = sum(w)
        for j in range(len(w)):
            w[j] /= sum_w
#        r = rate/(len(w)-1)
#        for j in range(len(w)):
#            if j != i:
#                w[j] += r
        return w
    
        
    def add_wei(self, w, i, rate = 0.003):
        w[i] += rate
        sum_w = sum(w)
        for j in range(len(w)):
            w[j] /= sum_w
        
#        r = rate/(len(w) - 1)
#        for j in range(len(w)):
#            if j != i:
#                w[j] -= r
        return w
    
    def predict(self, X):
        coef = np.array(self.W)
        coef = coef/coef.sum()
        preds = X[0,:,:]*coef[0]
        for i in range(1, len(coef)):
            preds = preds + X[i,:,:]*coef[i]
            
        return preds#.argmax(axis=1).flatten()