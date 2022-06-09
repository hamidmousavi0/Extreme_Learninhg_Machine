# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:42:10 2018

@author: Hamid
"""
from main_PCG import main
import numpy as np
def ELM_train(X_train,Y_train,num_hid,C):
    X_train_new = np.zeros((X_train.shape[0],X_train.shape[1]+1))
    X_train_new[:,0:X_train.shape[1]]=X_train
    X_train_new[:,X_train.shape[1]]=1
    parameter=initialize_parameters_random(X_train.shape[1],num_hid)
    W=parameter['W']
    b=parameter['b']
    W_new = np.zeros((X_train.shape[1]+1,num_hid))
    W_new[0:X_train.shape[1],:]=W
    W_new[X_train.shape[1],:]=b
    temp_H=(np.dot(X_train_new,W_new))
    H=sigmoid(temp_H)
    Beta_hat=main(H,Y_train)
    Y=np.dot(H,Beta_hat)
    Y=(Y)
    return W_new,Beta_hat,Y
#%%
def ELM_test(X,W_new,Beta_hat):
    X_new = np.zeros((X.shape[0],X.shape[1]+1))
    X_new[:,0:X.shape[1]]=X
    X_new[:,X.shape[1]]=1
    temp_H=(np.dot(X_new,W_new))
    h=sigmoid(temp_H)
    Y_predict=np.dot(h,Beta_hat)
    Y_predict=(Y_predict)
    return Y_predict  
#%%
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A
#%%
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=1,keepdims = True)
    return A 
#%%
def gaussian(Z):
     A=np.exp(-pow(Z,2.0))
     return A
#%%
def relu(Z):
    A = np.maximum(0,Z)
    return A    
#%%
def initialize_parameters_random(num_X,num_hid):             
    parameters = {}
#    parameters['W'] = np.random.uniform(np.sqrt(-6/num_hid+num_X),np.sqrt(6/num_hid+num_X),(num_X,num_hid))
    parameters['W'] = np.random.randn(num_X,num_hid)
    parameters['b'] = np.random.randn(1,num_hid)
    return parameters   
#%%    