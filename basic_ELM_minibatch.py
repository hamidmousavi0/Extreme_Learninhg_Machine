# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:18:44 2018

@author: Hamid
"""
import numpy as np
import math
def ELM_train_minibatch(X_train,Y_train,num_hid,C,keep_prob = 0.5):
    X_new = np.zeros((X_train.shape[0],X_train.shape[1]+1))
    X_new[:,0:X_train.shape[1]]=X_train
    X_new[:,X_train.shape[1]]=1
    parameter=initialize_parameters_random(X_train.shape[1],num_hid)
    W=parameter['W']
    b=parameter['b']
    W_new = np.zeros((X_train.shape[1]+1,num_hid))
    W_new[0:X_train.shape[1],:]=W
    W_new[X_train.shape[1],:]=b
    temp_H=(np.dot(X_new,W_new))
    H=sigmoid(temp_H)
    inverse_H=np.linalg.inv(H)
    Beta_hat=np.dot(inverse_H,Y_train)
    return W_new,Beta_hat
#%%
def ELM_test_minibatch(X,W_new,Beta_hat):
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
def initialize_parameters_random(num_X,num_hid):             
    parameters = {}
#    parameters['W'] = np.random.uniform(np.sqrt(-6/num_hid+num_X),np.sqrt(6/num_hid+num_X),(num_X,num_hid))
    parameters['W'] = np.random.randn(num_X,num_hid)
    parameters['b'] = np.random.randn(1,num_hid)
    return parameters   
#%%    
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                 
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:].reshape((m,Y.shape[1]))
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
#%%    