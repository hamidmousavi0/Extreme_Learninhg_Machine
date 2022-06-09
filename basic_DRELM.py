# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:19:42 2018

@author: Hamid
"""
import numpy as np 
from basic_elm import ELM_train,ELM_test,initialize_parameters_random,sigmoid,softmax
def DrElm_train(X_train,Y_train,hid_num,alpha,C,num_iteration):
    W_list=[]
    Beta_list=[]
    W_prime_list=[]
    for i in range (num_iteration):
        W,Beta,O=ELM_train(X_train,Y_train,hid_num,C)
        W_list.append(W)
        Beta_list.append(Beta)
        parameter=initialize_parameters_random(Y_train.shape[1],X_train.shape[1])
        W_prime=parameter['W']
        W_prime_list.append(W_prime)
        X_train=sigmoid(X_train+(alpha*np.dot(O,W_prime)))
    return W_list,Beta_list,W_prime_list
#%%
def DrElm_test(X_test,Y_train,hid_num,alpha,W_list,Beta_list,W_prime_list):
    for i in range(len(W_list)):
        O=ELM_test(X_test,W_list[i],Beta_list[i])
        X_test=sigmoid(X_test+(alpha*np.dot(O,W_prime_list[i])))
    return O  
#%%
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ         