# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:15:17 2018

@author: Hamid
"""
import numpy as np
from basic_elm import ELM_train,ELM_test,initialize_parameters_random,sigmoid,softmax
def DrElm_train(X_train,Y_train,hid_num,C,num_iterations):
    W_list=[]
    Beta_list=[]
    O_new=np.zeros_like(Y_train)
    for i in range (num_iterations):
        W,Beta,O=ELM_train(X_train,Y_train,hid_num,C)
        W_list.append(W)
        Beta_list.append(Beta)
        O_new+=O
        #O_new=np.zeros_like(O)
       # max_prob=np.argmax(O,axis=1)
        #for i in range(len(max_prob)):
        #    O_new[i,max_prob[i]]=1
        X_train=np.column_stack([X_train,O_new])
    return W_list,Beta_list
#%%
def DrElm_test(X_test,W_list,Beta_list,num_class):
    O_new=np.zeros((X_test.shape[0],num_class))
    for i in range(len(W_list)):
        O=ELM_test(X_test,W_list[i],Beta_list[i])
        O_new+=O
       # O_new=np.zeros_like(O)
        #max_prob=np.argmax(O,axis=1)
       # for i in range(len(max_prob)):
        #    O_new[i,max_prob[i]]=1
        X_test=np.column_stack([X_test,O_new])
    return O      