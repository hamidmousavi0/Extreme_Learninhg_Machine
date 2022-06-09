# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:44:47 2018

@author: Hamid
"""
from sklearn.metrics import mean_squared_error
from scipy.special import logit
import numpy as np 

#%%
def AutoEncoder(X_train,g,max_loop,D,C):
    j=1 
    param = initialize_parameters_random(X_train.shape[1],D)
    af = param [ 'W' ]
    bf = param [ 'b']
#    if W.shape[0] >W.shape[1]:
#          v1,s1,u1 =np.linalg.svd(W)
#          v2,s1,u2 =np.linalg.svd(b)
#          af=v1[:,0:W.shape[1]]
#          bf = (u2[0,0:b.shape[1]]).reshape((1,b.shape[1]))
#    else :
#          W = W.T
#          v1,s1,u1 =np.linalg.svd(W)
#          v2,s1,u2 =np.linalg.svd(b)
#          temp = v1[:,0:W.shape[1]]
#          af = temp.T 
#          bf = (u2[0,0:b.shape[1]]).reshape((1,b.shape[1]))
   
    while j < max_loop :
          HF = forward_pro(X_train,af,bf,g)
          HF_inverse = sudo_inverse(HF,C)
          
          if g =="sigmoid":
              X_train_normal=normalize(X_train,0.1,0.9)
              aa=logit(X_train_normal)
              an = np.dot(HF_inverse,logit(X_train_normal))
              bn = np.sqrt(mean_squared_error(np.dot(HF,an),logit(X_train_normal)))
          if g == "sin":
              an = np.dot(HF_inverse,np.arcsin(X_train))
              bn = np.sqrt(mean_squared_error(np.dot(HF,an),np.arcsin(X_train)))
          j= j + 1
          af = an.T
          bf = bn 
    HF = forward_pro(X_train,af,bf,g)          
    return HF,af,bf,an,bn
#%%
def multi_network_train (X_train , g,num_layer , max_loop , D ,C):
    j=1
    af_list=[]
    bf_list=[]
    an_list=[]
    bn_list=[]
    while j<num_layer :
        HF,af,bf,an,bn = AutoEncoder(X_train,g,max_loop,D,C) 
        af_list.append(af)
        bf_list.append(bf)
        an_list.append(an)
        bn_list.append(bn)
        j=j+1
        X_train = HF
    HC_train = X_train
    for i in range (1,len(an_list)+1):
        reconst_x =forward_pro(HF,an_list[len(an_list)-i],bn_list[len(an_list)-i],"sin")
        HF=reconst_x
    return HC_train,reconst_x,af_list,bf_list,an_list,bn_list
#%%
def multi_network_test (X_test , g,af_list,bf_list,an_list,bn_list ):
    j=0
    while j<len(af_list) :
        h=forward_pro(X_test,af_list[j],bf_list[j],g)
        X_test=h
        j=j+1
    HC_test=X_test  
    for i in range (1,len(an_list)+1):
        reconst_x =forward_pro(h,an_list[len(an_list)-i],bn_list[len(an_list)-i],"sigmoid")
        h=reconst_x
    return HC_test,reconst_x
#%%
def normalize(X,a,b):
    t1 = b-a
    Xmin= np.min(X)
    Xmax = np .max(X)
    t2 = X-Xmin
    t3 = Xmax - Xmin
    t4 = t2 / t3
    t5 = t1 * t4
    return t5+a       
#%%    
def initialize_parameters_random(num_X,num_hid):             
    parameters = {}
    parameters['W'] = np.random.randn(num_X,num_hid)
    parameters['b'] = np.random.randn(1,num_hid)
    return parameters 
#%% 
def sudo_inverse (X,C):
    X_n = np.dot(X.T,X)
    one_matrix=np.identity(X_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_X=X_n + one_matrix
    inverse_X=np.linalg.inv(new_X)
    invX = np.dot(inverse_X,X.T)
    np.savetxt("XX.txt",invX)
    return invX
#%%
def forward_pro(X,W,b,g):
    temp_H=(np.dot(X,W)+b)
    if g== "sigmoid":
       h=sigmoid(temp_H) 
    if g == "sin":
       h= np.sin(temp_H)
    return h  
#%%
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A
#%%
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0,keepdims = True)
    return A 
#%%
def normalize(X,a,b):
    t1 = b-a
    Xmin= np.min(X)
    Xmax = np .max(X)
    t2 = X-Xmin
    t3 = Xmax - Xmin
    t4 = t2 / t3
    t5 = t1 * t4
    return t5+a   