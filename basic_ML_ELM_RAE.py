# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:04:37 2018

@author: Hamid
"""
import numpy as np
def ML_RELMAE(X_train,Y_train,X_test,C,D,max_iter):
    Betahat_1=ELM_AE_RE(X_train,D[0],C[0],max_iter)  
    temp_H1=np.dot(X_train,Betahat_1)
    H1=sigmoid(temp_H1)
    Betahat_2=ELM_AE_RE(H1,D[1],C[1],max_iter) 
    temp_H2=np.dot(H1,Betahat_2)
    H2=sigmoid(temp_H2)
    Betahat_3=ELM_AE_RE(H2,D[2],C[2],max_iter)  
    temp_H3=np.dot(H2,Betahat_3)
    H3=sigmoid(temp_H3)
    H3_n=np.dot(H3.T,H3)
    one_matrix=np.identity(H3_n.shape[0])
    one_matrix=one_matrix* 1/C[2]
    new_H3=H3_n + one_matrix
    inverse_H3=np.linalg.inv(new_H3)
    Beta_hat=np.dot(np.dot(inverse_H3,H3.T),Y_train)
    H1_test=np.dot(X_test,Betahat_1)
    H1_test=sigmoid(H1_test)
    H2_test=np.dot(H1_test,Betahat_2)
    H2_test=sigmoid(H2_test)
    H3_test=np.dot(H2_test,Betahat_3)
    H3_test=sigmoid(H3_test)
    Y = (np.dot(H3_test,Beta_hat))
    return Y
#%%
def ELM_AE(X_train,num_hid,C,W):
    temp_H=(np.dot(X_train,W))
    H=sigmoid(temp_H)
    H_n=np.dot(H.T,H)
    one_matrix=np.identity(H_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_H=H_n + one_matrix
    inverse_H=np.linalg.inv(new_H)
    Beta_hat=np.dot(np.dot(inverse_H,H.T),X_train)
    return Beta_hat  
#%%    
def ELM_AE_RE(X_train,hid_num,C,num_iter):
    parameter=initialize_parameters_random(X_train.shape[1],hid_num)
    W=parameter['W']
    Ws = np.zeros((X_train.shape[1],hid_num))
    Ws=np.add(Ws,W)
    Beta_hat=ELM_AE(X_train,hid_num,C,W)
    for i in range(num_iter):
        W=Beta_hat.T
        Ws=np.add(Ws,W)
        Beta_hat=ELM_AE(X_train,hid_num,C,Ws)
    Wprime = (1/(num_iter))* Ws  
    return Wprime
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
def sudo_inverse (X,C):
    X_n = np.dot(X.T,X)
    one_matrix=np.identity(X_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_X=X_n + one_matrix
    inverse_X=np.linalg.inv(new_X)
    invX = np.dot(inverse_X,X.T)
    return invX
#%%
def initialize_parameters_random(num_X,num_hid):             
    parameters = {}
    parameters['W'] = np.random.randn(num_X,num_hid)
    parameters['b'] = np.random.randn(1,num_hid)
    return parameters    