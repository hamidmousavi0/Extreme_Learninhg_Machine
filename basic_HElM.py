# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:28:25 2018

@author: Hamid
"""
from scipy.stats import zscore
import numpy as np 
def H_ELM_train(train_x,train_y,b1,b2,b3,C):
    train_x = zscore(train_x.T).T
    one_vec = 0.1* np.ones((train_x.shape[0],1))
    H1=np.column_stack([train_x,one_vec])
    A1=np.dot(H1,b1.T)
    A1 = A1 / np.max(A1)
    A1=sigmoid(A1)
    beta1  =  sparse_elm_ae(A1,H1,1e-3,50).T
    T1 = np.dot(H1 ,beta1)
    T1 = T1 / np.max(T1)
    T1=sigmoid(T1)
    one_vec = 0.1* np.ones((T1.shape[0],1))
    H2=np.column_stack([T1,one_vec])
    A2=np.dot(H2,b2.T)
    A2 = A2/ np.max(A2)
    A2=sigmoid(A2)
    beta2  =  sparse_elm_ae(A2,H2,1e-3,50).T
    T2 = np.dot(H2,beta2)
    T2 = T2 / np.max(T2)
    T2=sigmoid(T2)
    one_vec = 0.1* np.ones((T2.shape[0],1))
    H3=np.column_stack([T2,one_vec])
    A3=np.dot(H3,b3.T)
    A3 = A3 / np.max(A3)
    A3=sigmoid(A3)
    A3_n=np.dot(A3.T,A3)
    one_matrix=np.identity(A3_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_A3=A3_n + one_matrix
    inverse_A3=np.linalg.inv(new_A3)
    beta3=np.dot(np.dot(inverse_A3,A3.T),train_y)
    return beta1,beta2,beta3
#%%
def H_ELM_test(test_x,beta1,beta2,beta3):
    test_x = zscore(test_x.T).T
    one_vec = 0.1* np.ones((test_x.shape[0],1))
    H1=np.column_stack([test_x,one_vec])
    A1=np.dot(H1,beta1)
    A1 = A1 / np.max(A1)
    A1=sigmoid(A1)
    one_vec = 0.1* np.ones((A1.shape[0],1))
    H2=np.column_stack([A1,one_vec])
    A2=np.dot(H2,beta2)
    A2 = A2/ np.max(A2)
    A2=sigmoid(A2)
    one_vec = 0.1* np.ones((A2.shape[0],1))
    H3=np.column_stack([A2,one_vec])
    A3=np.dot(H3,beta3)
    A3 = A3 / np.max(A3)
    Y_predict=softmax(A3)
    return Y_predict

#%%
def sparse_elm_ae(A,b,lam,itrs):
    AA = np.dot(A.T,A)
    w,v=np.linalg.eig(AA)
    LF=np.max(w)
    Li = 1/LF
    alp = lam * Li
    m = A.shape[1]
    n = b.shape[1]
    x = np.zeros((m,n))
    yk = x
    tk = 1
    L1 = 2*Li*np.dot(A.T,b)
    L2 = 2*Li*np.dot(A.T,b)
    for i in range(itrs):
        ck = ((yk - L1)*(yk + L2))
        x1 = (np.max(np.abs(ck)-alp,0))*np.sign(ck)
        tk1 = 0.5 + 0.5*np.sqrt(1+4*tk**2)
        tt = (tk-1)/tk1
        yk = x1 + tt*(x-x1)
        tk = tk1
        x = x1
    return x
#%%
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A
#%%
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=1,keepdims = True)
    return A     
