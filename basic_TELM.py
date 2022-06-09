# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:45:25 2018

@author: Hamid
"""
from scipy.special import logit
import numpy as np 
def initialize_parameters_random(num_X,num_hid):             
    parameters = {}
    parameters['W'] = np.random.randn(num_X,num_hid)
    parameters['b'] = np.random.randn(1,num_hid)
    return parameters 
#%%
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A
#%%
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=1,keepdims = True)
    return A 
#%%
def T_ELM_Train(X_train,Y_train,num_hid,C): 
    param = initialize_parameters_random(X_train.shape[1],num_hid)
    ones = np.ones((X_train.shape[0],1))
    Xe= np.column_stack([X_train,ones])
    W = param [ 'W' ]
    b = param [ 'b']
    Wie = np.row_stack([W,b])
    H = (np.dot(Xe,Wie))
    H=sigmoid(H)
    H_sudoinverse = sudo_inverse(H,C)
    Beta=np.dot(H_sudoinverse,Y_train)
    H1=np.dot(Y_train,sudo_inverse(Beta,C))
    ones = np.ones((H.shape[0],1))
    He=np.column_stack([H,ones])
    H1_max = np.max(H1)
    H1_min = np.min(H1)
    H1_normal=normalize(H1,0.1,0.9)
    He_inverse = sudo_inverse(He,C)
    inverse_activation = logit(H1_normal)
    inverse_activation=denormalize(inverse_activation,H1_max,H1_min,0.1,0.9)
    Whe=np.dot(He_inverse,inverse_activation)
    H2=(np.dot(He,Whe))
    H2=sigmoid(H2)
    #H2_denormal = 
    Beta_new = np.dot(sudo_inverse(H2,C),Y_train)
    return Wie,Whe,Beta_new 
#%%
def T_ELM_Test(X_test,Wie,Whe,Beta_new):
    ones = np.ones((X_test.shape[0],1))
    Xte= np.column_stack([X_test,ones])
    H1 = (np.dot(Xte,Wie))
    H1=sigmoid(H1)
    ones = np.ones((H1.shape[0],1))
    He=np.column_stack([H1,ones])
    H2=(np.dot(He,Whe))
    H2=sigmoid(H2)
    Y_predict=softmax(np.dot(H2,Beta_new)) 
    return Y_predict       
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
def denormalize(Y,H1_max,H1_min,b,a):
    t1 = b-a
    t2 = H1_max - H1_min
    t3 = Y * (t2/t1)
    t4 = t3+H1_min
    return t4           