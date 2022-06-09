# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:30:16 2018

@author: Hamid
"""
from scipy.special import logit
import numpy as np
def new_ELMAE(X_train,Y_train,X_test,C,D,max_iter):
    Beta1,W1=ELM_AE_RE(X_train,D,C,max_iter) 
    H1,Wie1,Whe1 =T_ELM_Train(X_train,Y_train,D,C,W1)
    Beta2,W2=ELM_AE_RE(H1,D,C,max_iter) 
    H2,Wie2,Whe2=T_ELM_Train(H1,Y_train,D,C,W2)
    Beta_new = np.dot(sudo_inverse(H2,C),Y_train)
    Y_predict=T_ELM_Test(X_test,Wie1,Wie2,Whe1,Whe2,Beta_new)
    return Y_predict

#%%    
def ELM_train(X_train,Y_train,num_hid,C):
    
#    c=100000
    parameter=initialize_parameters_random(X_train.shape[1],num_hid)
    W=parameter['W']
    temp_H=np.dot(X_train,W)
    H=sigmoid(temp_H)
    H_n=np.dot(H.T,H)
    one_matrix=np.identity(H_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_H=H_n + one_matrix
    inverse_H=np.linalg.inv(new_H)
    Beta_hat=np.dot(np.dot(inverse_H,H.T),Y_train)
    return W,Beta_hat     
#%%    
def ELM_AE(X_train,hid_num,C):
    W,Beta_hat=ELM_train(X_train,X_train,hid_num,C)
    return W,Beta_hat   
#%%
def ELM_AE_RE(X_train,hid_num,C,num_iter):
    W,Beta_hat=ELM_AE(X_train,hid_num,C)
   
    for i in range(num_iter):
        W=Beta_hat.T
        temp_H=np.dot(X_train,W)
        H=sigmoid(temp_H)
        H_n=np.dot(H.T,H)
        one_matrix=np.identity(H_n.shape[0])
        one_matrix=one_matrix* 1/C
        new_H=H_n + one_matrix
        inverse_H=np.linalg.inv(new_H)
        Beta_hat=np.dot(np.dot(inverse_H,H.T),X_train)
    return Beta_hat,W
#%%   
def denormalize(Y,H1_max,H1_min,b,a):
    t1 = b-a
    t2 = H1_max - H1_min
    t3 = Y * (t2/t1)
    t4 = t3+H1_min
    return t4           
        
    
        
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
#%%
def T_ELM_Train(X_train,Y_train,num_hid,C,W): 
    param = initialize_parameters_random(X_train.shape[1],num_hid)
    ones = np.ones((X_train.shape[0],1))
    Xe= np.column_stack([X_train,ones])
#    W = param [ 'W' ]
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
    return H2,Wie,Whe
#%%
def T_ELM_Test(X_test,Wie1,Wie2,Whe1,Whe2,Beta_new):
    ones = np.ones((X_test.shape[0],1))
    Xte= np.column_stack([X_test,ones])
    H1_temp = (np.dot(Xte,Wie1))
    H1_temp=sigmoid(H1_temp)
    ones = np.ones((H1_temp.shape[0],1))
    H1_temp=np.column_stack([H1_temp,ones])
    H1=sigmoid(np.dot(H1_temp,Whe1))
    ones = np.ones((H1.shape[0],1))
    H2_temps=np.column_stack([H1,ones])
    H2_temps=(np.dot(H2_temps,Wie2))
    H2_temps=sigmoid(H2_temps)
    ones = np.ones((H2_temps.shape[0],1))
    H2_temps=np.column_stack([H2_temps,ones])
    H2=sigmoid(np.dot(H2_temps,Whe2))
    Y_predict=softmax(np.dot(H2,Beta_new)) 
    return Y_predict          