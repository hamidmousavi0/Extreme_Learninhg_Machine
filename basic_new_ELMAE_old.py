# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:30:16 2018

@author: Hamid
"""
from scipy.special import logit
import numpy as np
def new_ELMAE(X_train,X_test,C,D,max_iter):
    Beta,W=ELM_AE_RE(X_train,D,C,max_iter)  
    HF=np.dot(X_train,W)
#    temp_HF=HF/np.max(HF)
    H1=sigmoid(HF)
    H1_n=np.dot(H1.T,H1)
    one_matrix=np.identity(H1_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_H1=H1_n + one_matrix
    inverse_H1=np.linalg.inv(new_H1)
    Beta_hat=np.dot(np.dot(inverse_H1,H1.T),X_train)
#    beta_inverse = np.linalg.pinv(Beta_hat)
    beta_inverse = sudo_inverse(Beta_hat,C)
    H2_back=np.dot(X_train,beta_inverse)
    max_h2 = np.max(H2_back)
    min_h2 = np.min(H2_back)
    H2_normal = normalize(H2_back,0.1,0.9)
    H2_inverse = sudo_inverse(H2_back,C)
    inverse_activation = logit(H2_normal)
    H2_normal_back = normalize(inverse_activation,min_h2,max_h2)
    weightsH1H2=np.dot(H2_inverse,H2_normal_back)
    H1=np.dot(X_train,W)
#    H1=H1/np.max(H1)
    H1=sigmoid(H1)
    H2=np.dot(H1,weightsH1H2)
#    H2=H2/np.max(H2)
    H2=sigmoid(H2)
    H_n=np.dot(H2.T,H2)
    one_matrix=np.identity(H_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_H=H_n + one_matrix
    inverse_H=np.linalg.inv(new_H)
    Beta_hat=np.dot(np.dot(inverse_H,H2.T),X_train)
#    beta_inverse = np.linalg.pinv(Beta_hat)
    beta_inverse = sudo_inverse(Beta_hat,C)
    H3_back=np.dot(X_train,beta_inverse)
    max_h3 = np.max(H3_back)
    min_h3 = np.min(H3_back)
    H3_normal = normalize(H3_back,0.1,0.9)
    H3_inverse = sudo_inverse(H3_back,C)
    inverse_activation = logit(H3_normal)
    H3_normal_back = normalize(inverse_activation,min_h3,max_h3)
    weightsH2H3=np.dot(H3_inverse,H3_normal_back)
    H3=np.dot(H2,weightsH2H3)
#    H3=H3/np.max(H3)
    H3=sigmoid(H3)
    reconst_x = np.dot(H3,Beta)
    reconst_x= normalize(reconst_x,0,1)
    reconst_x=(reconst_x)
    H1_test=np.dot(X_test,W)
#    H1_test=H1_test/np.max(H1_test)
    H1_test=sigmoid(H1_test)
    H2_test=np.dot(H1_test,weightsH1H2)
#    H2_test=H2_test/np.max(H2_test)
    H2_test=sigmoid(H2_test)
    H3_test=np.dot(H2_test,weightsH2H3)
#    H3_test=H3_test/np.max(H3_test)
    H3_test=sigmoid(H3_test)
    return reconst_x,H3,H3_test
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#%%    
def ELM_train(X_train,Y_train,num_hid,C):
    
#    c=100000
    parameter=initialize_parameters_random(X_train.shape[1],num_hid)
    W=parameter['W']
    temp_H=np.dot(X_train,W)
    temp_H=temp_H/np.max(temp_H)
    H=sigmoid(temp_H)
    H_n=np.dot(H.T,H)
    one_matrix=np.identity(H_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_H=H_n + one_matrix
    #moore_inv=np.linalg.pinv(H)
    inverse_H=np.linalg.inv(new_H)
    Beta_hat=np.dot(np.dot(inverse_H,H.T),Y_train)
    Y=np.dot(H,Beta_hat)
    Y=softmax(Y)
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
        temp_H=temp_H/np.max(temp_H)
        H=sigmoid(temp_H)
        H_n=np.dot(H.T,H)
        one_matrix=np.identity(H_n.shape[0])
        one_matrix=one_matrix* 1/C
        new_H=H_n + one_matrix
        inverse_H=np.linalg.inv(new_H)
        Beta_hat=np.dot(np.dot(inverse_H,H.T),X_train)
    return Beta_hat,W
#%%   
        
        
    
        
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
    return parameters    