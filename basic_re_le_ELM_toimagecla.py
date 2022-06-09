# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 06:15:01 2018

@author: Hamid
"""

import numpy as np
#%%
def ELM_full(X_train,num_hid,C):
    Wg=[]
    Beta_hat=ELM_AE_random(X_train,num_hid,C)
    H1=forward_pro(X_train,Beta_hat.T)
    Wg.append(Beta_hat.T)
    for i in range (num_hid-2):
        Beta_hat=ELM_AE_random(H1,num_hid,C)
        H1=forward_pro(H1,Beta_hat.T)
        Wg.append(Beta_hat.T)
    H_n=np.dot(H1.T,H1)
    one_matrix=np.identity(H_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_H=H_n + one_matrix
    #moore_inv=np.linalg.pinv(H)
    inverse_H=np.linalg.inv(new_H)
    Beta_hat=np.dot(np.dot(inverse_H,H1.T),X_train)    
    Wg.append(Beta_hat)
    return Wg   
#%%          
def ELM_class(X_train_c,Wg,C):
    num_hid=len(Wg)
    Wj = []
    Beta_hat=ELM_AE_WG(X_train_c,num_hid,C,Wg,0) 
    H1=forward_pro(X_train_c,Beta_hat.T)
    Wj.append(Beta_hat.T)
    for i in range(1,num_hid-1):
        Beta_hat=ELM_AE_WG(H1,num_hid,C,Wg,i) 
        H1=forward_pro(H1,Beta_hat.T)
        Wj.append(Beta_hat.T)
    H_n=np.dot(H1.T,H1)
    one_matrix=np.identity(H_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_H=H_n + one_matrix
    #moore_inv=np.linalg.pinv(H)
    inverse_H=np.linalg.inv(new_H)
    Beta_hat=np.dot(np.dot(inverse_H,H1.T),X_train_c)    
    Wj.append(Beta_hat)       
    return Wj
#%%
def ELM_RE_image_train(X_train,X_train_clases,num_hid,C):
    Wg = ELM_full(X_train,num_hid,C)
    Wj=[]
    for j in range (len(X_train_clases)):
        Wjj=ELM_class(X_train_clases[j],Wg,C)
        Wj.append(Wjj)
    return Wj    
#%%
def ELM_RE_image_test(X_test,Wj,C):
    rec_error=[]
    x_Rec=[]
    num_hidd=len(Wj[0])
    for im in range(X_test.shape[0]):
        for i in range(len(Wj)):
            h=forward_pro(X_test[im,:].reshape((1,X_test.shape[1])),Wj[i][0])
            for j in range(1,num_hidd):
                h=forward_pro(h,Wj[i][j])
            x_Rec.append(h)
            err=np.linalg.norm(X_test[im,:]-x_Rec[i])
            rec_error.append(err)
            
    return rec_error,x_Rec
#%%
def ELM_train_random(X_train,Y_train,num_hid,C):
        parameter=initialize_parameters_random(X_train.shape[1],num_hid)
        W=parameter['W']
        temp_H=(np.dot(X_train,W))
        temp_H=temp_H/np.max(temp_H)
        H=sigmoid(temp_H)
        H_n=np.dot(H.T,H)
        one_matrix=np.identity(H_n.shape[0])
        one_matrix=one_matrix* 1/C
        new_H=H_n + one_matrix
        #moore_inv=np.linalg.pinv(H)
        inverse_H=np.linalg.inv(new_H)
        Beta_hat=np.dot(np.dot(inverse_H,H.T),Y_train)
        return Beta_hat
#%%
def ELM_train_WG(X_train,Y_train,num_hid,C,WG,i):
        temp_H=(np.dot(X_train,WG[i]))
        temp_H=temp_H/np.max(temp_H)
        H=sigmoid(temp_H)
        H_n=np.dot(H.T,H)
        one_matrix=np.identity(H_n.shape[0])
        one_matrix=one_matrix* 1/C
        new_H=H_n + one_matrix
        #moore_inv=np.linalg.pinv(H)
        inverse_H=np.linalg.inv(new_H)
        Beta_hat=np.dot(np.dot(inverse_H,H.T),Y_train)
        return Beta_hat      
#%%
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A
#%%
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=1,keepdims = True)
    return A 
#%%
def forward_pro(X,W):
    temp_H=np.dot(X,W)
    temp_H=temp_H/np.max(temp_H)
    h=sigmoid(temp_H) 
    return h 
#%%
def initialize_parameters_random(num_X,num_hid):             
    parameters = {}
#    parameters['W'] = np.random.uniform(np.sqrt(-6/num_hid+num_X),np.sqrt(6/num_hid+num_X),(num_X,num_hid))
    parameters['W'] = np.random.rand(num_X,num_hid)
    parameters['b'] = np.random.rand(1,num_hid)
    return parameters  
#%%
def ELM_AE_random(X_train,hid_num,C):
    Beta_hat=ELM_train_random(X_train,X_train,hid_num,C)
    return Beta_hat 
#%%
def ELM_AE_WG(X_train,hid_num,C,WG,i):
    Beta_hat=ELM_train_WG(X_train,X_train,hid_num,C,WG,i)
    return Beta_hat     