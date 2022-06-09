# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:25:52 2018

@author: Hamid
"""

import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
def KELM_AE(X_train,Ci,sigmai,gi):
    gamma = 1/(2 * (sigmai**2))
    omega_i = rbf_kernel(X_train,X_train,gamma)
    one_matrix=np.identity(omega_i.shape[0])
    one_matrix=one_matrix* 1/Ci
    new_omega = one_matrix+omega_i
    inverse_omega=np.linalg.inv(new_omega)
    gamma_tild_i=np.dot(inverse_omega,X_train)
    if gi == "sigmoid":
        X_train=X_train/np.max(X_train)
        X_train_i_1=sigmoid(np.dot(X_train,gamma_tild_i.T))
    return X_train_i_1,gamma_tild_i
#%%
def ML_KELM_train(X_train,Y_train,Ci,sigmai,Nlayer,gi):
    X_2 , gamma_tild_1 =KELM_AE(X_train,Ci,sigmai,"sigmoid")     
    X_3 , gamma_tild_2 =KELM_AE(X_2,Ci,sigmai,"sigmoid") 
    gamma_tild_unified = gamma_tild_2
    for i in range(3,Nlayer-1):
        X_i_1,gamma_tild_i=KELM_AE(X_3,Ci,sigmai,"sigmoid")
        X_3=X_i_1
        gamma_tild_unified=gamma_tild_i*gamma_tild_unified
    X_final = X_i_1
    gamma = 1/(2 * (sigmai**2))
    omega_i = rbf_kernel(X_final,X_final,gamma)
    one_matrix=np.identity(omega_i.shape[0])
    one_matrix=one_matrix* 1/Ci
    new_omega = one_matrix+omega_i
    inverse_omega=np.linalg.inv(new_omega)
    Beta_hat=np.dot(inverse_omega,Y_train)   
    return X_final,gamma_tild_1,gamma_tild_unified,Beta_hat
#%%
def ML_KELM_test(X_test,gamma_tild_1,gamma_tild_unified,Beta_hat,sigmai):
    X_final = sigmoid(np.dot((np.dot(X_test,gamma_tild_1.T)),gamma_tild_unified.T))
    gamma = 1/(2 * (sigmai**2))
    omega_i = rbf_kernel(X_final,X_final,gamma)
    Y_predict= np.dot(omega_i,Beta_hat)
    return Y_predict
#%%
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A   
    
    