# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:49:30 2018

@author: Hamid
"""

from basic_ELM_AE_pcg import ELM_AE
from basic_elm_pcg import sigmoid,softmax
from main_PCG import main
import numpy as np
def ML_ELM_train (X_train,Y_train,hid_num,k,C):
    betahat_1= ELM_AE(X_train,hid_num[0],C[0]) 
    H_temp = np.dot(X_train,betahat_1.T)
    H_1 = sigmoid(H_temp)
    betahat_2 =ELM_AE(H_1,hid_num[1],C[1])
    H_2 = sigmoid(np.dot(H_1,betahat_2.T))
    betahat_3 =ELM_AE(H_2,hid_num[2],C[2])
    H_3 = sigmoid(np.dot(H_2,betahat_3.T))
    betahat_4=main(H_3,Y_train)
    Y=np.dot(H_3,betahat_4)
    Y=(Y)
    return betahat_1,betahat_2,betahat_3,betahat_4,Y
#%%
def ML_ELM_test(X,Y,betahat_1,betahat_2,betahat_3,betahat_4,k):
    H_temp = np.dot(X,betahat_1.T)
    H_1 = sigmoid(H_temp)
    H_2 = sigmoid(np.dot(H_1,betahat_2.T))
    H_3 = sigmoid(np.dot(H_2,betahat_3.T))
    Y_predict=np.dot(H_3,betahat_4)
    Y_predict=(Y_predict)
    return Y_predict  
