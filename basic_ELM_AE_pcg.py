# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:04:01 2018

@author: Hamid
"""

from basic_elm_pcg import ELM_train
def ELM_AE(X_train,hid_num,C):
    W,Beta_hat,Y=ELM_train(X_train,X_train,hid_num,C)
    return Beta_hat