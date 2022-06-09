# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:30:04 2018

@author: admin
"""
from basic_elm import ELM_train
def ELM_AE(X_train,hid_num,C):
    W,Beta_hat,Y=ELM_train(X_train,X_train,hid_num,C)
    return Beta_hat