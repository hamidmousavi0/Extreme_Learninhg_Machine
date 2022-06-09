# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:17:33 2018

@author: Hamid
"""
import numpy as np
from ForwardPropagation import forward_propagation_with_dropout,L_model_forward
def predict_new(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 
    p = np.zeros((1,m))
    y_index=np.zeros((1,m))
    y_index=(np.argmax(y,axis=0)).reshape((1,m))
    probas, caches = L_model_forward(X, parameters)
    for i in range(0, probas.shape[1]):
            max_index=np.where(probas[:,i]==probas[:,i].max())
            p[0,i] = max_index[0][0]
    print("Accuracy: "  + str(np.sum((p == y_index)/m)))
        
    return np.sum((p == y_index)/m)
#%%
def pred_test(X,parameters):
    m = X.shape[1]
    probas, caches = L_model_forward(X, parameters)
    p = np.zeros((1,m))
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    return p

        