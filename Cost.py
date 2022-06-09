# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:08:03 2018

@author: Hamid
"""
import numpy as np
def compute_cost_crossentropy(AL, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost    
#%%
def compute_cost_meansqure(AL,Y):
    m = Y.shape[1] 
    cost = (1./m) * np.sum(((AL-Y)**2))
    cost = np.squeeze(cost)    
    assert(cost.shape == ())
    
    return cost 
#%%
def compute_cost_softmax(AL, Y):
    m = Y.shape[1]
    temp=- np.sum((np.multiply(Y,np.log(AL.T))),axis=0)
    cost = (1./m) *np.sum ( temp)
    cost = np.squeeze(cost)     
    assert(cost.shape == ())
    return cost
#%%
def compute_cost_with_regularization(A5, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    W4 = parameters["W4"]
    W5 = parameters["W5"]
    cross_entropy_cost = compute_cost_softmax(A5, Y) 
    L2_regularization_cost =lambd/(2*m) *np.sum([np.sum(np.square(parameters[v])) for v in ["W1","W2","W3","W4","W5"]])
    cost = cross_entropy_cost + L2_regularization_cost
    return cost    