# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:01:59 2018

@author: Hamid
"""
import numpy as np
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A,cache
#%%
def sigmoid_backward(dA, cache):
   
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ  
#%%    