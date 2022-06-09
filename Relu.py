# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:03:09 2018

@author: Hamid
"""
import numpy as np
def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A,cache
#%%
def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ
#%%    