# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:03:56 2018

@author: Hamid
"""
import numpy as np
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0,keepdims = True)
    cache = Z
    
    return A,cache
#%%
    