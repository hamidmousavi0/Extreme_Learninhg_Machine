# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:04:35 2018

@author: Hamid
"""
import numpy as np
def convert_to_one_hot(Y, C):  
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y