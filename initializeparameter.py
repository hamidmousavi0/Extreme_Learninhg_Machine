# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:04:59 2018

@author: Hamid
"""
import numpy as np
import skfuzzy
#%%

def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)           
    for l in range(1, L): 
        parameters['W' + str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
      
    return parameters    
#%%
def initialize_parameters_random(layers_dims):
    np.random.seed(3)              
    parameters = {}
    L = len(layers_dims)          
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.rand(layers_dims[l],layers_dims[l-1])*10
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
    return parameters    
#%%
def initialize_parameters_X(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.uniform(np.sqrt(-6/layers_dims[l]+layers_dims[l-1]),np.sqrt(6/layers_dims[l]+layers_dims[l-1]),(layers_dims[l],layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))
        
    return parameters        
#%% 
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 
    for l in range(1, L + 1):
       
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))
        
    return parameters  
        
#%%
def initialize_parameters_deep(layer_dims):
   
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)          
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters   
#%%
def initialize_parameters_conv(f,n_C_prev, n_C):             
    parameters = {}
    parameters['W'] = np.random.rand(f,f,n_C_prev,n_C)
    parameters['b'] = np.random.rand(1,1,1,n_C)
    return parameters  
#%%
#def initialize_parameters_fuzzy(X,num_feature,num_mem):
#    cntr,u,u0,d,jm,p,fpc=skfuzzy.cluster.cmeans(X,10,3,0.0001,100)
#    return cntr,u,u0,d,jm,p,fpc
#     