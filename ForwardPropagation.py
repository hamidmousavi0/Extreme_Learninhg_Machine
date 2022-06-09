# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:05:41 2018

@author: Hamid
"""
import numpy as np
from ZeroPadding import Zero_pad
from Sigmoid import sigmoid
from Relu import relu
from Softmax import softmax
def linear_forward(A, W, b):
   
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
#%%
def linear_activation_forward(A_prev, W, b, activation):  
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
     
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    elif activation=="softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z) 
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
#%%
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2               
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    AL, cache =  linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "softmax")
    caches.append(cache)
            
    return AL, caches
#%%
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    np.random.seed(1)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]
    W5 = parameters["W5"]
    b5 = parameters["b5"]
    Z1 = np.dot(W1, X) + b1
    A1,cash = relu(Z1)
    D1 = np.random.rand(A1.shape[0],A1.shape[1])                                         
    D1 = (D1<keep_prob)                                         
    A1 = np.multiply(A1,D1)                                      
    A1 = A1/keep_prob                                     
    Z2 = np.dot(W2, A1) + b2
    A2,cash1 = relu(Z2)
    D2 = np.random.rand(A2.shape[0],A2.shape[1])  
    D2 = (D2<keep_prob)                                        
    A2 = np.multiply(A2,D2)                                        
    A2 = A2/keep_prob                                          
    Z3 = np.dot(W3, A2) + b3
    A3,cash2 = relu(Z3)
    D3 = np.random.rand(A3.shape[0],A3.shape[1])  
    D3 = (D3<keep_prob)                                        
    A3 = np.multiply(A3,D3)                                         
    Z4 = np.dot(W4, A3) + b4
    A4,cash3 = relu(Z4)
    D4 = np.random.rand(A4.shape[0],A4.shape[1])  
    D4 = (D4<keep_prob)                                         
    A4 = np.multiply(A4,D4)                                           
    A4 = A4/keep_prob
    Z5 = np.dot(W5, A4) + b5
    A5,chash4 = softmax(Z5)
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3,D3, A3, W3, b3, Z4, D4, A4, W4 ,b4, Z5, A5, W5, b5)
    
    return A5, cache    
#%%
def conv_single_step(a_slice_prev,W,b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    s=np.multiply(a_slice_prev,W)
    Z=np.sum(s)
    Z=Z+float(b)
    return Z
#%%
def conv_forward(A_prev,W,b,hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    (f,f,n_C_prev,n_C)=W.shape
    stride=hparameters['stride']
    pad=hparameters['pad']
    n_H=int((n_H_prev +2*pad-f)/stride)+1
    n_W=int((n_W_prev +2*pad-f)/stride)+1
    Z=np.zeros((m,n_H,n_W,n_C))
    A_prev_pad=Zero_pad(A_prev,pad)
    for i in range(m):
        a_prev_pad=A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start=h*stride
                    vert_end=vert_start+f
                    horiz_start=w*stride
                    horiz_end = horiz_start+f
                    a_slice_prev=a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    Z[i,h,w,c]=conv_single_step(a_slice_prev,W[...,c],b[...,c])
    assert(Z.shape == (m, n_H, n_W, n_C)) 
    cache=(A_prev,W,b,hparameters)
    return Z,cache    
#%%
def pool_forward(A_prev,hparameters,mode="max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    f=hparameters['f']
    stride=hparameters['stride']
    n_H=int((n_H_prev-f)/stride)+1
    n_W=int((n_W_prev-f)/stride)+1
    n_C=n_C_prev 
    A=np.zeros((m,n_H,n_W,n_C))
    for i in  range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start=h*stride
                    vert_end=vert_start+f
                    horiz_start=w*stride
                    horiz_end=horiz_start+f
                    a_prev_slice=A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    if mode=="max":
                        A[i,h,w,c]=np.max(a_prev_slice)
                    elif mode=="average":
                        A[i,h,w,c]=np.mean(a_prev_slice)    
    cache=(A_prev,hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    return A,cache
#%%
#def fuzzy_forward_propagate(X_train,parameters):
    