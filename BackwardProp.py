# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:11:53 2018

@author: Hamid
"""
import numpy as np 
from Sigmoid import sigmoid_backward
from Relu import relu_backward
def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
#%%

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)   
    elif activation == "sigmoid":
        dZ =  sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation =="softmax":
        dZ = dA
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    return dA_prev,dW,db    

#%%    
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = AL-Y
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "softmax")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
#%%
def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3,Z4, A4, W4, b4, Z5, A5, W5, b5) = cache
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T) + lambd/m*W3
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd/m*W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T) + lambd/m*W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
#%%
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3,D3, A3, W3, b3, Z4, D4, A4, W4 ,b4, Z5, A5, W5, b5) = cache
    #dA5 = -1/m  * np.sum(-np.sum(Y/A5,axis=0))
    #dZ5=dA5 * Z5 *(1-Z5)
    dZ5 =A5 -Y
    dW5 = 1./m * np.dot(dZ5, A4.T)
    db5 = 1./m * np.sum(dZ5, axis=1, keepdims = True)
    dA4 = np.dot(W5.T, dZ5)
    dA4 = np.multiply(dA4,D4)              
    dA4 = dA4/keep_prob              
    dZ4 = relu_backward(dA4,Z4)
    dW4 = 1./m * np.dot(dZ4, A3.T)
    db4 = 1./m * np.sum(dZ4, axis=1, keepdims = True)
    dA3 = np.dot(W4.T, dZ4)
    dA3 = np.multiply(dA3,D3)             
    dA3 = dA3/keep_prob                
    dZ3 = relu_backward(dA3,Z3)
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    dA2 = np.multiply(dA2,D2)              
    dA2 = dA2/keep_prob 
    dZ2 = relu_backward(dA2,Z2)
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    dA1 = np.dot(W2.T, dZ2)
    dA1 = np.multiply(dA1,D1)             
    dA1 = dA1/keep_prob    
    dZ1 = relu_backward(dA1,Z1)
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ5": dZ5, "dW5": dW5, "db5": db5,
                 "dA4":dA4,"dZ4": dZ4, "dW4": dW4, "db4": db4,
                 "dA3":dA3,"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2":dA2,"dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1":dA1,"dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
#%%
def conv_backward(dZ,cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    (A_prev,W,b,hparameters)=cache 
    (m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
    (f,f,n_C_prev,n_C)=W.shape
    stride=hparameters["stride"]
    pad=hparameters["pad"]
    (m,n_H,n_W,n_C)=dZ.shape
    dA_prev=np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW=np.zeros((f,f,n_C_prev,n_C))
    db=np.zeros((1,1,1,n_C))
    A_prev_pad=Zero_pad(A_prev,pad)
    dA_prev_pad=Zero_pad(dA_prev,pad) 
    for i in range(m):
        a_prev_pad=A_prev_pad[i]
        da_prev_pad=dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]     
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db 
#%%
def create_mask_from_windows(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = x == np.max(x)
    return mask
#%%
def distribute_value(dZ,shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    (n_H, n_W) = shape
    average = dZ / (n_H * n_W)
    a = np.ones(shape) * average
    return a
#%%
def pool_backward(dA,cache,mode="max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    (A_prev, hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    m, n_H_prev, n_W_prev, n_C_prev =  A_prev.shape
    m, n_H, n_W, n_C =  dA.shape
    dA_prev = np.zeros(A_prev.shape)       
    for i in range(m):  
         a_prev = A_prev[i]
         for h in range(n_H):                   
            for w in range(n_W):               
                for c in range(n_C): 
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_windows(a_prev_slice)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)    
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev       
#%%    
#def backward_fuzzy(X, Y, cache, keep_prob):
    