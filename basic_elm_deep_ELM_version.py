# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:17:14 2018

@author: Hamid
"""
from RandomMinibatch import random_mini_batches
from Cost import compute_cost_softmax
from Sigmoid import sigmoid
from Relu import relu,relu_backward
from Softmax import softmax
import numpy as np 
import matplotlib.pyplot as plt
def fusion_ELM_deep(X,Y ,learning_rate =  0.001, num_iterations = 10, print_cost = True,beta1=0.99,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 64):
    grads = {}
    costs = []  
    costs_temp =[]
    t=0
    seed=10
    m = X.shape[1]  
    n_x = X.shape[0]
    n_h1=256
    n_h2=256
    n_y=10
    layers_dims=(n_x, n_h1,n_h2,n_y)
    parameters= initialize_parameters_he(layers_dims)
    v,s=initialize_adam(parameters)
    for i in range(0, num_iterations):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            A2,A3,cache=forward_propagation(minibatch_X,parameters)
            cost = compute_cost_softmax(A3,minibatch_Y.T)
            grads= backward_propagation(minibatch_X, minibatch_Y, cache)
            t=t+1
            parameters,v,s= update_parameters_with_adam(parameters, grads, v, s, t, learning_rate ,
                                  beta1 , beta2 ,  epsilon )
            costs_temp.append(cost)
        costs_temp_new = np.array(costs_temp)
        costs_temp_new = np.mean(costs_temp_new)
        if print_cost :
           print("Cost after iteration {}: {}".format(i, np.squeeze(costs_temp_new)))
        if print_cost :
           costs.append(costs_temp_new)
    A2,A3,cache=forward_propagation(X,parameters)       
    plt.figure()       
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return A2,parameters
    
#%%
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]  
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    Z1 = np.dot(W1, X) + b1
    A1,cash1 = relu(Z1)                                                                    
    Z2 = np.dot(W2, A1) + b2
    A2,cash2 = relu(Z2)                                                          
    Z3 = np.dot(W3, A2) + b3
    A3,cash3 = softmax(Z3)     
    cache = (Z1,A1, W1, b1, Z2,A2,W2,b2,Z3,A3, W3, b3)
    return  A2,A3,cache
#%%
def backward_propagation(X, Y, cache1):  
    m = X.shape[1]
    (Z1,A1, W1, b1, Z2, A2, W2, b2,Z3,A3, W3, b3) = cache1
    dZ3 =A3 -Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)            
    dZ2 = relu_backward(dA2,Z2)
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = relu_backward(dA1,Z1)
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2":dA2,"dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1":dA1,"dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
#%%
def initialize_adam(parameters) :
    L = len(parameters) // 2 
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])
        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])
    
    return v, s
#%%

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8): 
    L = len(parameters) // 2               
    v_corrected = {}                         
    s_corrected = {}                       
    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
        v_corrected["dW" + str(l+1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l+1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l+1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
        s_corrected["dW" + str(l+1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))
        parameters["W" + str(l+1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s["db" + str(l + 1)] + epsilon)
    return parameters, v, s
#%%
def initialize_parameters_he(layers_dims):
    parameters = {}
    parameters['W1'] = np.random.randn(layers_dims[1],layers_dims[0])*np.sqrt(2/layers_dims[0])
    parameters['b1'] = np.zeros((layers_dims[1],1))
    parameters['W2'] = np.random.randn(layers_dims[2],layers_dims[1])*np.sqrt(2/layers_dims[1])
    parameters['b2'] = np.zeros((layers_dims[2],1))
    parameters['W3'] = np.random.randn(layers_dims[3],layers_dims[2])*np.sqrt(2/layers_dims[2])
    parameters['b3'] = np.zeros((layers_dims[3],1))

        
    return parameters   
#%%  
def predict_new(X, y, parameters,g,af_list,bf_list,an_list,bn_list):
    HC_test,reconst_x =multi_network_test(X.T , g,af_list,bf_list,an_list,bn_list )
    m = X.shape[1]
    p = np.zeros((1,m))
    y_index=np.zeros((1,m))
    y_index=(np.argmax(y,axis=0)).reshape((1,m))
    probas, caches = forward_propagation(HC_test.T,X, parameters)
    for i in range(0, probas.shape[1]):
            max_index=np.where(probas[:,i]==probas[:,i].max())
            p[0,i] = max_index[0][0]
    print("Accuracy: "  + str(np.sum((p == y_index)/m)))
        
    return np.sum((p == y_index)/m)    
    
    