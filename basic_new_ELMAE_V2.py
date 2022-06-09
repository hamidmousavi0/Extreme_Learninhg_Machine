# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:13:47 2018

@author: Hamid
"""
import matplotlib.pyplot as plt
import numpy as np
from ConvertToonehot import convert_to_one_hot
from RandomMinibatch import random_mini_batches
from Cost import compute_cost_softmax
from Sigmoid import sigmoid,sigmoid_backward
from Relu import relu,relu_backward
from Softmax import softmax
#%%
def new_ELMAE_V2(X_train,Y_train,X_test,C,hidden_number):
    parameters=model_deep(X_train.T,Y_train.T,hidden_number[0],C,learning_rate =  0.01, num_iterations = 100, print_cost = True,beta1=0.99,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 256)
    W1=parameters["W1"]
    b1=parameters["b1"]
    H1=np.dot(X_train,W1)+b1
    H1=sigmoid(H1)
    parameters=model_deep(H1,Y_train,hidden_number[1],C,learning_rate =  0.001, num_iterations = 100, print_cost = True,beta1=0.99,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 64)
    W2=parameters["W1"]
    b2=parameters["b1"]
    H2=np.dot(H1,W2)+b2
    H2=sigmoid(H2)
    parameters=model_deep(H2,Y_train,hidden_number[2],C,learning_rate =  0.001, num_iterations = 100, print_cost = True,beta1=0.99,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 64)
    W3=parameters["W1"]
    b3=parameters["b1"]
    H3=np.dot(H2,W3)+b3
    H3=sigmoid(H3)
    reconst_H2 = sigmoid(np.dot(H3,W3.T)+b3)
    reconst_H1 = sigmoid(np.dot(reconst_H2,W2.T)+b2)
    reconst_X = sigmoid(np.dot(reconst_H1,W1.T)+b1)
    H1_test=np.dot(X_test,W1)+b1
    H1_test=sigmoid(H1_test)
    H2_test = np.dot(H1_test,W2)+b2
    H2_test = sigmoid(H2_test)
    H3_test = np.dot(H2_test,W3)
    H3_test = sigmoid(H3_test)
    reconst_H2_test = sigmoid(np.dot(H3_test,W3.T)+b3)
    reconst_H1_test = sigmoid(np.dot(reconst_H2_test,W2.T)+b2)
    reconst_X_test = sigmoid(np.dot(reconst_H1_test,W1.T)+b1)
    return H3,H3_test,reconst_X_test
#%%
def model_deep(X,Y,hidden_number,C,learning_rate =  10, num_iterations = 100, print_cost = True,beta1=0.99,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 64):
    grads = {}
    costs = []  
    costs_temp =[]
    t=0
    seed=10
    m = X.shape[1]  
    n_x = X.shape[0]
    n_h1=hidden_number
    n_y=Y.shape[0]
    layers_dims=(n_x, n_h1)
    parameters= initialize_parameters_he(layers_dims)
    v,s=initialize_adam(parameters)
    for i in range(0, num_iterations):
        seed = seed + 1
            
        A2,cache=forward_propagation(X,Y,parameters,C)
        cost = compute_cost_softmax(A2,Y)
        grads= backward_propagation(X, Y, cache)
        t=t+1
        parameters,v,s= update_parameters_with_adam(parameters, grads, v, s, t, learning_rate ,
                                  beta1 , beta2 ,  epsilon )
#        costs_temp.append(cost)
#        costs_temp_new = np.array(costs_temp)
#        costs_temp_new = np.mean(costs_temp_new)
        if print_cost :
           print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost :
           costs.append(cost)
    plt.figure()       
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
#%%
def initialize_parameters_he(layers_dims):
    parameters = {}
    parameters['W1'] = np.random.randn(layers_dims[1],layers_dims[0])*np.sqrt(2/layers_dims[0])
    parameters['b1'] = np.zeros((layers_dims[1],1))   
    return parameters    
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
def forward_propagation(X,Y, parameters,C):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    Z1 = np.dot(W1, X) + b1
    A1,cash1 = relu(Z1) 
    A1_n=np.dot(A1,A1.T)
    one_matrix=np.identity(A1_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_A1=A1_n + one_matrix
    inverse_A1=np.linalg.inv(new_A1)
    Beta_hat=np.dot(np.dot(inverse_A1,A1),Y.T) 
    Z2=np.dot(A1.T,Beta_hat) 
    A2,cach = softmax(Z2)                                                                 
    cache = (Z1,A1, W1, b1, Z2,A2,Beta_hat)
    return  A2,cache
#%%
def backward_propagation(X, Y, cache1):  
    m = X.shape[1]
    (Z1,A1, W1, b1, Z2, A2, Beta) = cache1
    dZ2 =A2.T -Y
    dA1 = np.dot(Beta, dZ2)            
    dZ1 = relu_backward(dA1,Z1)
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    gradients = {"dZ2": dZ2, "dA1":dA1,"dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
#%%
    
