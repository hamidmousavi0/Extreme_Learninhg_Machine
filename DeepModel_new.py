# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:55:41 2017

@author: Hamid
"""
"""
data set bayad be in sorat bashad ke train_X=(num_feature,num_sample) train_Y=(num_class,num_sample)
"""
#%%
from initializeparameter import initialize_parameters_he
from RandomMinibatch import random_mini_batches
from ForwardPropagation import L_model_forward,forward_propagation_with_dropout
from Cost import compute_cost_crossentropy,compute_cost_softmax
from Cost import compute_cost_with_regularization,compute_cost_meansqure
from BackwardProp import L_model_backward,backward_propagation_with_dropout
from BackwardProp import backward_propagation_with_regularization
import numpy as np
from UpdateParameter import update_parameters_with_adam,initialize_adam
from UpdateParameter import update_parameters_with_gd,initialize_velocity
from UpdateParameter import update_parameters_with_momentum
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


#%%
def L_layer_model(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=False,keep_prob=1,lambd=0,mini_batch_size = 64):
    seed=10
    costs = []                        
    parameters = initialize_parameters_he(layers_dims)
    for i in range(0, num_iterations):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            if keep_prob == 1:
                 AL, caches = L_model_forward(X, parameters)
            elif keep_prob < 1:
                 AL, caches = forward_propagation_with_dropout(X, parameters, keep_prob)
            if lambd == 0:
                cost =  compute_cost_crossentropy(AL, Y)
            else:
                cost = compute_cost_with_regularization(AL, Y, parameters, lambd)
            if lambd == 0 and keep_prob == 1:
                grads =  L_model_backward(AL, Y, caches)
            elif lambd != 0:
                grads = backward_propagation_with_regularization(X, Y, caches, lambd)
            elif keep_prob < 1:
                grads = backward_propagation_with_dropout(X, Y, caches, keep_prob)
            parameters = update_parameters_with_gd(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

#%%
def five_layer_model(X, Y, layers_dims, learning_rate = 0.001, num_iterations = 1000, print_cost=False,keep_prob=1,beta1=0.9,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 128):
     grads = {}
     costs = []  
     costs_temp =[]
     t=0
     seed=10
     m = X.shape[1]   
     (n_x, n_h1,n_h2,v_h3,n_h4, n_y) = layers_dims
     parameters = initialize_parameters_he(layers_dims)
     v,s=initialize_adam(parameters)
    # v=initialize_velocity(parameters)
     for i in range(0, num_iterations):
         seed = seed + 1
         minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
         for minibatch in minibatches:
             (minibatch_X, minibatch_Y) = minibatch
             A5, cache =L_model_forward(minibatch_X, parameters)
         #   A5, cache =forward_propagation_with_dropout(minibatch_X, parameters, keep_prob )
             cost = compute_cost_softmax(A5.T,minibatch_Y)
        # cost = compute_cost_softmax(A5,Y)
             grads= L_model_backward(A5, minibatch_Y, cache)
    #         grads= backward_propagation_with_dropout(minibatch_X, minibatch_Y, cache, keep_prob)
             t=t+1
             parameters,v,s= update_parameters_with_adam(parameters, grads, v, s, t, learning_rate ,
                                  beta1 , beta2 ,  epsilon )
             costs_temp.append(cost)
        # parameters,v= update_parameters_with_momentum(parameters, grads, v, beta1, learning_rate)
         costs_temp_new = np.array(costs_temp)
         costs_temp_new = np.mean(costs_temp_new)
         if print_cost :
            print("Cost after iteration {}: {}".format(i, np.squeeze(costs_temp_new)))
         if print_cost :
            costs.append(costs_temp_new)
     plt.figure()       
     plt.plot(np.squeeze(costs))
     plt.ylabel('cost')
     plt.xlabel('iterations (per tens)')
     plt.title("Learning rate =" + str(learning_rate))
     plt.show()
    
     return parameters  
 #%%
def three_layer_model(X, Y, layers_dims, learning_rate = 0.001, num_iterations = 3000, print_cost=False,keep_prob=0.5,beta1=0.9,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 32):
     grads = {}
     costs = []  
     m = X.shape[1] 
     t=0
     costs_temp =[]
     seed=10
     (n_x, n_h1,n_h2,n_y) = layers_dims
     parameters = initialize_parameters_he(layers_dims)
     v,s=initialize_adam(parameters) 
     for i in range(0, num_iterations):
         seed = seed + 1
         minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
         for minibatch in minibatches:
             (minibatch_X, minibatch_Y) = minibatch
             A3, cache =L_model_forward(minibatch_X, parameters)
             cost = compute_cost_softmax(A3.T,minibatch_Y)
             grads= L_model_backward(A3, minibatch_Y, cache)
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
     plt.figure()       
     plt.plot(np.squeeze(costs))
     plt.ylabel('cost')
     plt.xlabel('iterations (per tens)')
     plt.title("Learning rate =" + str(learning_rate))
     plt.show()    
    
     return parameters      
#%%
#def fuzzy_model(X, Y, num_mem, learning_rate = 0.001, num_iterations = 3000, print_cost=False,keep_prob=0.5,beta1=0.9,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 32)     
