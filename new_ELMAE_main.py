# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:59:28 2018

@author: Hamid
"""
from basic_new_ELMAE import new_ELMAE
from ConvertToonehot import convert_to_one_hot
import numpy as np 
import time
import matplotlib.pyplot as plt
from DeepModel_new import five_layer_model
from RandomMinibatch import random_mini_batches
from Cost import compute_cost_softmax
from Sigmoid import sigmoid
from Relu import relu,relu_backward
from Softmax import softmax
from predict import predict_new
from basic_elm import ELM_train,ELM_test
from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
from preprocess_dataset import preprocess_diabet,preprocess_iris,preprocess_cifar10
from preprocess_dataset import preprocess_Liver,preprocess_segment,preprocess_wine
X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
# X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
#X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet=preprocess_diabet()
#X_train_iris,Y_train_iris,X_test_iris,Y_test_iris=preprocess_iris()
#X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10=preprocess_cifar10()
#X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver=preprocess_Liver()
#X_train_segment,Y_train_segment,X_test_segment,Y_test_segment=preprocess_segment()
#X_train_wine,Y_train_wine,X_test_wine,Y_test_wine=preprocess_wine()
#%%
def new_ELMAE_main(X_train,Y_train,X_test,Y_test):
    accuracy_test=np.zeros((1))
    n_hid=200 
    C=[10**6]
    start = time.time() 
    Y_predict=new_ELMAE(X_train,Y_train,X_test,C,n_hid,5)
#    plt.figure(figsize=(5,5))
#    for i in range(10):
#        plt.subplot(5,4,i+1)
#        plt.xticks([])
#        plt.yticks([])
#        plt.grid('off')
#        plt.imshow(reconst_X[i].reshape((28,28)), cmap=plt.cm.binary)

    for i in range(1):
        accuracy_test[i]=predict_new(Y_test,Y_predict)
        stop = time.time()  
    final_acc_test= np.sum(accuracy_test)/1  
    final_standard_div = np.sum((accuracy_test-final_acc_test)**2)/1    
    return final_acc_test,stop-start,final_standard_div
#%%

def model_deep(X,Y,learning_rate =  0.001, num_iterations = 100, print_cost = True,beta1=0.99,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 64):
    grads = {}
    costs = []  
    costs_temp =[]
    t=0
    seed=10
    m = X.shape[1]  
    n_x = X.shape[0]
    n_h1=64
    n_h2=64
    n_h3=64
    n_h4=64
    n_y=10
    layers_dims=(n_x, n_h1,n_h2,n_h3,n_h4,n_y)
    parameters= initialize_parameters_he(layers_dims)
    v,s=initialize_adam(parameters)
    for i in range(0, num_iterations):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            A4,cache=forward_propagation(minibatch_X,parameters)
            cost = compute_cost_softmax(A4,minibatch_Y)
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
    parameters['W2'] = np.random.randn(layers_dims[2],layers_dims[1])*np.sqrt(2/layers_dims[1])
    parameters['b2'] = np.zeros((layers_dims[2],1))
    parameters['W3'] = np.random.randn(layers_dims[4],layers_dims[3])*np.sqrt(2/layers_dims[3])
    parameters['b3'] = np.zeros((layers_dims[4],1))
    parameters['W4'] = np.random.randn(layers_dims[5],layers_dims[4])*np.sqrt(2/layers_dims[4])
    parameters['b4'] = np.zeros((layers_dims[5],1))
        
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
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]  
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]  
    Z1 = np.dot(W1, X) + b1
    A1,cash1 = relu(Z1)                                                                    
    Z2 = np.dot(W2, A1) + b2
    A2,cash2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3,cash3 = relu(Z3)                                                                   
    Z4 = np.dot(W4, A3) + b4
    A4,cash4 = softmax(Z4)     
    cache = (Z1,A1, W1, b1, Z2,A2,W2,b2,Z3,A3, W3, b3, Z4,A4,W4,b4)
    return  A4,cache
#%%
def backward_propagation(X, Y, cache1):  
    m = X.shape[1]
    (Z1,A1, W1, b1, Z2, A2, W2, b2,Z3,A3, W3, b3, Z4, A4, W4, b4) = cache1
    dZ4 =A4 -Y
    dW4 = 1./m * np.dot(dZ4, A3.T)
    db4 = 1./m * np.sum(dZ4, axis=1, keepdims = True)
    dA3 = np.dot(W4.T, dZ4)            
    dZ3 = relu_backward(dA3,Z3)
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)[0:64,:]
    dZ2 = relu_backward(dA2,Z2)
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = relu_backward(dA1,Z1)
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ4": dZ4, "dW4": dW4, "db4": db4,
                 "dA3":dA3,"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2":dA2,"dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1":dA1,"dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
#%%    
def predict_new_deep(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1,m))
    y_index=np.zeros((1,m))
    y_index=(np.argmax(y,axis=0)).reshape((1,m))
    probas, caches = forward_propagation(X, parameters)
    for i in range(0, probas.shape[1]):
            max_index=np.where(probas[:,i]==probas[:,i].max())
            p[0,i] = max_index[0][0]
    print("Accuracy: "  + str(np.sum((p == y_index)/m)))
        
    return np.sum((p == y_index)/m)   
#%%
result=open("result_proposed.txt","w")
acc_test_mnist,tim,final_standard_div_mnist= new_ELMAE_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
result.write("tim_test_mnist:{}\n".format(tim))
acc_test_sat,tim,final_standard_div_sat= new_ELMAE_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
result.write("tim_test_sat:{}\n".format(tim))
acc_test_duke,tim,final_standard_div_duke= new_ELMAE_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
result.write("tim_test_duke:{}\n".format(tim))
acc_test_hill,tim,final_standard_div_hill= new_ELMAE_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
result.write("tim_test_hill:{}\n".format(tim))
acc_test_usps,tim,final_standard_div_usps= new_ELMAE_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
result.write("tim_test_usps:{}\n".format(tim))
# acc_test_face,final_standard_div_face= new_ELMAE_main(X_train_face,Y_train_face,X_test_face,Y_test_face)
#acc_test_diabetes,final_standard_div_diabetes = new_ELMAE_main(X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet)  
#acc_test_iris,final_standard_div_iris = new_ELMAE_main(X_train_iris,Y_train_iris,X_test_iris,Y_test_iris) 
#acc_test_cifar10,final_standard_div_cifar10 = new_ELMAE_main(X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10) 
#acc_test_Liver,final_standard_div_Liver = new_ELMAE_main(X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver)
#acc_test_segment,final_standard_div_segment = new_ELMAE_main(X_train_segment,Y_train_segment,X_test_segment,Y_test_segment)  
#acc_train_wine,final_standard_div_wine = new_ELMAE_main(X_train_wine,Y_train_wine,X_test_wine,Y_test_wine)      


    