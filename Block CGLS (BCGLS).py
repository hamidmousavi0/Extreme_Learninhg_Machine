# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:30:56 2018

@author: Hamid
"""
from predict import predict_new
from preprocess_dataset import preprocess_MNIST
import numpy as np 
def BCGLS (A,B,X0,tol,maxitr):
    Ri = B - np.dot(A,X0)
    Si = np.dot(A.T,Ri)
    Pi = Si
    X_sol = X0
    for i in range (maxitr):
        Qi = np.dot(A,Pi)
        alphai = np.dot(np.dot(np.linalg.inv(np.dot(Qi.T,Qi)),Si.T),Si)
        X_sol = X_sol +np.dot(Pi,alphai)
        Ri = Ri  -np.dot(Qi,alphai)
        Si = np.dot(A.T,Ri)
        betai =np.dot(np.dot(np.linalg.inv(np.dot(Si.T,Si)),Si.T),Si)
        Pi = Si + np.dot(Pi,betai)
    return X_sol
#%%
def initialize_parameters_random(num_X,num_hid):             
    parameters = {}
#    parameters['W'] = np.random.uniform(np.sqrt(-6/num_hid+num_X),np.sqrt(6/num_hid+num_X),(num_X,num_hid))
    parameters['W'] = np.random.randn(num_X,num_hid)
    parameters['b'] = np.random.randn(1,num_hid)
    return parameters    
#%%
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A
    
#%%
X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
X_train_new = np.zeros((X_train_mnist.shape[0],X_train_mnist.shape[1]+1))
X_train_new[:,0:X_train_mnist.shape[1]]=X_train_mnist
X_train_new[:,X_train_mnist.shape[1]]=1   
parameter=initialize_parameters_random(X_train_mnist.shape[1],1000)
W=parameter['W']
b=parameter['b'] 
W_new = np.zeros((X_train_mnist.shape[1]+1,1000))
W_new[0:X_train_mnist.shape[1],:]=W
W_new[X_train_mnist.shape[1],:]=b
temp_H=(np.dot(X_train_new,W_new))
H=sigmoid(temp_H)
X0=initialize_parameters_random(1000,Y_train_mnist.shape[1])["W"]
Beta = BCGLS(H,Y_train_mnist,X0,0.001,50)
X_new = np.zeros((X_test_mnist.shape[0],X_train_mnist.shape[1]+1))
X_new[:,0:X_test_mnist.shape[1]]=X_test_mnist
X_new[:,X_test_mnist.shape[1]]=1
temp_H=(np.dot(X_new,W_new))
h=sigmoid(temp_H)
Y_predict_t=np.dot(H,Beta)
Y_predict=np.dot(h,Beta)
accuracy_t=predict_new(Y_train_mnist,Y_predict_t)
accuracy=predict_new(Y_test_mnist,Y_predict)
#%%






































#%%
 