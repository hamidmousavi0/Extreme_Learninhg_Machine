# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:14:48 2018

@author: Hamid
"""

from dataset import load_mnist
import numpy as np 
from sklearn.model_selection import KFold 
from predict import predict_new,convert_to_one_hot
from basic_HElM import H_ELM_train,H_ELM_test
from scipy.linalg import orth
def main_ML_ELM():
    N1=100
    N2 = 50
    N3=200
    mnist=load_mnist()
    X_train_mnist = mnist.train.images
    y_train_mnist = mnist.train.labels
    X_test_mnist =  mnist.test.images
    y_test_mnist = mnist.test.labels   
    y_train_mnist=convert_to_one_hot(y_train_mnist,10).T
    y_test_mnist=convert_to_one_hot(y_test_mnist,10).T 
    C = 2^-30 
    b1=2*np.random.rand(N1,X_train_mnist.shape[1]+1)-1
    b2=2*np.random.rand(N2,N1+1)-1
    b3=orth(2*np.random.rand(N3,N2+1))
    beta1,beta2,beta3= H_ELM_train(X_train_mnist,y_train_mnist,b1,b2,b3,C)
    Y_predict=H_ELM_test(X_test_mnist,beta1,beta2,beta3)
    accuracy=predict_new(y_test_mnist,Y_predict)
    
    return accuracy
#%%
accuracy=main_ML_ELM ()   