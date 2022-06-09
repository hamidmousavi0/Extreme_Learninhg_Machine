# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:57:07 2018

@author: Hamid
"""
from predict import predict_new
from preprocess_dataset import preprocess_MNIST
X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
import numpy as np
from scipy.io import loadmat
H = np.array(loadmat("H_new")["H"])
T = np.array(loadmat("T_new")["T"])
H_test = np.array(loadmat("H_test")["h"])
beta1 = np.array(loadmat("W1")["finalX1"])
beta2 = np.array(loadmat("W2")["finalX2"])
beta3 = np.array(loadmat("W3")["finalX3"])
Y_predict=np.dot(H_test,beta2)
accuracy=predict_new(Y_test_mnist,Y_predict)
print(accuracy)