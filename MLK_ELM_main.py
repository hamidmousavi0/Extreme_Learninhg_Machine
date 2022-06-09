# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:54:37 2018

@author: Hamid
"""

from dataset import load_mnist
import numpy as np 
from sklearn.model_selection import KFold 
from predict import predict_new,convert_to_one_hot
from basic_MLK_ELM import ML_KELM_train,ML_KELM_test
def main_MLK_ELM():
    mnist=load_mnist()
    X_train_mnist = mnist.train.images
    y_train_mnist = mnist.train.labels
    X_test_mnist =  mnist.test.images
    y_test_mnist = mnist.test.labels   
    num_layer =[10]
    C=[10**6]
    sigmai=10**6
    acc=np.zeros((5))
    accuracy=np.zeros((50))
    pred_chain=np.zeros((len(C)))
    for k in range(len(num_layer)):
        print(k)
        for kk in range(len(C)):
            print(kk)
            kf = KFold(n_splits=5)
            kf_i=0
            for train_index, test_index in kf.split(X_train_mnist):
                Train_X, Test_X = X_train_mnist[train_index], X_train_mnist[test_index]
                Train_Y, Test_Y = y_train_mnist[train_index], y_train_mnist[test_index]
                Train_Y=convert_to_one_hot(Train_Y,10).T
                Test_Y=convert_to_one_hot(Test_Y,10).T
                X_final,gamma_tild_1,gamma_tild_unified,Beta_hat=ML_KELM_train (Train_X,Train_Y,C[kk],sigmai,num_layer[k],"sigmoid")
                Y_predict= ML_KELM_test(Test_X,gamma_tild_1,gamma_tild_unified,Beta_hat,sigmai)
                acc[kf_i]=predict_new(Test_Y,Y_predict)
                kf_i=kf_i+1
            pred_chain[kk]=np.sum(acc)/5 
    num_layer_n=[100] 
    CC=[10**6]
    y_train_mnist=convert_to_one_hot(y_train_mnist,10).T
    y_test_mnist=convert_to_one_hot(y_test_mnist,10).T 
    X_train_mnist=X_train_mnist
    X_test_mnist=X_test_mnist
    for i in range(50):
         X_final,gamma_tild_1,gamma_tild_unified,Beta_hat=ML_KELM_train (X_train_mnist,y_train_mnist,CC[kk],sigmai,num_layer_n[k],"sigmoid")
         Y_predict= ML_KELM_test(X_test_mnist,gamma_tild_1,gamma_tild_unified,Beta_hat,sigmai)
         accuracy[i]=predict_new(y_test_mnist,Y_predict)
    final_acc= np.sum(accuracy)/50   
    return final_acc   
#%%
acc= main_MLK_ELM()    