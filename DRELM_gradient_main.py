# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:58:50 2018

@author: admin
"""
import time
import numpy as np 
from sklearn.model_selection import KFold 
from predict import predict_new,convert_to_one_hot
from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
from basic_elm import ELM_test,ELM_train
X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
#X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
#X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
#X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
#X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
#X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
from basic_DRELM_gradient import DrElm_test,DrElm_train
def DRELM_main(X_train,Y_train,X_test,Y_test):  
    accuracy_train=np.zeros((1))
    accuracy_test=np.zeros((1))
    n_hid=500 
    C=[10**6]
    for i in range(1):
         W,Beta=DrElm_train(X_train,Y_train,n_hid,1,C,5)
         Y_predict_test=DrElm_test(X_test,W,Beta)
         Y_predict_train=DrElm_test(X_train,W,Beta)
         accuracy_train[i]=predict_new(Y_train,Y_predict_train)
         accuracy_test[i]=predict_new(Y_test,Y_predict_test)
         
    final_acc_train= np.sum(accuracy_train)/1  
    final_acc_test= np.sum(accuracy_test)/1  
    return final_acc_train,final_acc_test
#%%
acc_train_mnist,acc_test_mnist= DRELM_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
#acc_test_sat = DRELM_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
#acc_test_duke = DRELM_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
#acc_test_hill,tim= DRELM_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
#acc_test_uspss= DRELM_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
#acc_test_face = DRELM_main(X_train_face,Y_train_face,X_test_face,Y_test_face)   