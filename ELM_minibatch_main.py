# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:35:18 2018

@author: Hamid
"""

import time
import numpy as np 
from predict import predict_new
from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
from basic_ELM_minibatch import ELM_test_minibatch,ELM_train_minibatch
X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
#X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
#X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
#X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
#X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
#X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
def ELM_main(X_train,Y_train,X_test,Y_test):    
    accuracy_test=np.zeros((10))
    accuracy_train=np.zeros((10))
    n_hid=1000
    keep_prob = 0.5
    C=10**6
    for i in range(10):
        print(i)
        W,Beta_hat=ELM_train_minibatch(X_train[0:1000,:],Y_train[0:1000,:],n_hid,C,keep_prob)
        Y_predict_test=ELM_test_minibatch(X_test,W,Beta_hat)
        Y_predict_train=ELM_test_minibatch(X_train[0:1000,:],W,Beta_hat)
        accuracy_train[i]=predict_new(Y_train[0:1000,:],Y_predict_train)
        accuracy_test[i]=predict_new(Y_test,Y_predict_test)
    final_acc_train= np.sum(accuracy_train)/10     
    final_acc_test= np.sum(accuracy_test)/10 
#    final_standard_div = np.sum((accuracy-final_acc)**2)/10
#    stop = time.time()
    return final_acc_train,final_acc_test
#%%
acc_train_mnist,acc_test_mnist= ELM_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
#acc_test_sat ,tim,final_standard_div_sat= ELM_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
#acc_test_duke ,tim,final_standard_div_duke= ELM_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
#acc_test_hill,tim,final_standard_div_hill= ELM_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
#acc_test_uspss ,tim,final_standard_div_usps= ELM_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
#acc_test_face ,tim,final_standard_div_face= ELM_main(X_train_face,Y_train_face,X_test_face,Y_test_face)

