# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:28:40 2018

@author: Hamid
"""

from basic_new_ELMAE_V2 import new_ELMAE_V2
import numpy as np 
import matplotlib.pyplot as plt
from predict import predict_new
from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
from basic_elm import ELM_train,ELM_test
X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
#X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
#X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
#X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
#X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
#X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
#%%
def new_ELMAE_V2_main():
    C=[2**10]
    num_hid=500
    H3_train,H3_test,reconst_X_test=new_ELMAE_V2(X_train_mnist,Y_train_mnist,X_test_mnist,C,[500,500,10])
#    plt.figure(figsize=(5,5))
#    for i in range(10):
#        plt.subplot(5,4,i+1)
#        plt.xticks([])
#        plt.yticks([])
#        plt.grid('off')
#        plt.imshow(reconst_X[i].reshape((28,28)), cmap=plt.cm.binary)
    W_new,Beta_hat,Y=ELM_train(H3_train,Y_train_mnist,num_hid,C)
    Y_predict  =ELM_test(H3_test,W_new,Beta_hat)
    accuracy=predict_new(Y_test_mnist,Y_predict)
    return reconst_X_test  ,accuracy
#%%
reconst_X_test  ,accuracy  =new_ELMAE_V2_main()
print(accuracy)
