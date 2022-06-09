# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 12:46:17 2018

@author: Hamid
"""
from basic_Autoencoder_with_inv_function import multi_network_train,multi_network_test
from ConvertToonehot import convert_to_one_hot
from  dataset import load_mnist
from basic_elm_deep import fusion_ELM_deep,predict_new
from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
#X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
#X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
#X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
#X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
#X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
def elm_deep_main(X_train,Y_train,X_test,Y_test):
    g="sigmoid"
    num_layer=2
    max_loop=5
    D=128
    C=10**6
    HC_train,reconst_x,af_list,bf_list,an_list,bn_list =multi_network_train(X_train, g,num_layer , max_loop , D ,C)
    parameters,g,af_list,bf_list = fusion_ELM_deep(X_train.T, Y_train.T,D,"sigmoid",af_list,bf_list,an_list,bn_list,learning_rate =  0.001, num_iterations = 10, print_cost = True,beta1=0.99,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 64)
    pred_train = predict_new(X_train.T, Y_train.T, parameters,"sigmoid",af_list,bf_list,an_list,bn_list)
    pred_test = predict_new(X_test.T, Y_test.T, parameters,"sigmoid",af_list,bf_list,an_list,bn_list)
    return pred_train,pred_test
#%%
#acc_train_mnist,acc_test_mnist= elm_deep_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
#acc_train_sat,acc_test_sat = elm_deep_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
#acc_train_duke,acc_test_duke = elm_deep_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
acc_train_hill,acc_test_hill= elm_deep_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
#acc_train_usps,acc_test_uspss = elm_deep_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
#acc_train_face,acc_test_face = elm_deep_main(X_train_face,Y_train_face,X_test_face,Y_test_face)  