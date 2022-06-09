# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 12:46:17 2018

@author: Hamid
"""
import time
import numpy as np
from predict import predict_new
from basic_Autoencoder_with_inv_function import multi_network_train,multi_network_test
from basic_elm import ELM_train,ELM_test
from basic_new_ELMAE import T_ELM_Train,ELM_AE_RE
from basic_elm_deep_ELM_version import fusion_ELM_deep,forward_propagation
from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
from preprocess_dataset import preprocess_diabet,preprocess_iris,preprocess_cifar10
from preprocess_dataset import preprocess_Liver,preprocess_segment,preprocess_wine
#%%
def sigmoid_f(Z):
    A = 1/(1+np.exp(-Z))
    return A
#%%
def T_ELM_Test(X_test,Wie1,Wie2,Whe1,Whe2):
    ones = np.ones((X_test.shape[0],1))
    Xte= np.column_stack([X_test,ones])
    H1_temp = np.dot(Xte,Wie1)
    H1_temp=sigmoid_f(H1_temp)
    ones = np.ones((H1_temp.shape[0],1))
    H1_temp=np.column_stack([H1_temp,ones])
    H1=sigmoid_f(np.dot(H1_temp,Whe1))
    ones = np.ones((H1.shape[0],1))
    H2_temps=np.column_stack([H1,ones])
    H2_temps=np.dot(H2_temps,Wie2)
    H2_temps=sigmoid_f(H2_temps)
    ones = np.ones((H2_temps.shape[0],1))
    H2_temps=np.column_stack([H2_temps,ones])
    H2=sigmoid_f(np.dot(H2_temps,Whe2))
    return H2    
#%%
#X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
#X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
#X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
#X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
#X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
#X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet=preprocess_diabet()
#X_train_iris,Y_train_iris,X_test_iris,Y_test_iris=preprocess_iris()
#X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10=preprocess_cifar10()
#X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver=preprocess_Liver()
#X_train_segment,Y_train_segment,X_test_segment,Y_test_segment=preprocess_segment()
#X_train_wine,Y_train_wine,X_test_wine,Y_test_wine=preprocess_wine()
def elm_deep_main(X_train,Y_train,X_test,Y_test):
    start = time.time()  
    g="sigmoid"
    num_layer=2
    max_iter=5
    D=256
    C=10**6
#    Beta11,W11=ELM_AE_RE(X_train,D,C,max_iter) 
#    H1,Wie11,Whe11 =T_ELM_Train(X_train,Y_train,D,C,W11)
#    Beta21,W21=ELM_AE_RE(H1,D,C,max_iter) 
#    HC_train,Wie21,Whe21=T_ELM_Train(H1,Y_train,D,C,W21)
#    HC_train,reconst_x,af_list,bf_list,an_list,bn_list =multi_network_train(X_train, g,num_layer , max_iter , D ,C)
    A2_train,parameters = fusion_ELM_deep(X_train.T, Y_train.T,learning_rate =  0.001, num_iterations = 5, print_cost = True,beta1=0.99,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 64)
#    Beta12,W12=ELM_AE_RE(A2_train.T,D,C,max_iter) #np.column_stack([HC_train,A2_train.T])
#    H2,Wie12,Whe12 =T_ELM_Train(A2_train.T,Y_train,D,C,W12)
#    Beta22,W22=ELM_AE_RE(H2,D,C,max_iter) 
#    HC_train_fu,Wie22,Whe22=T_ELM_Train(H2,Y_train,D,C,W22)
    HC_train_fu,reconst_x_fu,af_list_fu,bf_list_fu,an_list_fu,bn_list_fu =multi_network_train(A2_train.T, g,num_layer , max_iter , D ,C)#np.column_stack([HC_train,A2_train.T])
#    HC_test=T_ELM_Test(X_test,Wie11,Wie21,Whe11,Whe21)
#    HC_test,reconst_x=multi_network_test(X_test,g,af_list,bf_list,an_list,bn_list)
    A2_test,A3,cache=forward_propagation(X_test.T,parameters)
#    HC_test_fu=T_ELM_Test(A2_test.T,Wie12,Wie22,Whe12,Whe22)
    HC_test_fu,reconst_x_fu =multi_network_test(A2_test.T, g,af_list_fu,bf_list_fu,an_list_fu,bn_list_fu )
    HC_train_fu_n=np.dot(HC_train_fu.T,HC_train_fu)
    one_matrix=np.identity(HC_train_fu_n.shape[0])
    one_matrix=one_matrix* 1/C
    new_H=HC_train_fu_n + one_matrix
    inverse_H=np.linalg.inv(new_H)
    Beta_hat=np.dot(np.dot(inverse_H,HC_train_fu.T),Y_train)
    Y_predict_train=np.dot(HC_train_fu,Beta_hat)
    Y_predict_test=np.dot(HC_test_fu,Beta_hat)
    pred_train=predict_new(Y_train,Y_predict_train)
    pred_test=predict_new(Y_test,Y_predict_test)
    stop = time.time()
    return pred_train,pred_test,stop-start
#%%
#acc_train_mnist,acc_test_mnist= elm_deep_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
#acc_train_sat,acc_test_sat = elm_deep_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
#acc_train_duke,acc_test_duke = elm_deep_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
#acc_train_hill,acc_test_hill= elm_deep_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
acc_train_usps,acc_test_uspss,tim= elm_deep_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
#acc_train_face,acc_test_face = elm_deep_main(X_train_face,Y_train_face,X_test_face,Y_test_face)  
#acc_train_diabetes,acc_test_diabetes = elm_deep_main(X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet)  
#acc_train_iris,acc_test_iris = elm_deep_main(X_train_iris,Y_train_iris,X_test_iris,Y_test_iris) 
#acc_train_cifar10,acc_test_cifar10 = elm_deep_main(X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10) 
#acc_train_Liver,acc_test_Liver = elm_deep_main(X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver)
#acc_train_segment,acc_test_segment = elm_deep_main(X_train_segment,Y_train_segment,X_test_segment,Y_test_segment)  
#acc_train_wine,acc_test_wine = elm_deep_main(X_train_wine,Y_train_wine,X_test_wine,Y_test_wine)      
#   
