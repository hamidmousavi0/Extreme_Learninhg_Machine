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
from preprocess_dataset import preprocess_diabet,preprocess_iris,preprocess_cifar10
from preprocess_dataset import preprocess_Liver,preprocess_segment,preprocess_wine
X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
#X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
#X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
#X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
#X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
##X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
#X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet=preprocess_diabet()
#X_train_iris,Y_train_iris,X_test_iris,Y_test_iris=preprocess_iris()
#X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10=preprocess_cifar10()
#X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver=preprocess_Liver()
#X_train_segment,Y_train_segment,X_test_segment,Y_test_segment=preprocess_segment()
#X_train_wine,Y_train_wine,X_test_wine,Y_test_wine=preprocess_wine()
from basic_DRELM import DrElm_test,DrElm_train
def DRELM_main(X_train,Y_train,X_test,Y_test):  
    accuracy_train=np.zeros((10))
    accuracy_test=np.zeros((10))
    num_iterations=10
    n_hid=500 
    C=[10**6]
    for i in range(10):
         W_list,Beta_list,W_prime_list=DrElm_train(X_train,Y_train,n_hid,0.1,C,num_iterations)
         Y_predict_test=DrElm_test(X_test,Y_test,n_hid,0.1,W_list,Beta_list,W_prime_list)
         Y_predict_train=DrElm_test(X_train,Y_train,n_hid,0.1,W_list,Beta_list,W_prime_list)
         accuracy_train[i]=predict_new(Y_train,Y_predict_train)
         accuracy_test[i]=predict_new(Y_test,Y_predict_test)
    final_acc_train= np.sum(accuracy_train)/10  
    final_acc_test= np.sum(accuracy_test)/10       
    final_standard_div = np.sum((accuracy_test-final_acc_test)**2)/10     
    
    return final_acc_train,final_acc_test,final_standard_div
#%%
acc_train_mnist,acc_test_mnist,final_standard_div_mnist= DRELM_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
#acc_train_sat,acc_test_sat,final_standard_div_sat = DRELM_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
#acc_train_duke,acc_test_duke,final_standard_div_duke = DRELM_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
#acc_train_hill,acc_test_hill,final_standard_div_hill= DRELM_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
#acc_train_usps,acc_test_uspss,final_standard_div_uspss= DRELM_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
#acc_train_face,acc_test_face,final_standard_div_face= DRELM_main(X_train_face,Y_train_face,X_test_face,Y_test_face)
#acc_train_diabetes,acc_test_diabetes,final_standard_div_diabetes = DRELM_main(X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet)  
#acc_train_iris,acc_test_iris,final_standard_div_iris = DRELM_main(X_train_iris,Y_train_iris,X_test_iris,Y_test_iris) 
#acc_train_cifar10,acc_test_cifar10,final_standard_div_cifar10 = DRELM_main(X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10) 
#acc_train_Liver,acc_test_Liver,final_standard_div_Liver = DRELM_main(X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver)
#acc_train_segment,acc_test_segment,final_standard_div_segment = DRELM_main(X_train_segment,Y_train_segment,X_test_segment,Y_test_segment)  
#acc_train_wine,acc_test_wine,final_standard_div_wine = DRELM_main(X_train_wine,Y_train_wine,X_test_wine,Y_test_wine)      