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
from preprocess_dataset import preprocess_diabet,preprocess_iris,preprocess_cifar10
from preprocess_dataset import preprocess_Liver,preprocess_segment,preprocess_wine
from basic_elm import ELM_test,ELM_train
X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
# X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
#X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet=preprocess_diabet()
#X_train_iris,Y_train_iris,X_test_iris,Y_test_iris=preprocess_iris()
#X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10=preprocess_cifar10()
#X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver=preprocess_Liver()
#X_train_segment,Y_train_segment,X_test_segment,Y_test_segment=preprocess_segment()
#X_train_wine,Y_train_wine,X_test_wine,Y_test_wine=preprocess_wine()
from basic_stack_elm import DrElm_test,DrElm_train
def DRELM_main(X_train,Y_train,X_test,Y_test):  
    accuracy_train=np.zeros((1))
    accuracy_test=np.zeros((1))
    n_hid=700
    num_iterations=5
    num_class=Y_train.shape[1]
    C=[10**6]
    start = time.time() 
    for i in range(1):
         W,Beta=DrElm_train(X_train,Y_train,n_hid,C,num_iterations)
         Y_predict_test=DrElm_test(X_test,W,Beta,num_class)
         Y_predict_train=DrElm_test(X_train,W,Beta,num_class)
         accuracy_train[i]=predict_new(Y_train,Y_predict_train)
         accuracy_test[i]=predict_new(Y_test,Y_predict_test)
    stop = time.time()     
    final_acc_train= np.sum(accuracy_train)/1 
    final_acc_test= np.sum(accuracy_test)/1
    return final_acc_train,stop-start,final_acc_test
#%%
result=open("result_stack.txt","w")
acc_train_mnist,tim,acc_test_mnist= DRELM_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
result.write("tim_test_mnist:{}\n".format(tim))
acc_train_sat,tim,acc_test_sat = DRELM_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
result.write("tim_test_sat:{}\n".format(tim))
acc_train_duke,tim,acc_test_duke = DRELM_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
result.write("tim_test_duke:{}\n".format(tim))
acc_train_hill,tim,acc_test_hill= DRELM_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
result.write("tim_test_hill:{}\n".format(tim))
acc_train_usps,tim,acc_test_uspss= DRELM_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
result.write("tim_test_usps:{}\n".format(tim))
# acc_train_face,acc_test_face = DRELM_main(X_train_face,Y_train_face,X_test_face,Y_test_face)   
#acc_train_diabetes,acc_test_diabetes = DRELM_main(X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet)  
#acc_train_iris,acc_test_iris = DRELM_main(X_train_iris,Y_train_iris,X_test_iris,Y_test_iris) 
#acc_train_cifar10,acc_test_cifar10 = DRELM_main(X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10) 
#acc_train_Liver,acc_test_Liver = DRELM_main(X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver)
#acc_train_segment,acc_test_segment = DRELM_main(X_train_segment,Y_train_segment,X_test_segment,Y_test_segment)  
#acc_train_wine,acc_test_wine = DRELM_main(X_train_wine,Y_train_wine,X_test_wine,Y_test_wine)      