# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:43:17 2018

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
#X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
#X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet=preprocess_diabet()
#X_train_iris,Y_train_iris,X_test_iris,Y_test_iris=preprocess_iris()
#X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10=preprocess_cifar10()
#X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver=preprocess_Liver()
X_train_segment,Y_train_segment,X_test_segment,Y_test_segment=preprocess_segment()
X_train_wine,Y_train_wine,X_test_wine,Y_test_wine=preprocess_wine()
def ELM_main(X_train,Y_train,X_test,Y_test):    
    accuracy=np.zeros((10))
#    pred_chain=np.zeros(len(C))
#    for k in range(len(L_M)):
#        print(k)
#        for kk in range(len(C)):
#            print(kk)
#            kf = KFold(n_splits=5)
#            kf_i=0
#            for train_index, test_index in kf.split(X_train_mnist):
#                Train_X, Test_X = X_train_mnist[train_index], X_train_mnist[test_index]
#                Train_Y, Test_Y = y_train_mnist[train_index], y_train_mnist[test_index]
#                Train_Y=convert_to_one_hot(Train_Y,10).T
#                Test_Y=convert_to_one_hot(Test_Y,10).T
#                W,Beta_hat,b,Y=ELM_train(Train_X,Train_Y,L_M[k],C[kk])
#                Y_predict=ELM_test(Test_X,W,b,Beta_hat)
#                acc[kf_i]=predict_new(Test_Y,Y_predict)
#                kf_i=kf_i+1
#            pred_chain[kk]=np.sum(acc)/5  
    n_hid=700#L_M[np.argmax(pred_chain)] 
    CC=10**6#C[np.argmax(pred_chain)] 


    start = time.time()  
    for i in range(1):
        print(i)
        W,Beta_hat,Y=ELM_train(X_train,Y_train,n_hid,CC)
        Y_predict=ELM_test(X_test,W,Beta_hat)
        accuracy[i]=predict_new(Y_test,Y_predict)
    final_acc= np.sum(accuracy)/1
    final_standard_div = np.sum((accuracy-final_acc)**2)/1
    stop = time.time()
    return final_acc,stop-start,final_standard_div
#%%
result=open("result_ELM.txt","w")
acc_test_mnist,tim,final_standard_div_mnist= ELM_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
result.write("tim_test_mnist:{}\n".format(tim))
acc_test_sat ,tim,final_standard_div_sat= ELM_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
result.write("tim_test_sat:{}\n".format(tim))
acc_test_duke ,tim,final_standard_div_duke= ELM_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
result.write("tim_test_duke:{}\n".format(tim))
acc_test_hill,tim,final_standard_div_hill= ELM_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
result.write("tim_test_hill:{}\n".format(tim))
acc_test_uspss ,tim,final_standard_div_usps= ELM_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
result.write("tim_test_usps:{}\n".format(tim))
#acc_test_face ,tim,final_standard_div_face= ELM_main(X_train_face,Y_train_face,X_test_face,Y_test_face)
#acc_test_diabetes,tim,final_standard_div_diabetes = ELM_main(X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet)  
#acc_test_iris,tim,final_standard_div_iris = ELM_main(X_train_iris,Y_train_iris,X_test_iris,Y_test_iris) 
#acc_test_cifar10,tim,final_standard_div_cifar10 = ELM_main(X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10) 
#acc_test_Liver,tim,final_standard_div_Liver = ELM_main(X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver)
#acc_test_segment,tim,final_standard_div_segment = ELM_main(X_train_segment,Y_train_segment,X_test_segment,Y_test_segment)  
#acc_test_wine,tim,final_standard_div_wine = ELM_main(X_train_wine,Y_train_wine,X_test_wine,Y_test_wine)      
