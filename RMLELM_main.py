# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:20:53 2018

@author: Hamid
"""
import time
from  dataset import load_mnist
from dataset import load_sat
from dataset import load_duke
from dataset import load_hill_valley,load_olivetti_faces
from dataset import load_usps,load_cars,load_leaves
import numpy as np 
from sklearn.model_selection import KFold 
from predict import predict_new,convert_to_one_hot
from basic_ML_ELM_RAE import ML_RELMAE
from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
from preprocess_dataset import preprocess_diabet,preprocess_iris,preprocess_cifar10
from preprocess_dataset import preprocess_Liver,preprocess_segment,preprocess_wine
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
def RMLELM_main(X_train,Y_train,X_test,Y_test):
#    hid_num1=[1]
#    hid_num=[hid_num1]
#    C1=[10**6,10**6]
#    C=[C1]
#    acc=np.zeros((5))
    accuracy=np.zeros((1))
#    pred_chain=np.zeros((len(C)))
#    for k in range(len(hid_num)):
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
#                betahat_1,betahat_3,Y=ML_ELM_train (Train_X,Train_Y,hid_num[k],10,C[kk])
#                Y_predict= ML_ELM_test(Test_X,Test_Y,betahat_1,betahat_3,10)
#                acc[kf_i]=predict_new(Test_Y,Y_predict)
#                kf_i=kf_i+1
#            pred_chain[kk]=np.sum(acc)/5 
    start = time.time()  
    n_hid=[200,200,200] 
    CC=[10**6,10**6,10**6]
    for i in range(1):
         Y_predict= ML_RELMAE(X_train,Y_train,X_test,CC,n_hid,5)
         accuracy[i]=predict_new(Y_test,Y_predict)
    final_acc= np.sum(accuracy)/1
    final_standard_div = np.sum((accuracy-final_acc)**2)/1
    stop = time.time()
    return final_acc,stop-start,final_standard_div
#%%
#acc_test_mnist,tim,final_standard_div_mnist= RMLELM_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
#acc_test_sat ,tim,final_standard_div_sat= RMLELM_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
#acc_test_duke ,tim,final_standard_div_duke= RMLELM_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
#acc_test_hill,tim,final_standard_div_hill= RMLELM_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
acc_test_uspss ,tim,final_standard_div_usps= RMLELM_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
#acc_test_face ,tim,final_standard_div_face= RMLELM_main(X_train_face,Y_train_face,X_test_face,Y_test_face)
#acc_test_diabetes,tim,final_standard_div_diabetes = RMLELM_main(X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet)  
#acc_test_iris,tim,final_standard_div_iris = RMLELM_main(X_train_iris,Y_train_iris,X_test_iris,Y_test_iris) 
#acc_test_cifar10,tim,final_standard_div_cifar10 = RMLELM_main(X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10) 
#acc_test_Liver,tim,final_standard_div_Liver = RMLELM_main(X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver)
#acc_test_segment,tim,final_standard_div_segment = RMLELM_main(X_train_segment,Y_train_segment,X_test_segment,Y_test_segment)  
#acc_test_wine,tim,final_standard_div_wine = RMLELM_main(X_train_wine,Y_train_wine,X_test_wine,Y_test_wine)         