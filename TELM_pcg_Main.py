# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:45:20 2018

@author: Hamid
"""
from basic_TELM_pcg import T_ELM_Train,T_ELM_Test
from sklearn.model_selection import KFold 
from predict import predict_new,convert_to_one_hot
import numpy as np 
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
#X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
#X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
#X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
#X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
#X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
def TELM_main(X_train,Y_train,X_test,Y_test):    
    accuracy_test=np.zeros((10))
    accuracy_train=np.zeros((10))
#    pred_chain=np.zeros(len(num_layer))
#    for k in range(len(num_layer)):
#        print("baraye",num_layer[k])
#        for kk in range(len(C)):
#            print("baraye",C[kk])
#            kf = KFold(n_splits=5)
#            kf_i=0
#            for train_index, test_index in kf.split(PX_train):
#                Train_X, Test_X = PX_train[train_index], PX_train[test_index]
#                Train_Y, Test_Y = PY_train[train_index], PY_train[test_index]
#                Train_Y=convert_to_one_hot(Train_Y,10).T
#                Test_Y=convert_to_one_hot(Test_Y,10).T
#                weigt,w = DeepELM_wights(Train_X , Train_Y , L_M[0] , num_layer[k] , C[kk]) 
#                beta_hat = Deep_ELM_Train(Train_X,Train_Y , weigt,w,L_M[0],C[kk])
#                Y_predict=Deep_ELM_Test(Test_X,weigt,w,beta_hat,L_M[0])
#                RMS[kf_i] = sqrt(mean_squared_error(Test_Y, Y_predict))
#                print("error dar split",RMS[kf_i])
#                acc[kf_i]=predict_new(Test_Y,Y_predict)
#                print("error dar split",acc[kf_i])
#                kf_i=kf_i+1
#        pred_chain[k]=np.sum(acc)/5
#        pred_RMSE[k] =np.sum(RMS)/5
#        print("miangin",pred_chain[k])
    n_hid=700#L_M[np.argmax(pred_chain)] 
    C=10**6#C[np.argmax(pred_chain)] 
    import time
    start = time.time()  
    for i in range(1):
        print(i)
        Wie,Whe,Beta_new=T_ELM_Train(X_train,Y_train,n_hid,C)
        Y_predict_test=T_ELM_Test(X_test,Wie,Whe,Beta_new)
        Y_predict_train=T_ELM_Test(X_train,Wie,Whe,Beta_new)
        accuracy_test[i]=predict_new(Y_test,Y_predict_test)
        accuracy_train[i]=predict_new(Y_train,Y_predict_train)
    final_acc_test= np.sum(accuracy_test)/1 
    final_acc_train= np.sum(accuracy_train)/1
    final_standard_div = np.sum((accuracy_test-final_acc_test)**2)/1
    stop = time.time()    
    return final_acc_test,final_acc_train,stop-start,final_standard_div
#%%
acc_test_mnist,acc_train_mnist ,tim,final_standard_div_mnist= TELM_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
#acc_test_sat,acc_train_sat ,tim,final_standard_div_sat= TELM_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
#acc_test_duke,acc_train_duke ,tim,final_standard_div_duke= TELM_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
#acc_test_hill,acc_train_hill ,tim,final_standard_div_hill= TELM_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
#acc_test_usps,acc_train_usps ,tim,final_standard_div_usps= TELM_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
#acc_test_face,acc_train_face ,tim,final_standard_div_face= TELM_main(X_train_face,Y_train_face,X_test_face,Y_test_face)



    