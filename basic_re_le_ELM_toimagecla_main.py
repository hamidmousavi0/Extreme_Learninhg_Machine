# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 07:25:38 2018

@author: Hamid
"""

import numpy as np
from  dataset import load_mnist
from basic_re_le_ELM_toimagecla import ELM_RE_image_train,ELM_RE_image_test
from sklearn.model_selection import KFold 
from predict import predict_new,convert_to_one_hot
def main_ELM_RE_image():
    mnist=load_mnist()
    X_train_mnist = mnist.train.images
    y_train_mnist = mnist.train.labels
    X_test_mnist =  mnist.test.images
    y_test_mnist = mnist.test.labels   
    X_train_zero = X_train_mnist[np.where(y_train_mnist == 0)][:]
    X_train_one = X_train_mnist[np.where(y_train_mnist == 1)][:]
    X_train_two = X_train_mnist[np.where(y_train_mnist == 2)][:]
    X_train_three = X_train_mnist[np.where(y_train_mnist == 3)][:] 
    X_train_four = X_train_mnist[np.where(y_train_mnist == 4)][:] 
    X_train_five = X_train_mnist[np.where(y_train_mnist == 5)][:] 
    X_train_six = X_train_mnist[np.where(y_train_mnist == 6)][:] 
    X_train_seven = X_train_mnist[np.where(y_train_mnist == 7)][:] 
    X_train_eight = X_train_mnist[np.where(y_train_mnist == 8)][:] 
    X_train_nine = X_train_mnist[np.where(y_train_mnist == 9)][:] 
    X_train_clases=[X_train_zero,X_train_one,X_train_two,X_train_three,
                    X_train_four,X_train_five,X_train_six,X_train_seven
                    ,X_train_eight,X_train_nine]
    num_hid=[150]
    C=[10**6]
    acc=np.zeros((5))
    accuracy=np.zeros((50))
    pred_chain=np.zeros(len(C))
    for k in range(len(num_hid)):
        print(k)
        for kk in range(len(C)):
            print(kk)
            kf = KFold(n_splits=2)
            kf_i=0
            for train_index, test_index in kf.split(X_train_mnist):
                Train_X, Test_X = X_train_mnist[train_index], X_train_mnist[test_index]
                Train_Y, Test_Y = y_train_mnist[train_index], y_train_mnist[test_index]
                Train_X_zero = Train_X[np.where(Train_Y == 0)][:]
                Train_X_one = Train_X[np.where(Train_Y == 1)][:]
                Train_X_two = Train_X[np.where(Train_Y == 2)][:]
                Train_X_three = Train_X[np.where(Train_Y == 3)][:] 
                Train_X_four = Train_X[np.where(Train_Y == 4)][:] 
                Train_X_five = Train_X[np.where(Train_Y == 5)][:] 
                Train_X_six = Train_X[np.where(Train_Y == 6)][:] 
                Train_X_seven = Train_X[np.where(Train_Y == 7)][:] 
                Train_X_eight = Train_X[np.where(Train_Y == 8)][:] 
                Train_X_nine = Train_X[np.where(Train_Y == 9)][:] 
                Train_X_clases=[Train_X_zero,Train_X_one,Train_X_two,
                                Train_X_three,Train_X_four,Train_X_five,
                                Train_X_six,Train_X_seven,Train_X_eight,
                                Train_X_nine]
               
                Train_Y=convert_to_one_hot(Train_Y,10).T
                Test_Y=convert_to_one_hot(Test_Y,10).T
                Wj=ELM_RE_image_train(Train_X,Train_X_clases,num_hid[k],C[kk]) 
                rec_error,x_Rec=ELM_RE_image_test(Test_X,Wj,C)
#                Y_predict=Deep_ELM_Test(Test_X,weigt,w,beta_hat)
#                acc[kf_i]=predict_new(Test_Y,Y_predict)
#                kf_i=kf_i+1
#            pred_chain[kk]=np.sum(acc)/5  
#    n_hid=150#L_M[np.argmax(pred_chain)] 
#    CC=10**6#C[np.argmax(pred_chain)] 
#    y_train_mnist=convert_to_one_hot(y_train_mnist,10).T
#    y_test_mnist=convert_to_one_hot(y_test_mnist,10).T 
#    X_train_mnist=X_train_mnist
#    X_test_mnist=X_test_mnist
#    for i in range(50):
#        print(i)
#        weigt,w = DeepELM_wights(X_train_mnist , y_train_mnist , 300 , 5 , 10)
#        beta_hat = Deep_ELM_Train(X_train_mnist,y_train_mnist , weigt,w,C)
#        Y_predict=Deep_ELM_Test(X_test_mnist,weigt,w,beta_hat)
#        accuracy[i]=predict_new(y_test_mnist,Y_predict)
#    final_acc= np.sum(accuracy)/50   
    return rec_error,x_Rec
#%%
rec_error,x_Rec=main_ELM_RE_image()    
            