# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:48:57 2018

@author: Hamid
"""
import time
from  dataset import load_mnist
from dataset import load_sat
from dataset import load_duke
from dataset import load_hill_valley,load_olivetti_faces
from dataset import load_usps,load_cars,load_leaves
import numpy as np 
from basic_Autoencoder_with_inv_function import multi_network_train,multi_network_test
from sklearn.model_selection import KFold 
from predict import predict_new,convert_to_one_hot
import numpy as np 
from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
from basic_elm import ELM_train,ELM_test
from preprocess_dataset import preprocess_diabet,preprocess_iris,preprocess_cifar10,preprocess_iris2
from preprocess_dataset import preprocess_Liver,preprocess_segment,preprocess_wine
X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
#X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet=preprocess_diabet()
# X_train_iris,Y_train_iris,X_test_iris,Y_test_iris=preprocess_iris2()
#X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10=preprocess_cifar10()
X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver=preprocess_Liver()
X_train_segment,Y_train_segment,X_test_segment,Y_test_segment=preprocess_segment()
X_train_wine,Y_train_wine,X_test_wine,Y_test_wine=preprocess_wine()
def AE_INV_main(X_train,Y_train,X_test,Y_test):    
#    D=[150]
#    C=[2**10]
#    acc=np.zeros((5))
    accuracy=np.zeros((10))
#    pred_chain=np.zeros(len(C))
#    for k in range(len(D)):
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
#                HC_train,reconst_x,af_list,bf_list=multi_network_train (Train_X,'sigmoid',5, 50,D[k],C[kk])
#                W,Beta_hat,b,Y=ELM_train(HC_train,Train_Y,D[k],C[kk])
#                HC_test=multi_network_test (Test_X , 'sigmoid',af_list,bf_list)
#                Y_predict=ELM_test(HC_test,W,b,Beta_hat)
#                acc[kf_i]=predict_new(Test_Y,Y_predict)
#                kf_i=kf_i+1
#            pred_chain[kk]=np.sum(acc)/5  
    n_hid=200#L_M[np.argmax(pred_chain)] 
    CC=10**6#C[np.argmax(pred_chain)] 
    start = time.time()  
    for i in range(1):
        print(i)
        HC_train,reconst_x,af_list,bf_list,an_list,bn_list=multi_network_train (X_train , 'sigmoid',2,3,200,CC)
        W,Beta_hat,Y=ELM_train(HC_train,Y_train,n_hid,CC)
        HC_test,reconst=multi_network_test (X_test , 'sigmoid',af_list,bf_list,an_list,bn_list)
        Y_predict=ELM_test(HC_test,W,Beta_hat)
        accuracy[i]=predict_new(Y_test,Y_predict)
    final_acc= np.sum(accuracy)/1
    final_standard_div = np.sum((accuracy-final_acc)**2)/1
    stop = time.time()
#    reconst = reconst.reshape((reconst.shape[0],28,28))
#    import matplotlib.pyplot as plt
#    plt.figure(figsize=(5,5))
#    for i in range(10):
#        plt.subplot(5,10,i+1)
#        plt.xticks([])
#        plt.yticks([])
#        plt.grid('off')
#        plt.imshow(reconst[i])
#    plt.savefig('AE_with.pdf')  
    return final_acc,stop-start,final_standard_div
#%%
# result=open("result_AE_with_inv.txt","w")
# acc_test_mnist,tim,final_standard_div_mnist= AE_INV_main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
# result.write("tim_test_mnist:{}\n".format(tim))
acc_test_sat ,tim,final_standard_div_sat= AE_INV_main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
# result.write("tim_test_sat:{}\n".format(tim))
acc_test_duke ,tim,final_standard_div_duke= AE_INV_main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
# result.write("tim_test_duke:{}\n".format(tim))
acc_test_hill,tim,final_standard_div_hill= AE_INV_main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
# result.write("tim_test_hill:{}\n".format(tim))
acc_test_uspss ,tim,final_standard_div_usps= AE_INV_main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
# result.write("tim_test_usps:{}\n".format(tim))
#acc_test_face ,tim,final_standard_div_face= AE_INV_main(X_train_face,Y_train_face,X_test_face,Y_test_face)
acc_test_diabetes,tim,final_standard_div_diabetes = AE_INV_main(X_train_diabet,Y_train_diabet,X_test_diabet,Y_test_diabet)  
# acc_test_iris,tim,final_standard_div_iris = AE_INV_main(X_train_iris,Y_train_iris,X_test_iris,Y_test_iris) 
#acc_test_cifar10,tim,final_standard_div_cifar10 = AE_INV_main(X_train_cifar10,Y_train_cifar10,X_test_cifar10,Y_test_cifar10) 
acc_test_Liver,tim,final_standard_div_Liver = AE_INV_main(X_train_Liver,Y_train_Liver,X_test_Liver,Y_test_Liver)
acc_test_segment,tim,final_standard_div_segment = AE_INV_main(X_train_segment,Y_train_segment,X_test_segment,Y_test_segment)  
acc_test_wine,tim,final_standard_div_wine = AE_INV_main(X_train_wine,Y_train_wine,X_test_wine,Y_test_wine)  
print(acc_test_mnist)    