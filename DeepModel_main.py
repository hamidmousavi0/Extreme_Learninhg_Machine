# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:29:46 2018

@author: Hamid
"""
import time
from Predict_feed import predict_new
from ConvertToonehot import convert_to_one_hot
import dataset
from DeepModel_new import five_layer_model,three_layer_model
from preprocess_dataset import preprocess_MNIST,preprocess_sat,preprocess_face
from preprocess_dataset import preprocess_hill,preprocess_duke,preprocess_usps
#X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist=preprocess_MNIST()
#X_train_sat,Y_train_sat,X_test_sat,Y_test_sat=preprocess_sat()
#X_train_duke,Y_train_duke,X_test_duke,Y_test_duke=preprocess_duke()
#X_train_hill,Y_train_hill,X_test_hill,Y_test_hill=preprocess_hill()
X_train_usps,Y_train_usps,X_test_usps,Y_test_usps=preprocess_usps()
#X_train_face,Y_train_face,X_test_face,Y_test_face=preprocess_face()
def Deep_Main(X_train,Y_train,X_test,Y_test):
    start=time.time()
    n_x = X_train.T.shape[0]
    n_h1=256
    n_h2=10
    n_y=10
    layers_dims = (n_x, n_h1,n_h2, n_y)
    parameters = three_layer_model(X_train.T, Y_train.T, layers_dims,learning_rate =  0.001, num_iterations = 100, print_cost = True,keep_prob=0.8,beta1=0.99,beta2 = 0.999,epsilon = 1e-8,mini_batch_size = 64)
    pred_train = predict_new(X_train.T, Y_train.T, parameters)
    pred_test = predict_new(X_test.T, Y_test.T, parameters)
    stop=time.time()
    return pred_train,pred_test,stop-start
#%%
#acc_train_mnist,acc_test_mnist= Deep_Main(X_train_mnist,Y_train_mnist,X_test_mnist,Y_test_mnist)
#acc_train_sat,acc_test_sat = Deep_Main(X_train_sat,Y_train_sat,X_test_sat,Y_test_sat)
#acc_train_duke,acc_test_duke = Deep_Main(X_train_duke,Y_train_duke,X_test_duke,Y_test_duke)
#acc_train_hill,acc_test_hill= Deep_Main(X_train_hill,Y_train_hill,X_test_hill,Y_test_hill)
acc_train_usps,acc_test_uspss,tim = Deep_Main(X_train_usps,Y_train_usps,X_test_usps,Y_test_usps)
#acc_train_face,acc_test_face = elm_deep_main(X_train_face,Y_train_face,X_test_face,Y_test_face)      
    