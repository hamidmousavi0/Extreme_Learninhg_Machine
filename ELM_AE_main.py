# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:35:46 2018

@author: admin
"""
import numpy as np
from  dataset import load_mnist,load_cars,load_olivetti_faces
from basic_ELM_AE import ELM_AE
import matplotlib.pyplot as plt
import cv2
def main_ELM_AE():
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
#    cars = load_cars() 
#    gray_cars = np.zeros((526,240,360))
#    gray_cars_resize = np.zeros((526,200,200))
#    for i in range (cars.shape[0]):
#        gray_cars[i,:,:] = cv2.cvtColor(cars[i], cv2.COLOR_BGR2GRAY)
#        gray_cars_resize[i,:,:] = cv2.resize(gray_cars[i,:,:],(28,28))
#        gray_cars_resize[i,:,:]=gray_cars_resize[i,:,:]/np.max(gray_cars_resize[i,:,:])
#    plt.imshow(gray_cars_resize[0,:,:], cmap=plt.cm.binary)    
#    gray_cars_flat =   gray_cars_resize.reshape((526,784))  
#    X_train_cars = gray_cars_flat[0:400,:]
#    X_test_cars = gray_cars_flat[400:,:]
#    face=load_olivetti_faces()
#    face_data = face.data
#    face_target = face.target
#    permutation = np.loadtxt('per.txt', dtype=int)
#    shuffled_face_X = face_data[permutation,:]
#    shuffled_face_Y = face_target[permutation]
#    face_train = shuffled_face_X[0:390,:]
#    face_train_traget = shuffled_face_Y[0:390]
#    face_train_traget = face_train_traget.astype(int)
#    face_test = shuffled_face_X[390:400,:]
#    face_test_traget = shuffled_face_Y[390:400]
#    face_test_traget = face_test_traget.astype(int)
#    face=load_olivetti_faces()
#    face_data = face.data
#    face_target = face.target
#    face_train = face_data[0:400,:]
#    face_test = np.row_stack([face_train[126,:],face_train[241,:],face_train[212,:],
#                              face_train[133,:],face_train[217,:],face_train[396,:],
#                              face_train[261,:],face_train[165,:],face_train[370,:],
#                              face_train[301,:]])
    numhid=50
    C=10**6
    Train_X=X_train_zero
    beta=ELM_AE(Train_X,numhid,C)
    beta_hat=beta.reshape((beta.shape[0],28,28))    
    plt.figure(figsize=(15,10))
    for i in range(10):
        plt.subplot(5,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(beta_hat[i],cmap='gray')
    plt.savefig('ELM-AE.pdf')    
    return beta_hat
beta=main_ELM_AE()