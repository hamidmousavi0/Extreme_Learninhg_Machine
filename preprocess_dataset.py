# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 09:55:16 2018

@author: Hamid
"""
from  dataset import load_mnist
from dataset import load_sat
from dataset import load_duke,load_cfar10,load_scene15
from dataset import load_hill_valley,load_olivetti_faces
from dataset import load_usps,load_cars,load_leaves
from  dataset import complex_mmodal_nonlinear2
from dataset import load_optdigits,load_segment,load_vowel
from dataset import load_pendigits,load_sataleit
from dataset import sing_unimodal_nonli,complex_mmodal_nonlinear
from predict import convert_to_one_hot
from dataset import load_diabet,load_Liver,load_iris,load_Wine

import numpy as np
#%%
def preprocess_diabet():
    diabet=load_diabet()
    diabet_data=diabet[:,0:8]
    diabet_target=diabet[:,8]
    mean_of_column=np.mean(diabet_data,axis=0)
    var_of_column=np.std(diabet_data,axis=0)
    for i in range(diabet_data.shape[1]):
        diabet_data[:,i]=(diabet_data[:,i]-mean_of_column[i])/var_of_column[i]
    min_of_column=np.min(diabet_data,axis=0)
    max_of_column=np.max(diabet_data,axis=0)
    for i in range(diabet_data.shape[1]):
        diabet_data[:,i]=(diabet_data[:,i]-min_of_column[i])/(max_of_column[i]-min_of_column[i])
    diabet_train=diabet_data[0:512,:]
    diabet_train_target=diabet_target[0:512]    
    diabet_test=diabet_data[512:,:]
    diabet_test_target=diabet_target[512:]   
    diabet_train_target=diabet_train_target.astype(int)
    diabet_train_target=convert_to_one_hot(diabet_train_target,2).T
    diabet_test_target=diabet_test_target.astype(int)
    diabet_test_target=convert_to_one_hot(diabet_test_target,2).T
    return diabet_train,diabet_train_target,diabet_test,diabet_test_target 
#%%
def preprocess_Liver():
    Liver=np.loadtxt("Liver.txt",delimiter=',')  
    Liver_data=Liver[:,0:6]
    Liver_target=Liver[:,6]
    Liver_target[np.where(Liver_target==1)]=0
    Liver_target[np.where(Liver_target==2)]=1
    mean_of_column=np.mean(Liver_data,axis=0)
    var_of_column=np.std(Liver_data,axis=0)
    for i in range(Liver_data.shape[1]):
        Liver_data[:,i]=(Liver_data[:,i]-mean_of_column[i])/var_of_column[i]
    min_of_column=np.min(Liver_data,axis=0)
    max_of_column=np.max(Liver_data,axis=0)
    for i in range(Liver_data.shape[1]):
        Liver_data[:,i]=(Liver_data[:,i]-min_of_column[i])/(max_of_column[i]-min_of_column[i])
    Liver_train=Liver_data[0:230,:]
    Liver_train_target=Liver_target[0:230]    
    Liver_test=Liver_data[230:,:]
    Liver_test_target=Liver_target[230:]   
    Liver_train_target=Liver_train_target.astype(int)
    Liver_train_target=convert_to_one_hot(Liver_train_target,2).T
    Liver_test_target=Liver_test_target.astype(int)
    Liver_test_target=convert_to_one_hot(Liver_test_target,2).T
    return Liver_train,Liver_train_target,Liver_test,Liver_test_target
#%%
def preprocess_iris2():
    iris=load_iris()   
    iris_data=iris.data
    iris_target=iris.target
    max_of_column=np.max(iris_data,axis=0)
    for i in range(iris_data.shape[1]):
        iris_data[:,i]=(iris_data[:,i]/(max_of_column[i]))
    max_data=np.where(iris_data==1)    
    permutation = list(np.random.permutation(iris_data.shape[0]))
    shuffled_iris_X = iris_data[permutation,:]
    shuffled_iris_Y = iris_target[permutation]
    iris_train = shuffled_iris_X[0:100,:]
    iris_train_target = shuffled_iris_Y[0:100]
    iris_test = shuffled_iris_X[100:,:]
    iris_test_target = shuffled_iris_Y[100:]
    iris_train_target=iris_train_target.astype(int)
    iris_train_target=convert_to_one_hot(iris_train_target,3).T
    iris_test_target=iris_test_target.astype(int)
    iris_test_target=convert_to_one_hot(iris_test_target,3).T
    return iris_train,iris_train_target,iris_test,iris_test_target
def preprocess_iris():
    iris=load_iris()   
    iris_data=iris.data
    iris_target=iris.target
    mean_of_column=np.mean(iris_data,axis=0)
    var_of_column=np.std(iris_data,axis=0)
    for i in range(iris_data.shape[1]):
        iris_data[:,i]=(iris_data[:,i]-mean_of_column[i])/var_of_column[i]
    min_of_column=np.min(iris_data,axis=0)
    max_of_column=np.max(iris_data,axis=0)
    for i in range(iris_data.shape[1]):
        iris_data[:,i]=(iris_data[:,i]-min_of_column[i])/(max_of_column[i]-min_of_column[i])
    permutation = list(np.random.permutation(iris_data.shape[0]))
    shuffled_iris_X = iris_data[permutation,:]
    shuffled_iris_Y = iris_target[permutation]
    iris_train = shuffled_iris_X[0:100,:]
    iris_train_target = shuffled_iris_Y[0:100]
    iris_test = shuffled_iris_X[100:,:]
    iris_test_target = shuffled_iris_Y[100:]
    iris_train_target=iris_train_target.astype(int)
    iris_train_target=convert_to_one_hot(iris_train_target,3).T
    iris_test_target=iris_test_target.astype(int)
    iris_test_target=convert_to_one_hot(iris_test_target,3).T
    return iris_train,iris_train_target,iris_test,iris_test_target
#%%
def preprocess_wine():
    wine=load_Wine()
    wine_data=wine.data
    wine_target=wine.target
    mean_of_column=np.mean(wine_data,axis=0)
    var_of_column=np.std(wine_data,axis=0)
    for i in range(wine_data.shape[1]):
        wine_data[:,i]=(wine_data[:,i]-mean_of_column[i])/var_of_column[i]
    min_of_column=np.min(wine_data,axis=0)
    max_of_column=np.max(wine_data,axis=0)
    for i in range(wine_data.shape[1]):
        wine_data[:,i]=(wine_data[:,i]-min_of_column[i])/(max_of_column[i]-min_of_column[i])
    permutation = list(np.random.permutation(wine_data.shape[0]))
    shuffled_wine_X = wine_data[permutation,:]
    shuffled_wine_Y = wine_target[permutation]
    wine_train = shuffled_wine_X[0:119,:]
    wine_train_target = shuffled_wine_Y[0:119]
    wine_test = shuffled_wine_X[119:,:]
    wine_test_target = shuffled_wine_Y[119:]
    wine_train_target=wine_train_target.astype(int)
    wine_train_target=convert_to_one_hot(wine_train_target,3).T
    wine_test_target=wine_test_target.astype(int)
    wine_test_target=convert_to_one_hot(wine_test_target,3).T
    return wine_train, wine_train_target, wine_test,wine_test_target  
#%%
def preprocess_segment():
    segment=load_segment()
    segment_data=segment[:,0:19]
    segment_target=segment[:,19]
    segment_target[np.where(segment_target==1)]=0
    segment_target[np.where(segment_target==2)]=1
    segment_target[np.where(segment_target==3)]=2
    segment_target[np.where(segment_target==4)]=3
    segment_target[np.where(segment_target==5)]=4
    segment_target[np.where(segment_target==6)]=5
    segment_target[np.where(segment_target==7)]=6
    mean_of_column=np.mean(segment_data,axis=0)
    var_of_column=np.std(segment_data,axis=0)
    for i in range(segment_data.shape[1]):
        if i!=2 :
            segment_data[:,i]=(segment_data[:,i]-mean_of_column[i])/var_of_column[i]
    min_of_column=np.min(segment_data,axis=0)
    max_of_column=np.max(segment_data,axis=0)
    for i in range(segment_data.shape[1]):
        if i!=2 :
           segment_data[:,i]=(segment_data[:,i]-min_of_column[i])/(max_of_column[i]-min_of_column[i])
    segment_data[:,2]=0.9
    segment_train=segment_data[0:1540,:]
    segment_train_target=segment_target[0:1540]    
    segment_test=segment_data[1540:,:]
    segment_test_target=segment_target[1540:]   
    segment_train_target=segment_train_target.astype(int)
    segment_train_target=convert_to_one_hot(segment_train_target,7).T
    segment_test_target=segment_test_target.astype(int)
    segment_test_target=convert_to_one_hot(segment_test_target,7).T
    return segment_train, segment_train_target, segment_test, segment_test_target 
#%%    
def preprocess_MNIST():
    mnist = load_mnist()
    X_train_mnist = mnist.train.images
    y_train_mnist = mnist.train.labels
    X_test_mnist =  mnist.test.images
    y_test_mnist = mnist.test.labels  
    y_train_mnist=convert_to_one_hot(y_train_mnist,10).T
    y_test_mnist=convert_to_one_hot(y_test_mnist,10).T 
    return X_train_mnist,y_train_mnist,X_test_mnist,y_test_mnist
def preprocess_sat():
    sat_tr,sat_te = load_sat()
    sat_train = sat_tr[:,0:36]
    sat_train_target = sat_tr[:,36]
    sat_train_target[np.where(sat_train_target==1)]=0
    sat_train_target[np.where(sat_train_target==2)]=1
    sat_train_target[np.where(sat_train_target==3)]=2
    sat_train_target[np.where(sat_train_target==4)]=3
    sat_train_target[np.where(sat_train_target==5)]=4
    sat_train_target[np.where(sat_train_target==7)]=5
    sat_train_target = sat_train_target.astype(int)
    sat_test = sat_te[:,0:36]
    sat_test_target = sat_te[:,36]
    sat_test_target[np.where(sat_test_target==1)]=0
    sat_test_target[np.where(sat_test_target==2)]=1
    sat_test_target[np.where(sat_test_target==3)]=2
    sat_test_target[np.where(sat_test_target==4)]=3
    sat_test_target[np.where(sat_test_target==5)]=4
    sat_test_target[np.where(sat_test_target==7)]=5
    sat_test_target = sat_test_target.astype(int)
    sat_train_target=convert_to_one_hot(sat_train_target,6).T
    sat_test_target=convert_to_one_hot(sat_test_target,6).T 
    sat_data=np.row_stack([sat_train,sat_test])
    mean_of_column=np.mean(sat_data,axis=0)
    var_of_column=np.std(sat_data,axis=0)
    for i in range(sat_data.shape[1]):
        sat_data[:,i]=(sat_data[:,i]-mean_of_column[i])/var_of_column[i]
    min_of_column=np.min(sat_data,axis=0)
    max_of_column=np.max(sat_data,axis=0)
    for i in range(sat_data.shape[1]):
        sat_data[:,i]=(sat_data[:,i]-min_of_column[i])/(max_of_column[i]-min_of_column[i])
    sat_train=sat_data[0:4435,:]
    sat_test=sat_data[4435:,:]
    return sat_train,sat_train_target,sat_test,sat_test_target
def preprocess_duke():
    duke,duke_target=load_duke()
    permutation = list(np.random.permutation(duke.shape[1]))
    shuffled_duke_X = duke[:, permutation]
    shuffled_duke_Y = duke_target[:, permutation].reshape((duke_target.shape[0],duke.shape[1]))
    duke_train = shuffled_duke_X[:,0:22].T
    duke_train_traget = shuffled_duke_Y[:,0:22].T
    duke_train_traget[np.where(duke_train_traget==-1),:]=0
    duke_train_traget = duke_train_traget.astype(int)
    duke_test = shuffled_duke_X[:,22:44].T
    duke_test_traget = shuffled_duke_Y[:,22:44].T
    duke_test_traget[np.where(duke_train_traget==-1),:]=0
    duke_test_traget = duke_test_traget.astype(int)
    duke_train_traget=convert_to_one_hot(duke_train_traget,2).T
    duke_test_traget=convert_to_one_hot(duke_test_traget,2).T 
    duke_train=duke_train
    duke_test=duke_test
    return duke_train,duke_train_traget,duke_test,duke_test_traget
def preprocess_hill():
    hill_train,hill_test=load_hill_valley()
    hill_valey_train = hill_train[:,0:100] 
    hill_valey_train_target = hill_train[:,100]
    hill_valey_train_target = hill_valey_train_target.astype(int)
    hill_valey_test = hill_test[:,0:100] 
    hill_valey_test_target = hill_test[:,100]
    hill_valey_test_target = hill_valey_test_target.astype(int)
    hill_valey_train_target=convert_to_one_hot(hill_valey_train_target,2).T
    hill_valey_test_target=convert_to_one_hot(hill_valey_test_target,2).T 
    hill_data=np.row_stack([hill_valey_train,hill_valey_test])
    mean_of_column=np.mean(hill_data,axis=0)
    var_of_column=np.std(hill_data,axis=0)
    for i in range(hill_data.shape[1]):
        hill_data[:,i]=(hill_data[:,i]-mean_of_column[i])/var_of_column[i]
    min_of_column=np.min(hill_data,axis=0)
    max_of_column=np.max(hill_data,axis=0)
    for i in range(hill_data.shape[1]):
        hill_data[:,i]=(hill_data[:,i]-min_of_column[i])/(max_of_column[i]-min_of_column[i])
    hill_valey_train=hill_data[0:606,:]
    hill_valey_test=hill_data[606:,:]
    return hill_valey_train,hill_valey_train_target,hill_valey_test,hill_valey_test_target
def preprocess_usps():
    usps_train,usps_train_target,usps_test,usps_test_target = load_usps()
    usps_train_target[np.where(usps_train_target==1)]=0
    usps_train_target[np.where(usps_train_target==2)]=1
    usps_train_target[np.where(usps_train_target==3)]=2
    usps_train_target[np.where(usps_train_target==4)]=3
    usps_train_target[np.where(usps_train_target==5)]=4
    usps_train_target[np.where(usps_train_target==6)]=5
    usps_train_target[np.where(usps_train_target==7)]=6
    usps_train_target[np.where(usps_train_target==8)]=7
    usps_train_target[np.where(usps_train_target==9)]=8
    usps_train_target[np.where(usps_train_target==10)]=9
    usps_train_target = usps_train_target.astype(int)
    usps_test_target[np.where(usps_test_target==1)]=0
    usps_test_target[np.where(usps_test_target==2)]=1
    usps_test_target[np.where(usps_test_target==3)]=2
    usps_test_target[np.where(usps_test_target==4)]=3
    usps_test_target[np.where(usps_test_target==5)]=4
    usps_test_target[np.where(usps_test_target==6)]=5
    usps_test_target[np.where(usps_test_target==7)]=6
    usps_test_target[np.where(usps_test_target==8)]=7
    usps_test_target[np.where(usps_test_target==9)]=8
    usps_test_target[np.where(usps_test_target==10)]=9
    usps_test_target = usps_test_target.astype(int)
    usps_train_target=convert_to_one_hot(usps_train_target,10).T
    usps_test_target=convert_to_one_hot(usps_test_target,10).T
    usps_train=usps_train.T
    usps_test=usps_test.T
    return usps_train,usps_train_target,usps_test,usps_test_target
def preprocess_face():
    face=load_olivetti_faces()
    face_data = face.data
    face_target = face.target
    permutation = list(np.random.permutation(face_data.shape[0]))
    shuffled_face_X = face_data[permutation,:]
    shuffled_face_Y = face_target[permutation]
    face_train = shuffled_face_X[0:350,:]
    face_train_traget = shuffled_face_Y[0:350]
    face_train_traget = face_train_traget.astype(int)
    face_test = shuffled_face_X[350:400,:]
    face_test_traget = shuffled_face_Y[350:400]
    face_test_traget = face_test_traget.astype(int)
    face_train_traget=convert_to_one_hot(face_train_traget,40).T
    face_test_traget=convert_to_one_hot(face_test_traget,40).T 
    face_train=face_train
    face_test=face_test
    return face_train,face_train_traget,face_test,face_test_traget
#%%
import cv2
def sift_des (data):
    sift_data=np.empty((0,128), np.float32)
    for i in range(len(data)):
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        sift=cv2.xfeatures2d.SIFT_create()
        detector = sift.detect(gray, None)
        kpts, des = sift.compute(gray, detector)
        sift_data=np.vstack([sift_data,des])
    return sift_data
#%%    
def preprocess_scene15():
    bedroom=load_scene15()
    for i in range(len(bedroom)):
        np.save('scene_categories/bedroom_del_sift/{}.npy'.format(i),sift_des(bedroom[i]))
        #%%
def preprocess_cifar10():
    train_features,train_labels,test_features,test_labels=load_cfar10()
    cifar_train=train_features.reshape((train_features.shape[0],train_features.shape[1]*train_features.shape[2]*train_features.shape[3]))
    cifar_train=cifar_train/255.0
    cifar_test=test_features.reshape((test_features.shape[0],test_features.shape[1]*test_features.shape[2]*test_features.shape[3]))
    train_labels=convert_to_one_hot(train_labels,10).T
    test_labels=convert_to_one_hot(test_labels,10).T 
    return cifar_train,train_labels,cifar_test,test_labels
          