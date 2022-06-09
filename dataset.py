# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:20:46 2018

@author: admin
"""
import bz2
import numpy as np 
import pandas as pd
from sklearn import datasets
#from PIL import Image
import os 
import cv2
#%%
#%%
def load_diabet():
    diabets=np.loadtxt("diabets.txt",delimiter=',')
    return diabets
#%%
def load_Liver():
    Liver=np.loadtxt("Liver.txt",delimiter=',')
    return Liver
#%%
def load_iris():
    iris=datasets.load_iris()
    return iris
#%%
def load_Wine():
    wine=datasets.load_wine()
    return wine
#%%
def load_segment():
    segment=np.loadtxt("segment.txt",delimiter=' ')
    return segment
#%%
def load_vowel():
    vowel=np.loadtxt("vowel.txt",delimiter=' ')
    return vowel
#%%
def load_sataleit():
    staleit_train=np.loadtxt("staleit_train.txt",delimiter=' ')
    staleit_test=np.loadtxt("staleit_test.txt",delimiter=' ')
    return staleit_train ,staleit_test  
#%%  
def load_pendigits():
    pendigits_train =np.loadtxt("pendigits_train .txt",delimiter=',')
    pendigits_test =np.loadtxt("pendigits_test .txt",delimiter=',')
    return pendigits_train ,pendigits_test   
#%%
def load_optdigits():
    optdigits_train =np.loadtxt("optdigits_train.txt",delimiter=',')
    optdigits_test =np.loadtxt("optdigits_test.txt",delimiter=',')
    return optdigits_train ,optdigits_test       
#%%
import random    
def sing_unimodal_nonli(num_data,dim):
    X_train = np.zeros((num_data,dim))
    X_test  = np.zeros((num_data,dim))
    Y_train = np.zeros((num_data,1))
    Y_test  = np.zeros((num_data,1))
    for i in range (num_data):
         Features=np.array([random.uniform(0,1) for i in range(dim)])
         output = np.sum(Features**2)
         X_train[i,:]=Features
         Y_train[i,0]=output
    miny=np.min(Y_train)
    maxy=np.max(Y_train)
    Y_train=2 * ((Y_train-miny)/(maxy-miny)) - 1  
    for i in range (num_data):
         Features=np.array([random.uniform(0,1) for i in range(dim)])
         output = np.sum(Features**2)
         X_test[i,:]=Features
         Y_test[i,0]=output
    miny=np.min(Y_test)
    maxy=np.max(Y_test)
    Y_test=2 * ((Y_test-miny)/(maxy-miny)) - 1  
    return X_train,Y_train,X_test,Y_test
#%%
import math
def complex_mmodal_nonlinear(num_data,dim):
    X_train = np.zeros((num_data,dim))
    X_test  = np.zeros((num_data,dim))
    Y_train = np.zeros((num_data,1))
    Y_test  = np.zeros((num_data,1))
    for i in range (num_data):
         Features=np.array([random.uniform(0,1) for i in range(dim)])
         temp1 = -0.2 * np.sqrt(np.sum((Features**2)/dim))
         temp2 =np.sum((np.cos(2*math.pi*Features))/dim)
         output = 20-(20*np.exp(temp1))-np.exp(temp2)+math.e
         X_train[i,:]=Features
         Y_train[i,0]=output
    miny=np.min(Y_train)
    maxy=np.max(Y_train)
    Y_train=2 * ((Y_train-miny)/(maxy-miny)) - 1  
    for i in range (num_data):
         Features=np.array([random.uniform(0,1) for i in range(dim)])
         temp1 = -0.2 * np.sqrt(np.sum((Features**2)/dim))
         temp2 =np.sum((np.cos(2*math.pi*Features))/dim)
         output = 20-20*np.exp(temp1)-np.exp(temp2)+math.e
         X_test[i,:]=Features
         Y_test[i,0]=output
    miny=np.min(Y_test)
    maxy=np.max(Y_test)
    Y_test=2 * ((Y_test-miny)/(maxy-miny)) - 1  
    return X_train,Y_train,X_test,Y_test  
#%%
def complex_mmodal_nonlinear2(num_data,dim):
    X_train = np.zeros((num_data,dim))
    X_test  = np.zeros((num_data,dim))
    Y_train = np.zeros((num_data,1))
    Y_test  = np.zeros((num_data,1))
    for i in range (num_data):
         Features=np.array([random.uniform(0,1) for i in range(dim)])
         temp1 =Features**2
         temp2 =10*np.cos(2*math.pi*Features)
         output = np.sum(temp1-temp2+10)
         X_train[i,:]=Features
         Y_train[i,0]=output
    miny=np.min(Y_train)
    maxy=np.max(Y_train)
    Y_train=2 * ((Y_train-miny)/(maxy-miny)) - 1  
    for i in range (num_data):
         Features=np.array([random.uniform(0,1) for i in range(dim)])
         temp1 =Features**2
         temp2 =10*np.cos(2*math.pi*Features)
         output = np.sum(temp1-temp2+10)
         X_test[i,:]=Features
         Y_test[i,0]=output
    miny=np.min(Y_test)
    maxy=np.max(Y_test)
    Y_test=2 * ((Y_test-miny)/(maxy-miny)) - 1  
    return X_train,Y_train,X_test,Y_test            
#%%
def load_sonar():
    sonar = np.zeros((208,60))
    sonar_target = np.zeros((1,208))    
    sonartxt=open("sonar.txt",'r').readlines()
    for i in range(len(sonar)):
        temp=(sonartxt[i].split(',')[0:60])
        temp1=[float(i) for i in temp]
        sonar[i,:]=np.array(temp1)
        if sonartxt[i].split(',')[60]=='R\n':
            sonar_target[0,i]=1
        else:
            sonar_target[0,i]=0
    return sonar,sonar_target        
#%%
def load_Abalone():
    Abalone=np.loadtxt("Abalone.txt",delimiter='\t')
    Abalone_target=open("Abalone_target.txt",'r').readlines()
    return Abalone,Abalone_target
#%%
def load_pyrim():
    pyrim=np.loadtxt("pyrim.txt",delimiter=',')
    return pyrim
#%%
def load_housing():
    housing=np.loadtxt("housing.txt",delimiter='\t')
    return housing
#%%
def load_strike():
    strike=np.loadtxt("strike.txt",delimiter='\t')
    return strike
#%%
def load_spiral():
    spiral=np.loadtxt("spiral.txt",delimiter=' ')
    return spiral
#%%
def load_hill_valley():
    hill_valey_train = np.loadtxt("hillvaley_train.txt",delimiter=',')
    hill_valey_test = np.loadtxt("hillvaley_test.txt",delimiter=',')
    return hill_valey_train,hill_valey_test
#%%
def load_balloon():
    balloon=pd.read_excel('balloon.xlsx')
    return balloon
#%%
def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist=input_data.read_data_sets('my/directory')
    return mnist
#%%
def load_sat():
    sat_train=np.loadtxt("sat_train.txt",delimiter=' ')
    sat_test=np.loadtxt("sat_test.txt",delimiter=' ')
    return sat_train,sat_test
#%% 
def load_colon():
    colon = np.zeros((2000,62))
    colon_target = np.zeros((1,62))      
    colon_read = (bz2.open("colon-cancer.bz2")).readlines()
    for i in range(len(colon_read)):
        colon_dec = colon_read[i].decode()
        colon_dec= colon_dec.split('  ')
        colon_target[0,i]=colon_dec[0]
        colon_dec_sp = colon_dec[1].split(' ')
        for j in range (len(colon_dec_sp)):
            colon_dec_sp_sp = colon_dec_sp[j].split(':')
            colon[j,i]=colon_dec_sp_sp[1]
    return colon,colon_target        
#%%  
def load_duke():
    duke = np.zeros((7129,44))
    duke_target = np.zeros((1,44))      
    duke_read = (bz2.open("duke.bz2")).readlines() 
    for i in range(len(duke_read)):  
        duke_dec = duke_read[i].decode()
        duke_dec= duke_dec.split('  ') 
        duke_target[0,i]=duke_dec[0]
        duke_dec_sp = duke_dec[1].split(' ')
        for j in range (len(duke_dec_sp)):
            duke_dec_sp_sp = duke_dec_sp[j].split(':')
            duke[j,i]=duke_dec_sp_sp[1]
    return duke,duke_target        
#%%
def load_leu():
    leu_train = np.zeros((7129,38))
    leu_train_target = np.zeros((1,38))
    leu_test = np.zeros((7129,34))
    leu_test_target = np.zeros((1,34))    
    leu_tarin_read = (bz2.open("leu.bz2")).readlines() 
    for i in range(len(leu_tarin_read)):  
        leu_dec = leu_tarin_read[i].decode()
        leu_dec_sp= leu_dec.split('  ') 
        leu_train_target[0,i]=leu_dec_sp[0]
        leu_dec_sp_sp = leu_dec_sp[1].split(' ')
        for j in range (len(leu_dec_sp_sp)):
            leu_dec_sp_sp_sp = leu_dec_sp_sp[j].split(':')
            leu_train[j,i]=leu_dec_sp_sp_sp[1]
    leu_test_read = (bz2.open("leu.t.bz2")).readlines() 
    for i in range(len(leu_test_read)):  
        leu_dec = leu_test_read[i].decode()
        leu_dec_sp= leu_dec.split('  ') 
        leu_test_target[0,i]=leu_dec_sp[0]
        leu_dec_sp_sp = leu_dec_sp[1].split(' ')
        for j in range (len(leu_dec_sp_sp)):
            leu_dec_sp_sp_sp = leu_dec_sp_sp[j].split(':')
            leu_test[j,i]=leu_dec_sp_sp_sp[1] 
    return leu_train,leu_train_target,leu_test,leu_test_target        
#%%
def load_usps():
    usps_train = np.zeros((256,7291))
    usps_train_target = np.zeros((1,7291))
    usps_test = np.zeros((256,2007))
    usps_test_target = np.zeros((1,2007))    
    usps_tarin_read = (bz2.open("usps.bz2")).readlines() 
    for i in range(len(usps_tarin_read)):  
        usps_dec = usps_tarin_read[i].decode()
        usps_dec_sp= usps_dec.split(' ') 
        usps_train_target[0,i]=usps_dec_sp[0]
        for j in range (1,257):
            usps_dec_sp_sp= usps_dec_sp[j].split(':')
            usps_train[j-1,i]=usps_dec_sp_sp[1]
       
    usps_test_read = (bz2.open("usps.t.bz2")).readlines() 
    for i in range(len(usps_test_read)):  
        usps_dec = usps_test_read[i].decode()
        usps_dec_sp= usps_dec.split(' ') 
        usps_test_target[0,i]=usps_dec_sp[0]
        for j in range (1,257):
            usps_dec_sp_sp= usps_dec_sp[j].split(':')
            usps_test[j-1,i]=usps_dec_sp_sp[1]    
    return usps_train,usps_train_target,usps_test,usps_test_target        
#%%
def load_acoustic():
    acoustic_train = np.zeros((50,78823))
    acoustic_train_target = np.zeros((1,78823))
    acoustic_test = np.zeros((50,19705))
    acoustic_test_target = np.zeros((1,19705))    
    acoustic_tarin_read = (bz2.open("acoustic_scale.bz2")).readlines() 
    for i in range(len(acoustic_tarin_read)):  
        acoustic_dec = acoustic_tarin_read[i].decode()
        acoustic_dec_sp= acoustic_dec.split(' ') 
        acoustic_train_target[0,i]=acoustic_dec_sp[0]
        for j in range (1,51):
            acoustic_dec_sp_sp= acoustic_dec_sp[j].split(':')
            acoustic_train[j-1,i]=acoustic_dec_sp_sp[1]       
    acoustic_test_read = (bz2.open("acoustic_scale.t.bz2")).readlines() 
    for i in range(len(acoustic_test_read)):  
        acoustic_dec = acoustic_test_read[i].decode()
        acoustic_dec_sp= acoustic_dec.split(' ') 
        acoustic_test_target[0,i]=acoustic_dec_sp[0]
        for j in range (1,51):
            acoustic_dec_sp_sp= acoustic_dec_sp[j].split(':')
            acoustic_test[j-1,i]=acoustic_dec_sp_sp[1]  
    return acoustic_train,acoustic_train_target,acoustic_test,acoustic_test_target        
       
#%%  
def load_cfar10():
      from keras.datasets import cifar10
      (train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
      return train_features,train_labels,test_features,test_labels
#%%
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img=cv2.resize(img, (300, 250)) 
        if img is not None:
            images.append(img)
    return images
#%%
#def load_images_from_folder_gray(folder):
#    images_path = [ os.path.join(folder, item)  for item in  os.listdir(folder) ]
#    image_data = []
#    image_labels = []
#    
#    for i,im_path in enumerate(images_path):
#        im=Image.open(im_path).convert('L')
#        image_data.append(np.array(im, dtype='uint8'))
#        label = int(os.path.split(im_path)[1].split(".")[0].replace("subject", ""))-1
#        image_labels.append(label)
#    return image_data ,image_labels   
#%%
def load_scene15():
    bedroom=np.array(load_images_from_folder('scene_categories/bedroom_del'))
#    CALsuburb=np.array(load_images_from_folder('scene_categories/CAKsubrub_del'))
#    industrial=np.array(load_images_from_folder('scene_categories/industrial_del'))
#    kitchen=np.array(load_images_from_folder('scene_categories/kitchen_del'))
#    livingroom=np.array(load_images_from_folder('scene_categories/livingroom_del'))
#    MITcoast=np.array(load_images_from_folder('scene_categories/MITcoast_del'))
#    MITforest=np.array(load_images_from_folder('scene_categories/MITforest_del'))
#    MIThighway=np.array(load_images_from_folder('scene_categories/MIThighway_del'))
#    MITinsidecity=np.array(load_images_from_folder('scene_categories/MITinsidecity_del'))
#    MITmountain=np.array(load_images_from_folder('scene_categories/MITmountain_del'))
#    MITopencountry=np.array(load_images_from_folder('scene_categories/MITopencountry_del'))
#    MITstreet=np.array(load_images_from_folder('scene_categories/MITstreet_del'))
#    MITtallbuilding=np.array(load_images_from_folder('scene_categories/MiTtallbuilding_del'))
#    PARoffice=np.array(load_images_from_folder('scene_categories/PARoffice_del'))
#    store=np.array(load_images_from_folder('scene_categories/store_del'))
    return bedroom
#,CALsuburb,industrial,kitchen,livingroom\
#    ,MITcoast,MITforest,MIThighway,MITinsidecity,MITmountain\
#    ,MITopencountry,MITstreet,MITtallbuilding,PARoffice,store
#%%
def load_cars():
    cars =np.array(load_images_from_folder('cars_brad'))
    return cars
#%%
#def load_yalefaces():
#    yalefaces_data,yalefaces_label =np.array(load_images_from_folder_gray('yalefaces'))
#    return yalefaces_data,yalefaces_label   
#%%
def load_leaves():
    leaves =np.array(load_images_from_folder('leaves'))
    return leaves    
#%%
def Caltech101():
    accordion=np.array(load_images_from_folder('Caltech101/accordion'))
    airplanes=np.array(load_images_from_folder('Caltech101/airplanes'))
    anchor=np.array(load_images_from_folder('Caltech101/anchor'))
    ant=np.array(load_images_from_folder('Caltech101/ant'))
    BACKGROUND_Google=np.array(load_images_from_folder('Caltech101/BACKGROUND_Google'))
    barrel=np.array(load_images_from_folder('Caltech101/barrel'))
    bass=np.array(load_images_from_folder('Caltech101/bass'))
    beaver=np.array(load_images_from_folder('Caltech101/beaver'))
    binocular=np.array(load_images_from_folder('Caltech101/binocular'))
    bonsai=np.array(load_images_from_folder('Caltech101/bonsai'))
    brain=np.array(load_images_from_folder('Caltech101/brain'))
    brontosaurus=np.array(load_images_from_folder('Caltech101/brontosaurus'))
    buddha=np.array(load_images_from_folder('Caltech101/buddha'))
    butterfly=np.array(load_images_from_folder('Caltech101/butterfly'))
    camera=np.array(load_images_from_folder('Caltech101/camera'))
    cannon=np.array(load_images_from_folder('Caltech101/cannon'))
    car_side=np.array(load_images_from_folder('Caltech101/car_side'))
    ceiling_fan=np.array(load_images_from_folder('Caltech101/ceiling_fan'))
    cellphone=np.array(load_images_from_folder('Caltech101/cellphone'))
    chair=np.array(load_images_from_folder('Caltech101/chair'))
    chandelier=np.array(load_images_from_folder('Caltech101/chandelier'))
    cougar_body=np.array(load_images_from_folder('Caltech101/cougar_body'))
    cougar_face=np.array(load_images_from_folder('Caltech101/cougar_face'))
    crab=np.array(load_images_from_folder('Caltech101/crab'))
    crayfish=np.array(load_images_from_folder('Caltech101/crayfish'))
    crocodile=np.array(load_images_from_folder('Caltech101/crocodile'))
    crocodile_head=np.array(load_images_from_folder('Caltech101/crocodile_head'))
    cup=np.array(load_images_from_folder('Caltech101/cup'))
    dalmatian=np.array(load_images_from_folder('Caltech101/dalmatian'))
    dollar_bill=np.array(load_images_from_folder('Caltech101/dollar_bill'))
    dolphin=np.array(load_images_from_folder('Caltech101/dolphin'))
    dragonfly=np.array(load_images_from_folder('Caltech101/dragonfly'))
    electric_guitar=np.array(load_images_from_folder('Caltech101/electric_guitar'))
    elephant=np.array(load_images_from_folder('Caltech101/elephant'))
    emu=np.array(load_images_from_folder('Caltech101/emu'))
    euphonium=np.array(load_images_from_folder('Caltech101/euphonium'))
    ewer=np.array(load_images_from_folder('Caltech101/ewer'))
    Faces_easy=np.array(load_images_from_folder('Caltech101/Faces_easy'))
    ferry=np.array(load_images_from_folder('Caltech101/ferry'))
    flamingo=np.array(load_images_from_folder('Caltech101/flamingo'))
    flamingo_head=np.array(load_images_from_folder('Caltech101/flamingo_head'))
    garfield=np.array(load_images_from_folder('Caltech101/garfield'))
    gerenuk=np.array(load_images_from_folder('Caltech101/gerenuk'))
    gramophone=np.array(load_images_from_folder('Caltech101/gramophone'))
    grand_piano=np.array(load_images_from_folder('Caltech101/grand_piano'))
    hawksbill=np.array(load_images_from_folder('Caltech101/hawksbill'))
    headphone=np.array(load_images_from_folder('Caltech101/headphone'))
    hedgehog=np.array(load_images_from_folder('Caltech101/hedgehog'))
    helicopter=np.array(load_images_from_folder('Caltech101/helicopter'))
    ibis=np.array(load_images_from_folder('Caltech101/ibis'))
    inline_skate=np.array(load_images_from_folder('Caltech101/inline_skate'))
    joshua_tree=np.array(load_images_from_folder('Caltech101/joshua_tree'))
    kangaroo=np.array(load_images_from_folder('Caltech101/kangaroo'))
    ketch=np.array(load_images_from_folder('Caltech101/ketch'))
    lamp=np.array(load_images_from_folder('Caltech101/lamp'))
    laptop=np.array(load_images_from_folder('Caltech101/laptop'))
    Leopards=np.array(load_images_from_folder('Caltech101/Leopards'))
    llama=np.array(load_images_from_folder('Caltech101/llama'))
    lobster=np.array(load_images_from_folder('Caltech101/lobster'))
    lotus=np.array(load_images_from_folder('Caltech101/lotus'))
    mandolin=np.array(load_images_from_folder('Caltech101/mandolin'))
    mayfly=np.array(load_images_from_folder('Caltech101/mayfly'))
    menorah=np.array(load_images_from_folder('Caltech101/menorah'))
    metronome=np.array(load_images_from_folder('Caltech101/metronome'))
    minaret=np.array(load_images_from_folder('Caltech101/minaret'))
    Motorbikes=np.array(load_images_from_folder('Caltech101/Motorbikes'))
    nautilus=np.array(load_images_from_folder('Caltech101/nautilus'))
    octopus=np.array(load_images_from_folder('Caltech101/octopus'))
    okapi=np.array(load_images_from_folder('Caltech101/okapi'))
    pagoda=np.array(load_images_from_folder('Caltech101/pagoda'))
    panda=np.array(load_images_from_folder('Caltech101/panda'))
    pigeon=np.array(load_images_from_folder('Caltech101/pigeon'))
    pizza=np.array(load_images_from_folder('Caltech101/pizza'))
    platypus=np.array(load_images_from_folder('Caltech101/platypus'))
    pyramid=np.array(load_images_from_folder('Caltech101/pyramid'))
    revolver=np.array(load_images_from_folder('Caltech101/revolver'))
    rhino=np.array(load_images_from_folder('Caltech101/rhino'))
    rooster=np.array(load_images_from_folder('Caltech101/rooster'))
    saxophone=np.array(load_images_from_folder('Caltech101/saxophone'))
    schooner=np.array(load_images_from_folder('Caltech101/schooner'))
    scissors=np.array(load_images_from_folder('Caltech101/scissors'))
    scorpion=np.array(load_images_from_folder('Caltech101/scorpion'))
    sea_horse=np.array(load_images_from_folder('Caltech101/sea_horse'))
    snoopy=np.array(load_images_from_folder('Caltech101/snoopy'))
    soccer_ball=np.array(load_images_from_folder('Caltech101/soccer_ball'))
    stapler=np.array(load_images_from_folder('Caltech101/stapler'))
    starfish=np.array(load_images_from_folder('Caltech101/starfish'))
    stegosaurus=np.array(load_images_from_folder('Caltech101/stegosaurus'))
    stop_sign=np.array(load_images_from_folder('Caltech101/stop_sign'))
    strawberry=np.array(load_images_from_folder('Caltech101/strawberry'))
    sunflower=np.array(load_images_from_folder('Caltech101/sunflower'))
    tick=np.array(load_images_from_folder('Caltech101/tick'))
    trilobite=np.array(load_images_from_folder('Caltech101/trilobite'))
    umbrella=np.array(load_images_from_folder('Caltech101/umbrella'))
    watch=np.array(load_images_from_folder('Caltech101/watch'))
    water_lilly=np.array(load_images_from_folder('Caltech101/water_lilly'))
    wheelchair=np.array(load_images_from_folder('Caltech101/wheelchair'))
    wild_cat=np.array(load_images_from_folder('Caltech101/wild_cat'))
    windsor_chair=np.array(load_images_from_folder('Caltech101/windsor_chair'))
    wrench=np.array(load_images_from_folder('Caltech101/wrench'))
    yin_yang=np.array(load_images_from_folder('Caltech101/yin_yang'))
    return accordion,airplanes,anchor,ant,\
    BACKGROUND_Google,barrel,bass,beaver,binocular,\
    bonsai,brain,brontosaurus,buddha,butterfly,camera,\
    cannon,car_side,ceiling_fan,cellphone,chair,chandelier,\
    cougar_body,cougar_face,crab,crayfish,crocodile,crocodile_head,\
    cup,dalmatian,dollar_bill,dolphin,dragonfly,electric_guitar,\
    elephant,emu,euphonium,ewer,Faces_easy,ferry,flamingo,\
    flamingo_head,garfield,gerenuk,gramophone,grand_piano,\
    hawksbill,headphone,hedgehog,helicopter,ibis,inline_skate,\
    joshua_tree,kangaroo,ketch,lamp,laptop,llama,lobster,lotus,\
    mandolin,mayfly,menorah,metronome,minaret,Motorbikes,nautilus,\
    octopus,okapi,pagoda,panda,pigeon,pizza,platypus,pyramid,revolver,\
    rhino,rooster,saxophone,schooner,scissors,scorpion,sea_horse,snoopy,\
    soccer_ball,stapler,starfish,stegosaurus,stop_sign,strawberry,sunflower,\
    tick,trilobite,umbrella,watch,water_lilly,wheelchair,wild_cat,\
    windsor_chair,wrench,yin_yang
#%%
from sklearn import datasets    
def load_olivetti_faces():
    face= datasets.fetch_olivetti_faces()
    return face
#%%