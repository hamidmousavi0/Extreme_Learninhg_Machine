# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:39:22 2018

@author: Hamid
"""
import os 
import cv2
import matplotlib.pyplot as plt
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        img=cv2.resize(img,(64, 64)) 
        if img is not None:
            images.append(img)
    return images
#%%
pic =  load_images_from_folder("image_AE_re")
plt.figure(figsize=(15,10))
for i in range(10):
        plt.subplot(5,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(pic[i])
plt.savefig('AER_test.pdf')     
