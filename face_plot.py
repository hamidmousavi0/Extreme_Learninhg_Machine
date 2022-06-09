# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:31:50 2018

@author: Hamid
"""

import numpy as np
from  dataset import load_olivetti_faces
import matplotlib.pyplot as plt
face=load_olivetti_faces()
face_data = face.data
face_target = face.target
face_train = face_data[0:400,:]
face_test = np.row_stack([face_train[126,:],face_train[241,:],face_train[212,:],
                          face_train[133,:],face_train[217,:],face_train[396,:],
                          face_train[261,:],face_train[165,:],face_train[370,:],
                          face_train[301,:]])
face_test=face_test.reshape((face_test.shape[0],64,64))   
plt.figure(figsize=(15,10))
for i in range(10):
        plt.subplot(5,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(face_test[i,:,:])
plt.savefig('face_test.pdf')   