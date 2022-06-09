# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:22:25 2018

@author: Hamid
"""

import numpy as np
from  dataset import load_mnist,load_cars
import matplotlib.pyplot as plt
mnist=load_mnist()
X_train_mnist = mnist.train.images
y_train_mnist = mnist.train.labels
X_test_mnist =  mnist.test.images
y_test_mnist = mnist.test.labels 
X_test_mnist=X_test_mnist.reshape((X_test_mnist.shape[0],28,28))   
plt.figure(figsize=(10,10))
for i in range(10):
        plt.subplot(5,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(X_test_mnist[i,:,:], cmap=plt.cm.binary)
plt.savefig('mnist_test.pdf')    