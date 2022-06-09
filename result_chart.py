# -*- coding: UTF-8 -*-
"""
Created on Sun Oct  7 12:00:40 2018

@author: Hamid
"""

import matplotlib.pyplot as plt
import numpy as np
X_g  = [1,2,3,4,5,6,7,8,9,10]
X = np.linspace(0, 10,10)
ELM = [0.1135,0.1493,0.2404,0.3388,0.2732,0.3299,0.4048,0.4342,0.4108,0.4626]
ML_ELM = [0.133,0.218,0.2378,0.363,0.3789,0.4354,0.496,0.5367,0.6473,0.7117]
T_ELM=[0.1526,0.2044,0.2559,0.3506,0.3129,0.4151,0.4338,0.4558,0.488,0.4970]
ELM_W_INV=[0.212,0.2126,0.3429,0.5129,0.607,0.657,0.6988,0.7304,0.788,0.8626]
OURS=[0.179,0.311,0.413,0.541,0.708,0.695,0.747,0.825,0.863,0.914]
plt.figure(figsize=(5,5))
plt.plot(X,ELM, '-or', label='ELM')
plt.plot(X,ML_ELM, '-ob', label='ML_ELM')
plt.plot(X,T_ELM, '-og', label='T_ELM')
plt.plot(X,ELM_W_INV, '-oy', label='ELM_W_INV')
plt.plot(X,OURS, '-ok', label='OURS')
legend =plt.legend(loc='upper left', shadow=True, fontsize='x-small')
legend.get_frame().set_facecolor('C0')
plt.xlabel("dimension reduction")
plt.ylabel("Accuracy")
plt.savefig('result.pdf') 
