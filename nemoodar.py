# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:18:02 2019

@author: Hamid
"""
import numpy as np
import matplotlib.pyplot as plt
width = 0.35 
porposed=[0.9023,0.9103,0.9204,0.9308,0.9355,0.9377,0.9402,0.9436,0.9468,0.9482]
DRELM=[0.9027,0.9044,0.9178,0.9243,0.9275,0.9288,0.9297,0.9299,0.9305,0.9316]
ind=np.arange(10)
fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, porposed, width,
                color='SkyBlue', label='porposed')
rects2 = ax.bar(ind + width/2, DRELM, width, 
                color='IndianRed', label='DRELM')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Number of block')
ax.set_title('compare accuracy ')
ax.set_xticks(ind)
ax.set_xticklabels(('1', '2', '3', '4', '5','6','7','8','9','10'))
ax.legend()
#def autolabel(rects, xpos='center'):
#    """
#    Attach a text label above each bar in *rects*, displaying its height.
#
#    *xpos* indicates which side to place the text w.r.t. the center of
#    the bar. It can be one of the following {'center', 'right', 'left'}.
#    """
#
#    xpos = xpos.lower()  # normalize the case of the parameter
#    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
#    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
#
#    for rect in rects:
#        height = rect.get_height()
#        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
#                '{}'.format(height), ha=ha[xpos], va='bottom')
#autolabel(rects1, "left")
#autolabel(rects2, "right")
plt.ylim(0.8,1)
plt.show()
#%%
import scipy.stats as ss

def plot_normal(x_range, mu=0, sigma=1, cdf=False, label="ELM",**kwargs):
    '''
    Plots the normal distribution function for a given x range
    If mu and sigma are not provided, standard normal is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    x = x_range
    if cdf:
        y = ss.norm.cdf(x, mu, sigma)
    else:
        y = ss.norm(mu,sigma).pdf(x)
        y=y/np.max(y)
    plt.ylim(0,1) 
    plt.plot(x, y,label=label,**kwargs)
    plt.legend()
x = np.linspace(-2, 2, 500)    
plot_normal(x, 0.9195, 0.056, color='red', lw=2, ls='-', alpha=0.5,label="ELM")
plot_normal(x, 0.9268, 0.15, color='blue', lw=2, ls='-', alpha=0.5,label="ML-ELM")
plot_normal(x, 0.9269, 0.082, color='green', lw=2, ls='-', alpha=0.5,label="T-ELM")
plot_normal(x, 0.9426, 0.0204, color='black', lw=2, ls='-', alpha=0.9,label="propoesd")
#%%
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.power(x, 2) # Effectively y = x**2
e = np.array([1.5, 2.6, 3.7, 4.6, 5.5])

plt.errorbar(x, y, e, linestyle='None', marker='^')

plt.show()
#%%
mean,std=(0.9195,0.9268,0.92696,0.9426),(0.0056,0.015,0.0082,0.0024)
ind = np.arange(len(mean))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind , mean, width, yerr=std,
                color='SkyBlue')
plt.ylim(0.4,1)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('mean and variance')
ax.set_xticks(ind)
ax.set_xticklabels(('ELM', 'ML-ELM', 'T-ELM', 'Proposed'))


