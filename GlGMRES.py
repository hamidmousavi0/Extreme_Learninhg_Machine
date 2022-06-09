# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:38:25 2018

@author: Hamid
"""

import numpy as np 
def GIGMRES(A,B,n,s,m):
    X=np.zeros((n,s))
    R = B-np.dot(A,X)
    beta= np.linalg.norm(R,'fro')
    V=np.zeros((R.shape[0],s*(m+1)))
    H=np.zeros((m+1,m))
    V[:,0:s]=R/beta
    for j in range (m):
        W = np.dot(A,V[:,(j)*s:(j+1)*s])
        for i in range (j):
            H[i,j]=np.trace(np.dot(W.T,V[:,(i)*s:(i+1)*s]))
            W = W -np.dot(H[i,j],V[:,(i)*s:(i+1)*s])
        H[j+1,j] = np.linalg.norm(W,'fro') 
        if H[j+1,j]<10e-12:
            m = j 
            break 
        V[:,(j+1)*s:(j+2)*s] = W/H[j+1,j]
    y= np.linalg.lstsq(H,beta * np.eye(m+1,1), rcond=-1)[0]
    X = X + np.dot(V[:,0:m * s] , np.kron(y,np.eye(s)))
    return X
            