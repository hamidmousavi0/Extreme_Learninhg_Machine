# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:38:35 2018

@author: Hamid
"""
import numpy as np 
def BICGSTAB(A,B,n,s,m):
    X = np.zeros((n,s))
    R= B- np.dot(A,X)
    Rsatr=R
    a=Rsatr
    P=R
    for j in range(m):
        W= np.dot(A,P)
        alpha = np.trace(np.dot(R.T,Rsatr))/np.trace(np.dot(W.T,Rsatr))
        S = R-(alpha * W)
        Z = np.dot(A,S)
        omega = np.trace(np.dot(Z.T,S))/np.linalg.norm(Z,'fro')**2
        X=X+alpha*P+omega*S
        RR = S - omega * Z 
        beta = np.trace(np.dot(a.T , RR))/np.trace(np.dot(a.T , R)) * (alpha/omega)
        P = RR + beta * (P - omega * W)
        R = RR 
    return X    