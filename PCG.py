# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:37:57 2018

@author: Hamid
"""
from BICGSTAB import BICGSTAB
from GlGMRES import GIGMRES
import numpy as np
import time
def PCG(A,B,X,tol,n,s,m,l):
    start_time= time.time()
    R = B-np.dot(A,X)
    P=R
    a=np.linalg.norm(R,'fro')
    res=a
    b=a
    tol = tol*a
    iteration=1
    resvec=[]
    resvec.append(res/a)
    while(res >=tol and iteration<500):
        if l==1 :
            Z = BICGSTAB( A,P,n,s,m ) 
        elif l == 2 :
            Z = P
        else:
            Z = GIGMRES( A,P,n,s,m ) 
        W = np.dot(A,Z)
        alpha = b**2/np.trace(np.dot(P.T,W))
        X = X + alpha *Z
        RR = R - alpha * W 
        res = np.linalg.norm(RR, 'fro')
        beta = res**2/b**2 
        P = RR + beta * P 
        R = RR 
        iteration = iteration + 1 
        resvec.append(res/a)
        b = res 
    end_time = time.time()
    finalX=X
    return np.array(resvec),end_time-start_time,iteration,res,finalX
            
    
    