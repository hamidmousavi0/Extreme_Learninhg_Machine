# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:01:18 2018

@author: Hamid
"""
import matplotlib.pyplot as plt
import numpy as np
from PCG import PCG
def main(H,T):
    m=10
    tol = 10**-11
    AA=H
    BB=T
    A = np.dot(H.T,H)
    B = np.dot(H.T,T)
    n= A.shape[0]
    s= B.shape[1]
    X=np.zeros((n,s))
#    print('FCG with Bicgstab preconditioned')
#    resvec,time,iteration,res,finalX1= PCG( A,B,X,tol,n,s,m,1)
#    print('The number of restarts for the FSGl-CMRH (SGl-CMRH) is ',iteration)
#    print('The total CPU-time in seconds for the FSGl-CMRH (SGl-CMRH) is ',time)
#    print('The Frobenius norm of the residual for the FSGl-CMRH (SGl-CMRH) is ',res)
#    jj =np.arange(0,iteration)
#    plt.semilogy(jj,resvec,'r--','LineWidth',1);
##    plt.hold(True)
#    print('FCG without preconditioning')
    resvec1,time1,iteration1,res1,finalX2 = PCG( A,B,X,tol,n,s,m,2);
#    print('The number of restarts for the FSGl-CMRH (SGl-CMRH) is ',iteration1)
#    print('The total CPU-time in seconds for the FSGl-CMRH (SGl-CMRH) is ',time1)
#    print('The Frobenius norm of the residual for the FSGl-CMRH (SGl-CMRH) is ',res1)
#    j = np.arange(0,iteration1)
#    plt.semilogy(j,resvec1,'b-.','LineWidth',1)
#    plt.hold(True)
#    print('FCG with GMRES preconditioned')
#    resvec2,time2,iter2,res2,finalX3 = PCG( A,B,X,tol,n,s,m,3) 
#    print('The number of restarts for the FSGl-CMRH (SGl-CMRH) is %d.\n',iter2)
#    print('The total CPU-time in seconds for the FSGl-CMRH (SGl-CMRH) is %f.\n',time2)
#    print('The Frobenius norm of the residual for the FSGl-CMRH (SGl-CMRH) is %f.\n',res2)
#    i = np.arange(0,iter2)
#    plt.semilogy(i,resvec2,'c-*','LineWidth',1);
#    plt.hold(True)
#    plt.xlabel('number of restarts')
#    plt.ylabel('$$\log(\frac{\Vert \mathcal{R}_k\Vert_F}{\Vert \mathcal{R}_{0}\Vert_F})$$','Interpreter','latex')
#    plt.legend('FCG(Bicgstab)','CG','FCG(GlGMRES)')
#    plt.hold(False)
    return finalX2  
