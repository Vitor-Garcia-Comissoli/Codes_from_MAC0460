# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:03:33 2022

@author: vitor
"""
import numpy as np

X = np.array([[1,2,3],[4,5,6],[7,8,9]])

print ('X =',X,'\n')

y = np.array([1,2,3])

print ('y =',y,'\n')

yi_Xi = y @ X

print('X @ y =',yi_Xi,'\n')

w = np.array([1,2,3])

print('w =',w,'\n')

wt = np.transpose(w)

print('w^t =',wt,'\n')

wt_Xi = wt @ X

print('wt @ X =',wt_Xi,'\n')

yi_wt_Xi = y * wt_Xi

print('y * wt @ X =',yi_wt_Xi,'\n')

N = X.shape[0]

print('N =',N,'\n')

loss = (1/N) * np.log(1 + np.exp(-yi_wt_Xi)).sum()

print('loss =',loss,'\n')

grad = -(1/N) * yi_Xi/(1 + np.exp(yi_wt_Xi)).sum()

print('grad =', grad,'\n')