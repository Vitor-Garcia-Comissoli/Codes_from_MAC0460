# -*- coding: utf-8 -*-
"""
Created on Sat May 14 15:32:23 2022

@author: vitor
"""
import numpy as np

def cross_entropy_loss(w, X, y):  
    
    N = X.shape[0]
    
    Xw = X @ w
    ywX = y * Xw
    
    loss = (1/N) * np.log(1 + np.exp(-ywX)).sum()
        
    return loss

def cross_entropy_gradient(w, X, y):
    
    Xw = X @ w
    ywX = y * Xw
    
    yX = X.T * y
    
    grad = -(yX/(1 + np.exp(ywX))).mean(axis=1)
    
    return grad

def train_logistic(X, y, learning_rate = 1e-1, w0 = None, num_iterations = 300, return_history = False):
    
    N = X.shape[0]
    
    x_1 = np.ones((N,1))    
    X = np.append(x_1, X, axis=1)
    
    if w0 == None:
        w0 = np.random.normal(loc = 0, scale = 1, size = X.shape[1])
    
    list = [cross_entropy_loss(w0, X, y)]
    
    wi = w0
    
    for i in range(num_iterations):
        wi = wi - learning_rate * cross_entropy_gradient(wi, X, y)
        list += [cross_entropy_loss(wi, X, y)]
    
    weight = wi   
    
    if return_history == True:
        return weight, list
    else:
        return weight
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_logistic(X, w):
    
    N = X.shape[0]
    
    x_1 = np.ones((N,1))    
    X = np.append(x_1, X, axis=1)
    
    y = X @ w 
    
    prediction = sigmoid(y)
    return prediction
    
def main():
    
    X_ini = np.array([[1,2],[4,5],[7,8]]) #(N,d)
    X = np.array([[1,1,2],[1,4,5],[1,7,8]]) #(N,1+d)

    y = np.array([1,2,3]) #(N, )
    w = np.array([1,2,3]) #(1+d, )
    
    print(cross_entropy_loss(w, X, y),'\n')
    print(cross_entropy_gradient(w, X, y),'\n')
    
    a, b = train_logistic(X_ini, y, return_history = True)
    print(a,'\n')
    #print(b,'\n')
    
    c = predict_logistic(X_ini, w)
    print(c, '\n')
    
main()