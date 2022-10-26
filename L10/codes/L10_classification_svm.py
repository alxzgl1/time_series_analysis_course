# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC

# from scipy.interpolate import spline

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
# def get_continuous_sequence(T, N):
#     
#     n = N // 20
#     x = np.linspace(0, T, n)
#     y = np.random.rand(n)
#     xnew = np.linspace(0, T, N)
#     ynew = spline(x, y, xnew)
# 
#     return ynew
# 
#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def get_sequence(min_len, threshold, N):
    x = np.zeros(N)
    r = np.random.rand(N)
    count = 0
    state = 0
    for i in range(0, N):
        if count > min_len:
            if r[i] > threshold:
                state = 1 - state
                count = 0
        count = count + 1
        x[i] = state
    return x

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L10_main():
    
    # sampling parameters
    fs = 200    # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    M = 8       # channels
    
    # time variable
    t = np.linspace(0, T, N)
    
    # generate data
    np.random.seed(5)
    X = np.random.randn(M, N)
    y = get_sequence(5, 0.8, N) # e.g., stimulus type
    
    # induce some correlation between X and y
    w = 0.4
    X = X + w * np.tile(y, (M, 1))
    X = X * 0.7
    
    # divide data into 2 parts: training and testing datasets
    L = N // 2
   
    # train classifier 
    model = SVC(kernel='linear')
    model.fit(X[:, :L].T, y[:L])
    
    # classifier outcome
    coef = model._get_coef()
    intercept = model.intercept_
    
    # what is decision function?
    Z = np.zeros(N)
    for i in range(0, N):
        Z[i] = np.sum(X[:, i] * coef) + intercept
        
    # decision function
    z = model.decision_function(X.T)

    # test
    v = y[L:]
    u = model.predict(X[:, L:].T)
    u = u > 0.5
    
    # accuracy
    a = np.mean(v == u)
    print('accuracy: %1.2f' % (a))
    
    # plot
    b_fig_1 = 0
    if b_fig_1 == 1:
        plt.subplot(2, 1, 1)
        u = z > 0
        plt.plot(t[:L], z[:L], 'k')
        plt.plot(t[L:], z[L:], 'lightgray')
        plt.plot(t[L:], np.zeros(L), 'g')
        plt.plot(t, y + 6, 'r')
        plt.plot(t[L:], u[L:] + 6, 'b-.')
        plt.plot([t[L], t[L]], [-6, 8], 'k--', linewidth=1.0)
        plt.xlim(0, T)
        plt.ylim(-6, 8)
        plt.xlabel('time')
        plt.ylabel('response')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
    
    b_fig_2 = 1
    if b_fig_2 == 1:
        plt.subplot(2, 1, 1)
        for i in range(0, M):
            plt.plot(t, X[i, :] + 4*i, 'k')
        
        plt.plot(t[:L], y[:L] * 4 + 4*M + 1, 'r')
        plt.plot(t[L:], y[L:] * 4 + 4*M + 1, linestyle='-.', color=(0.8,0.8,0.8))
        plt.plot([t[L], t[L]], [-4, 40], 'k--', linewidth=1.0)
        plt.xlim(0, T)
        plt.ylim(-4, 40)
        plt.xlabel('time')
        plt.ylabel('amplitude / response')
        
        plt.subplot(2, 1, 2)
        plt.plot(t[:L], y[:L], 'r')
        plt.plot(t[L:], y[L:], linestyle='-.', color=(0.8,0.8,0.8))
        plt.plot([t[L], t[L]], [-0.1, 1.1], 'k--', linewidth=1.0)
        plt.plot(t[L:], u, 'r.')
        plt.xlim(0, T)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('time')
        plt.ylabel('response')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
      
        return
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
