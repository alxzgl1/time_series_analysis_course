# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L10_main():
    
    # parameters
    fs = 1000
    T = 1
    N = T * fs
    t = np.linspace(0, T, N)
    np.random.seed(5)
    
    # sources
    M = 2
    S0 = np.sin(2 * np.pi * 7 * t)
    S1 = np.random.randn(N)
    
    S = np.zeros((M, N))
    S[0, :] = S0 / np.std(S0)
    S[1, :] = S1 / np.std(S1)

    # mixing matrix
    A = np.array([[1.0, 0.0], \
                  [0.5, 0.0]])

    # mix sources
    X = np.dot(A, S) # mixing
    
    # subtract mean
    X = X - np.tile(np.mean(X, axis=1), (N, 1)).transpose()
    
    X0 = X[0, :]
    X1 = X[1, :]
    
    # polyfit
    p = np.polyfit(X0, X1, 1) 
    X2 = p[0] * X0 + p[1]
    
    p0 = np.sum(X1 / X0) / N
    print(p0)
    
    print(p[0])
    print(p[1])
    
    # covariance
    C = np.cov(X)[0,1]
    print(C)
    
    # plot
    
    plt.subplot(2, 2, 1)
    plt.plot(t, X0, 'k')
    plt.plot(t, X1 + 3, 'r')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 2)
    plt.plot(X0, X1, '.')
    plt.plot(X0, X2, '-')
    plt.xlabel('amplitude')
    plt.ylabel('amplitude') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
