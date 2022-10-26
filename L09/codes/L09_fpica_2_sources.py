# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import scipy

#------------------------------------------------------------------------------
# Function
# What decorrelation does?
#   This makes ICA basis vectors orthogonal (orthonormal) after each iteration.
#------------------------------------------------------------------------------
def symmetric_decorrelation(B):
    
    # B = B * (B' * B) ^ -0.5   
    B = np.dot(B, scipy.linalg.fractional_matrix_power(np.dot(B.T, B), -0.5))
    return B

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
    X1 = np.sin(2 * np.pi * 7 * t)
    X2 = np.random.randn(N)
    
    X = np.zeros((M, N))
    X[0, :] = X1 / np.std(X1)
    X[1, :] = X2 / np.std(X2)

    # mixing matrix
    A = np.array([[0.6, 0.4], \
                  [0.4, 0.6]])

    # mix sources
    Y = np.dot(A, X) # mixing

    # whitening 
    [D, E] = np.linalg.eigh(np.cov(Y))
    D = np.diag(D)
    WM = np.dot(np.linalg.inv(np.sqrt(D)), E.T)  # whitening matrix
    Z = np.dot(WM, Y)
    
    
    # *** ICA algorithm begins here ***
    
    #  algorithm parameters
    max_iter = 1000
    tol = 1e-6

    # number of channels
    n = np.shape(Z)[0]
    
    # init matrix
    B = np.random.randn(n, n)                            
    BOld = np.zeros(np.shape(B))
    
    # routine
    for i in range(0, max_iter):
      
      # symmetric decorrelation (orthogonalization after each iteration)
      B = symmetric_decorrelation(B)    
      
      # convergence condition
      minAbsCos = np.min(np.abs(np.diag(np.dot(B.T, BOld)))) # min(abs(diag(B' * BOld)))
      if (1 - minAbsCos < tol):
        break
      BOld = B
    
      # whitened_X * B
      x = np.dot(Z.T, B)
      
      # nonlinearity
      exp = np.exp(-(x ** 2) / 2)
      g   = x * exp
      dg  = (1 - x ** 2) * exp
      
      # update B
      B = np.dot(Z, g) - np.tile(np.sum(dg, axis=0), (n, 1)) * B

    # calculate ICA filters
    W = np.dot(B.T, WM)

    # *** ICA algorithm ends here ***
    
    # unmixing
    Z = np.dot(W, Y)
    
    # plot 
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(t, X[0, :], 'k')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 2, 2)
    plt.plot(t, X[1, :], 'k')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 2, 3)
    plt.plot(t, Y[0, :], 'g')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 2, 4)
    plt.plot(t, Y[1, :], 'g')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    plt.figure(2)
    
    plt.subplot(2, 2, 1)
    plt.plot(t, Y[0, :], 'g')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 2, 2)
    plt.plot(t, Y[1, :], 'g')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 2, 3)
    plt.plot(t, Z[0, :], 'b')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 2, 4)
    plt.plot(t, Z[1, :], 'b')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
