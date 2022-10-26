# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import scipy

#-------------------------------------------------------------------------------
# Function, fpica
# Reference, Hyvarinen et al., 
#-------------------------------------------------------------------------------
def fpica(X, max_iter, tol):

    # whitening 
    [D, E] = np.linalg.eigh(np.cov(X))
    D = np.diag(D)
    WM = np.dot(np.linalg.inv(np.sqrt(D)), E.T)  # whitening matrix
    DM = np.dot(E, np.sqrt(D))                   # dewhitening matrix
    Z = np.dot(WM, X)

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
        print('Convergence after %d steps' % (i))
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
      
      # check
      if i == (max_iter - 1):
          print('No convergence')

    # calculate ICA filters
    A = np.dot(DM, B)
    W = np.dot(B.T, WM)
    
    return W, A

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

    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 2      # duration, in seconds
    N  = T * fs # duration, in samples
    M  = 8      # number of time series
    
    # time variable
    t = np.linspace(0, T, N)
    
    # smooth signal
    S = np.zeros((M, N))
    
    for i in range(0, M):
        S[i, :] = np.sin(2 * np.pi * (i + 1) * t)
    
    # mixing matrix
    A = np.random.rand(M, M) * 0.7
    A = np.triu(A, 1) + np.triu(A, 1).T + np.eye(M) # symmetric
    
    # mixing
    X = np.dot(A, S)
    
    # set ICA algorithm parameters
    max_iter = 1000
    tol = 1e-4
    W, AE = fpica(X, max_iter, tol) # fast-ica 

    # unmixing
    Z = np.dot(W, X)
    Z = Z / np.max(Z)
    
    # plot 
    plt.subplot(2, 2, 1)
    for i in range(0, M):
        plt.plot(t, S[i] + 3*i)
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 2, 2)
    plt.imshow(A)
    
    plt.subplot(2, 2, 3)
    for i in range(0, M):
        plt.plot(t, Z[i] + 2*i)
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 2, 4)
    plt.imshow(AE)

    # set layout
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
