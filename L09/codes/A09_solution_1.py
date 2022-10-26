# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy import signal

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
      
      # 
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
    np.random.seed(25)
    S = np.random.rand(M, N)
    [b, a] = signal.butter(4, [10.0/(fs/2), 20.0/(fs/2)], 'bandpass')
    L = 200
    S = signal.filtfilt(b, a, np.concatenate((np.zeros((M, L)), S, np.zeros((M, L))), axis=1))
    S = S[:, L:(N+L)]
    S = signal.filtfilt(b, a, S)
    
    # mixing matrix
    A = np.random.rand(M, M) * 0.7
    A = np.triu(A, 1) + np.triu(A, 1).T + np.eye(M) # symmetric
    
    # mixing
    X = np.dot(A, S)
    
    # set ICA algorithm parameters
    max_iter = 1000
    tol = 1e-4
    W, A = fpica(X, max_iter, tol) # fast-ica 

    # unmixing
    Z = np.dot(W, X)
    Z = Z / np.max(Z)
    
    # plot 
    plt.subplot(2, 2, 1)
    for i in range(0, M):
        plt.plot(t, S[i] + i)
    plt.xlim(0, T)
    
    plt.subplot(2, 2, 3)
    for i in range(0, M):
        plt.plot(t, Z[i] + i)
    plt.xlim(0, T)
    
    
    """
    plt.subplot(4, 3, 2)
    plt.plot(Y[0, :], color=(1, 0.5, 0))
    plt.subplot(4, 3, 5)
    plt.plot(Y[1, :], color=(1, 0.5, 0))
    plt.subplot(4, 3, 8)
    plt.plot(Y[2, :], color=(1, 0.5, 0))
    plt.subplot(4, 3, 11)
    plt.plot(Y[3, :], color=(1, 0.5, 0))
    
    plt.subplot(4, 3, 3)
    plt.plot(Z[0, :], color=(0.5, 0, 0.5))
    plt.subplot(4, 3, 6)
    plt.plot(Z[1, :], color=(0.5, 0, 0.5))
    plt.subplot(4, 3, 9)
    plt.plot(Z[2, :], color=(0.5, 0, 0.5))
    plt.subplot(4, 3, 12)
    plt.plot(Z[3, :], color=(0.5, 0, 0.5))
    
    """

    # set layout
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
