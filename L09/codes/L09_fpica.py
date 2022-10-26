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
      
      # nonlinearity (tanh)
      g = np.tanh(x)
      dg = np.sum(1 - g ** 2, axis=0)

      # nonlinearity (exponent)
      # exp = np.exp(-(x ** 2) / 2)
      # g   = x * exp
      # dg  = np.sum((1 - x ** 2) * exp, axis=0)

      # update B
      B = np.dot(Z, g) - np.tile(dg, (n, 1)) * B

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

    # parameters
    fs = 1000
    T = 1
    N = T * fs
    t = np.linspace(0, T, N)
    
    # sources
    M = 4
    X1 = np.sin(2 * np.pi * 7 * t)
    X2 = signal.sawtooth(2 * np.pi * 5 * t)
    X3 = np.abs(np.cos(2 * np.pi * 3 * t)) - 0.5
    X4 = np.random.randn(N)
    # X4 = np.sign(np.sin(2 * np.pi * 5 * t))
    
    X = np.zeros((M, N))
    X[0, :] = X1 / np.std(X1)
    X[1, :] = X2 / np.std(X2)
    X[2, :] = X3 / np.std(X3)
    X[3, :] = X4 / np.std(X4)

    # mixing matrix
    A = np.array([[1.0, 0.2, 0.4, 0.6],   \
                  [0.3, -0.5, -0.1, 1.0], \
                  [-0.1, 0.2, 1.0, 0.9],  \
                  [0.1, 0.8, -0.7, 1.0]])

    # mix sources
    Y = np.dot(A, X) # mixing
    
    # note, ideal solution for unmixing matrix
    # W = np.linalg.pinv(A)
    
    # set ICA algorithm parameters
    max_iter = 1000
    tol = 1e-4
    W, A = fpica(Y, max_iter, tol) # fast-ica 

    # unmixing
    Z = np.dot(W, Y)
    
    # plot 
    plt.subplot(4, 3, 1)
    plt.plot(X[0, :], color=(0, 0.5, 1))
    plt.subplot(4, 3, 4)
    plt.plot(X[1, :], color=(0, 0.5, 1))
    plt.subplot(4, 3, 7)
    plt.plot(X[2, :], color=(0, 0.5, 1))
    plt.subplot(4, 3, 10)
    plt.plot(X[3, :], color=(0, 0.5, 1))
    
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

    # set layout
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
