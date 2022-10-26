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
    S0 = np.sin(2 * np.pi * 7 * t)
    S1 = np.random.randn(N)
    
    S = np.zeros((M, N))
    S[0, :] = S0 / np.std(S0)
    S[1, :] = S1 / np.std(S1)

    # mixing matrix
    A = np.array([[0.6, 0.4], \
                  [0.4, 0.6]])

    # mix sources
    X = np.dot(A, S) # mixing
    
    # simulate non-zero mean
    X = X + np.tile(np.array([2.0, -1.0]), (N, 1)).transpose()
    XM = X

    # select algorithm
    max_iter = 1000
    tol = 1e-6
    
    # *** ICA algorithm begins here ***

    # remove mean
    X = X - np.tile(np.mean(X, axis=1), (N, 1)).transpose()
    
    # whitening 
    [D, E] = np.linalg.eigh(np.cov(X))
    D = np.diag(D)
    WM = np.dot(np.linalg.inv(np.sqrt(D)), E.T)  # whitening matrix
    Z = np.dot(WM, X)
    ZW = Z

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
    Z = np.dot(W, X)
    
    # PCA
    C = np.cov(X)
    [D, V] = np.linalg.eigh(C)
    Q = np.dot(np.diag(D), V.T)
    P = np.dot(Q, X)
    
    # plot 
    b_original = 1
    if b_original == 1:
        plt.subplot(2, 2, 1)
        plt.plot(t, X[0, :], 'k')
        plt.plot(t, X[1, :] + 5, 'k')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')

        plt.subplot(2, 2, 3)
        plt.plot(t, S[0, :], 'b')
        plt.plot(t, S[1, :] + 5, 'b')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('ICA')

        plt.subplot(2, 2, 4)
        plt.plot(t, P[0, :], 'g')
        plt.plot(t, P[1, :] + 5, 'g')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('PCA')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
        return

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
