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
    
    # estimated mixing matrix
    print(np.linalg.pinv(W))
    
    # unmixing
    Z = np.dot(W, X)
    
    # plot 
    b_zero_mean = 0
    if b_zero_mean == 1:
        plt.subplot(2, 2, 1)
        plt.plot(XM[0, :], XM[1, :], 'k.')
        plt.plot([np.mean(XM[0, :]), np.mean(XM[0, :])], [np.min(XM[1, :]), np.max(XM[1, :])], color='r')
        plt.plot([np.min(XM[0, :]), np.max(XM[0, :])], [np.mean(XM[1, :]), np.mean(XM[1, :])], color='r')
        plt.xlabel('amplitude')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 2)
        plt.plot(X[0, :], X[1, :], 'g.')
        plt.plot([np.mean(X[0, :]), np.mean(X[0, :])], [np.min(X[1, :]), np.max(X[1, :])], color='r')
        plt.plot([np.min(X[0, :]), np.max(X[0, :])], [np.mean(X[1, :]), np.mean(X[1, :])], color='r')
        plt.xlabel('amplitude')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 3)
        plt.plot(t, X[0, :], 'g')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 4)
        plt.plot(t, X[1, :], 'g')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
        return
    
    b_whitening = 0
    if b_whitening == 1:
    
        plt.subplot(2, 2, 1)
        plt.plot(X[0, :], X[1, :], 'g.')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xlabel('amplitude')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 2)
        plt.plot(ZW[0, :], ZW[1, :], 'm.')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel('amplitude')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 3)
        plt.plot(t, ZW[0, :], 'm')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 4)
        plt.plot(t, ZW[1, :], 'm')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
    
    b_unmixing = 0
    if b_unmixing == 1:
    
        plt.subplot(2, 2, 1)
        plt.plot(ZW[0, :], ZW[1, :], 'm.')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel('amplitude')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 2)
        plt.plot(Z[0, :], Z[1, :], 'b.')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel('amplitude')
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
        
        return
    
    b_original = 1
    if b_original == 1:
        plt.subplot(2, 2, 1)
        plt.plot(t, S[0, :], 'k')
        plt.plot(t, S[1, :] + 5, 'k')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')

        plt.subplot(2, 2, 2)
        plt.plot(S[0, :], S[1, :], 'k.')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel('amplitude (S0)')
        plt.ylabel('amplitude (S1)')
        
        plt.subplot(2, 2, 3)
        plt.plot(t, Z[0, :], 'b')
        plt.plot(t, Z[1, :] + 5, 'b')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 4)
        plt.plot(Z[0, :], Z[1, :], 'b.')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel('amplitude (Z0)')
        plt.ylabel('amplitude (Z1)')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
        
        plt.subplot(2, 2, 3)
        plt.plot(t, X[0, :], 'g')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 4)
        plt.plot(t, X[1, :], 'g')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
        return

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
