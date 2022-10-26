# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def get_histogram(x, xmin, xmax, bins):
    
    b = np.linspace(xmin, xmax, bins)
    h, b = np.histogram(x, b)
    h = h / np.sum(h)
    b = b[:-1] + (b[1] - b[0]) / 2 
    
    return b, h

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
    P0 = np.random.randn(N)
    P1 = np.random.randn(N)
    
    S = np.zeros((M, N))
    S[0, :] = S0 / np.std(S0)
    S[1, :] = S1 / np.std(S1)
    
    P = np.zeros((M, N))
    P[0, :] = P0 / np.std(P0)
    P[1, :] = P1 / np.std(P1)

    # mixing matrix
    A = np.array([[0.6, 0.4], \
                  [0.4, 0.6]])

    # mix sources
    X = np.dot(A, S) # mixing
    Q = np.dot(A, P) # mixing
    
    # histograms
    xmin = -3
    xmax = 3
    bins = 100
    bS0, hS0 = get_histogram(S[0, :], xmin, xmax, bins)
    bS1, hS1 = get_histogram(S[1, :], xmin, xmax, bins)
    bP0, hP0 = get_histogram(P[0, :], xmin, xmax, bins)
    bP1, hP1 = get_histogram(P[1, :], xmin, xmax, bins)
    
    # plot 
    b_fig_1 = 0
    if b_fig_1 == 1:
        plt.subplot(2, 2, 1)
        plt.plot(t, S[0, :], 'k')
        plt.plot(t, S[1, :] + 5, 'k')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('independent & uncorrelated')
        
        plt.subplot(2, 2, 2)
        plt.plot(t, X[0, :], 'k')
        plt.plot(t, X[1, :] + 5, 'k')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('independent & correlated')
        
        plt.subplot(2, 2, 3)
        plt.plot(t, P[0, :], 'b')
        plt.plot(t, P[1, :] + 5, 'b')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('non-independent & uncorrelated')
        
        plt.subplot(2, 2, 4)
        plt.plot(t, Q[0, :], 'b')
        plt.plot(t, Q[1, :] + 5, 'b')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.title('non-independent & correlated')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
        
    b_fig_2 = 0
    if b_fig_2 == 1:
        plt.subplot(2, 2, 1)
        plt.plot(S[0, :], S[1, :], 'k.')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel('amplitude')
        plt.ylabel('amplitude')
        plt.title('independent & uncorrelated')
        
        plt.subplot(2, 2, 2)
        plt.plot(X[0, :], X[1, :], 'k.')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel('amplitude')
        plt.ylabel('amplitude')
        plt.title('independent & correlated')
        
        plt.subplot(2, 2, 3)
        plt.plot(P[0, :], P[1, :], 'b.')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel('amplitude')
        plt.ylabel('amplitude')
        plt.title('non-independent & uncorrelated')
        
        plt.subplot(2, 2, 4)
        plt.plot(Q[0, :], Q[1, :], 'b.')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel('amplitude')
        plt.ylabel('amplitude')
        plt.title('non-independent & correlated')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
        return

    b_fig_3 = 1
    if b_fig_3 == 1:
        plt.subplot(2, 2, 1)
        plt.bar(bS0, hS0, color='k', width=0.05)
        plt.xlabel('amplitude')
        plt.ylabel('counts')
        plt.title('independent')
        
        plt.subplot(2, 2, 2)
        plt.bar(bS1, hS1, color='k', width=0.05)
        plt.xlabel('amplitude')
        plt.ylabel('counts')
        # plt.title('independent')
        
        plt.subplot(2, 2, 3)
        plt.bar(bP0, hP0, color='b', width=0.05)
        plt.xlabel('amplitude')
        plt.ylabel('counts')
        plt.title('non-independent')
        
        plt.subplot(2, 2, 4)
        plt.bar(bP1, hP1, color='b', width=0.05)
        plt.xlabel('amplitude')
        plt.ylabel('counts')
        # plt.title('non-independent')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
        return
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
