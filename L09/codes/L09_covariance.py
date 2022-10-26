# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

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
    S = np.random.randn(M, N)
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
    
    # un-mixing
    W = np.linalg.inv(A)
    Z = np.dot(W, X)
 
    # what is dot? 
    # X[0, :] =  A[0, 0] * S[0, :] + A[0, 1] * S[1, :] ...
    # X[1, :] =  A[1, 0] * S[0, :] + A[1, 1] * S[1, :] ...
    
    # covarince
    C = np.zeros((M, M))
    for i in range(0, M):
        for j in range(0, M):
            C[i,j] = np.sum((X[i] - np.mean(X[i])) * (X[j] - np.mean(X[j]))) / N
    
    # numpy implementation   
    # C = np.cov(X)
    
    # correlation
    R = np.corrcoef(X)
    
    # un-mixing
    H = np.linalg.inv(C) / N
    Y = np.dot(H, X)

    # plot
    b_fig_1 = 0
    if b_fig_1 == 1:
        plt.subplot(2, 1, 1)
        plt.plot(np.tile(t, (M, 1)).T, X.T + np.arange(0, M))
        plt.xlim(0, T)
        # plt.ylim(-1.2, 1.2)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude') 
        
        plt.subplot(2, 2, 3)
        plt.imshow(C)
        plt.colorbar()
        plt.xlabel('sensors')
        plt.ylabel('sensors')
        plt.title('covariance')
        
        plt.subplot(2, 2, 4)
        plt.imshow(R)
        plt.colorbar()
        plt.xlabel('sensors')
        plt.ylabel('sensors')
        plt.title('correlation')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
    
    b_fig_2 = 0
    if b_fig_2 == 1:

        plt.subplot(2, 2, 1)
        plt.imshow(R)
        plt.colorbar()
        plt.xlabel('sensors')
        plt.ylabel('sensors')
        plt.title('correlation')
        
        plt.subplot(2, 2, 2)
        plt.imshow(A)
        plt.colorbar()
        plt.xlabel('sensors')
        plt.ylabel('sensors')
        plt.title('mixing matrix')
        
        plt.subplot(2, 2, 3)
        plt.imshow(R - A)
        plt.colorbar()
        plt.xlabel('sensors')
        plt.ylabel('sensors')
        plt.title('correlation - mixing')
        
        return
        
    b_fig_3 = 1
    if b_fig_3 == 1:
        
        plt.subplot(2, 1, 1)
        plt.plot(np.tile(t, (M, 1)).T, Z.T + np.arange(0, M))
        plt.xlim(0, T)
        # plt.ylim(-1.2, 1.2)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude') 
        
        plt.subplot(2, 1, 2)
        plt.plot(np.tile(t, (M, 1)).T, Y.T + np.arange(0, M))
        plt.xlim(0, T)
        # plt.ylim(-1.2, 1.2)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude') 

        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
