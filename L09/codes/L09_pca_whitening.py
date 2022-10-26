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
    
    # covariance
    C = np.cov(X)
    
    # eigen-decomposition
    [D, V] = np.linalg.eigh(C)
    
    # whitening
    D = np.diag(D)
    WM = np.dot(np.linalg.inv(np.sqrt(D)), V.T)  # whitening matrix
    Z = np.dot(WM, X) * 0.15
    
    b_fig_1 = 0
    if b_fig_1 == 1:
        plt.subplot(2, 1, 1)
        plt.plot(np.tile(t, (M, 1)).T, X.T + np.arange(0, M))
        plt.xlim(0, T)
        # plt.ylim(-1.2, 1.2)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude') 
        
        plt.subplot(2, 1, 2)
        M = np.shape(Z)[0]
        plt.plot(np.tile(t, (M, 1)).T, Z.T + np.arange(0, M))
        plt.xlim(0, T)
        # plt.ylim(-1.2, 1.2)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude') 
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
        
    b_fig_2 = 1
    if b_fig_2 == 1:

        plt.subplot(2, 2, 1)
        plt.imshow(np.cov(X))
        plt.colorbar()
        plt.xlabel('sensors')
        plt.ylabel('sensors')
        plt.title('covariance (original)')
        
        plt.subplot(2, 2, 2)
        plt.imshow(np.cov(Z))
        plt.colorbar()
        plt.xlabel('sensors')
        plt.ylabel('sensors')
        plt.title('covariance (whitened)')
        
        return
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
