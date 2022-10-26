# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L08_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # time variable
    t = np.linspace(0, T, N)
    
    # smooth signal
    np.random.seed(25)
    S0 = np.random.randn(1, N) * 2.0
    S1 = np.random.randn(1, N) * 2.0
    S = np.concatenate((S0, S1))
    [b, a] = signal.butter(4, [10.0/(fs/2), 20.0/(fs/2)], 'bandpass')
    S = signal.filtfilt(b, a, S)
    S0 = S[0, :]
    S1 = S[1, :]
    
    # mixing matrix
    A = np.array([[1.0, 0.0], \
                  [0.0, 1.0]])
    
    # mixing
    X = np.dot(A, S)
    X0 = X[0, :]
    X1 = X[1, :]
 
    # what is dot? 
    # X[0, :] =  A[0, 0] * S[0, :] + A[0, 1] * S[1, :]
    # X[1, :] =  A[1, 0] * S[0, :] + A[1, 1] * S[1, :]
    
    # similarity between x and y via Pearson correlation coefficient
    # r = np.sum(((x - np.mean(x)) * (y - np.mean(y))) / (np.std(x) * np.std(y))) / N
    # r = np.corrcoef(x, y)[0, 1] # numpy implementation
    
    # linear fit / represent y using linearly weighted x
    # p = np.polyfit(x, y, 1) 
    # fx = p[0] * x + p[1]

    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, S0, 'k')
    plt.xlim(0, T)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('time (s)')
    plt.ylabel('S0') 
    
    plt.subplot(2, 2, 2)
    plt.plot(t, S1, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('time (s)')
    plt.ylabel('S1') 
    
    plt.subplot(2, 2, 3)
    plt.plot(t, X0, 'k')
    plt.xlim(0, T)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('time (s)')
    plt.ylabel('Z0') 
    
    plt.subplot(2, 2, 4)
    plt.plot(t, X1, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('time (s)')
    plt.ylabel('Z1') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L08_main()
