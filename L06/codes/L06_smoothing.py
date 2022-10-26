# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def do_smoothing(x, M):
    
    N = len(x)
    y = np.zeros(N)
    
    for i in range(0, N-M):
        y[i+M] = np.sum(x[i:(i+M)]) / M
        
    return y

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L06_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    nFFT = fs   # resolution
    
    # time variable
    t = np.linspace(0, T, N)
    f = np.linspace(0, fs, nFFT)
    
    # signal
    x = 0.5 * np.sin(2 * np.pi * 5 * t)  + \
        0.2 * np.sin(2 * np.pi * 10 * t) + \
        0.1 * np.sin(2 * np.pi * 20 * t) + \
        0.5 * np.random.randn(N)
    X = np.abs(fft(x)) / N
    
    # smoothing, M samples
    M = 20
    y = do_smoothing(x, M)
    Y = np.abs(fft(y)) / N
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x,  color=(0.5, 0.5, 0.5))
    plt.plot(t, y,  'r')
    plt.xlim(0, T)
    plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['x', 'y'], loc='upper right')
    
    plt.subplot(2, 1, 2)
    plt.plot(f, X,  color=(0.5, 0.5, 0.5))
    plt.plot(f, Y,  'r-.')
    plt.xlim(0, 100)
    plt.ylim(0, 0.25)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    plt.legend(['X', 'Y'], loc='upper right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
