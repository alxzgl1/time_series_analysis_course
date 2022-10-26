# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft
from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L03_main():
    
    # sampling parameters
    fs = 1000    # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    M = 20
    
    # time variable
    t = np.linspace(0, T, N)
    f = np.linspace(0, fs/M, int(N/M)) 
    
    # sin signal
    f0 = 5
    x1 = np.sin(2 * np.pi * f0 * t)
    # chirp signal
    f0 = 1
    f1 = 10
    t1 = T
    x2 = signal.chirp(t, f0, t1, f1) 
    # non-periodic signal
    f0 = 10
    x3 = np.sin(2 * np.pi * f0 * t) 
    x3[int(N/4):] = 0
    # random noise
    x4 = np.random.randn(N)
    
    # fft
    y1 = np.abs(fft(x1))
    y1 = y1[0:int(N/M)]

    y2 = np.abs(fft(x2))
    y2 = y2[0:int(N/M)]

    y3 = np.abs(fft(x3))
    y3 = y3[0:int(N/M)]
    
    y4 = np.abs(fft(x4))
    y4 = y4[0:int(N/M)]
    
    # plot
    plt.subplot(4, 2, 1)
    plt.plot(t, x1)
    plt.subplot(4, 2, 2)
    plt.plot(f, y1)
    plt.subplot(4, 2, 3)
    plt.plot(t, x2)
    plt.subplot(4, 2, 4)
    plt.plot(f, y2)
    plt.subplot(4, 2, 5)
    plt.plot(t, x3)
    plt.subplot(4, 2, 6)
    plt.plot(f, y3)
    plt.subplot(4, 2, 7)
    plt.plot(t, x4)
    plt.subplot(4, 2, 8)
    plt.plot(f, y4)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
