# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def my_filter(b, a, x):
    
    # init
    N = len(x)
    M = max(len(a), len(b))

    # init
    y = np.zeros(N)

    # convolution
    for i in range(M, N):
        y[i] = np.sum(b * x[i:(i-M):-1]) - np.sum(a[1:] * y[(i-1):(i-M):-1])

    return y

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L06_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # time variable
    t = np.linspace(0, T, N)
    
    # signal
    x = 0.5 * np.sin(2 * np.pi * 5 * t)  + \
        0.2 * np.sin(2 * np.pi * 10 * t) + \
        0.1 * np.sin(2 * np.pi * 20 * t) + \
        0.5 * np.random.randn(N)
    
    
    # design filter in time domain
    f0 = 25
    [b, a] = signal.butter(4, f0 / (fs/2), 'low')
    
    # apply filter
    u = signal.lfilter(b, a, x)
    
    # apply filter  
    y = signal.filtfilt(b, a, x) # zero-phase filtering
    
    # apply filter 
    fu = my_filter(b, a, x)
    
    # apply filter 
    fy = my_filter(b, a, x)
    fy = fy[::-1]
    fy = my_filter(b, a, fy)
    fy = fy[::-1]
    # or in compact form
    # fy = my_filter(b, a, my_filter(b, a, x)[::-1])[::-1]
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, x,  'k')
    plt.plot(t, y,  'r')
    plt.xlim(0, T)
    plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 2)
    plt.plot(t, y,  'r')
    plt.plot(t, u + 0.05,  'b')
    plt.xlim(0, T)
    plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['filtfilt', 'lfilter'], loc='upper right')
    
    plt.subplot(2, 2, 3)
    plt.plot(t, u,  'r')
    plt.plot(t, fu + 0.05,  'g')
    plt.xlim(0, T)
    plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['lfilter', 'my_filter'], loc='upper right')
    
    plt.subplot(2, 2, 4)
    plt.plot(t, y,  'r')
    plt.plot(t, fy + 0.05,  'g')
    plt.xlim(0, T)
    plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['filtfilt', 'lfilter + reverse'], loc='upper right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
