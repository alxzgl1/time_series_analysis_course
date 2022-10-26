# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L06_main():
    
    # parameters
    fs = 100
    
    T = 2
    N = T * fs
    t = np.linspace(0, T, N)
    
    # data
    x = np.random.randn(N)
    x = x / np.max(x)
    
    # design low-pass IIR filter
    order = 8
    [b, a] = signal.butter(order, 10.0/(fs/2), 'low')
    y1 = signal.filtfilt(b, a, x)
    
    # design high-pass IIR filter
    order = 8
    [b, a] = signal.butter(order, 40.0/(fs/2), 'high')
    y2 = signal.filtfilt(b, a, x)
    
    # design band-pass IIR filter
    order = 8
    [b, a] = signal.butter(order, [10.0/(fs/2), 15.0/(fs/2)], 'bandpass')
    y3 = signal.filtfilt(b, a, x)
    
    # design stop-band IIR filter
    order = 8
    [b, a] = signal.butter(order, [10.0/(fs/2), 15.0/(fs/2)], 'bandstop')
    y4 = signal.filtfilt(b, a, x)
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, x, 'k')
    plt.plot(t, y1, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.05, 1.05)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
 
    plt.subplot(2, 2, 2)
    plt.plot(t, x, 'k')
    plt.plot(t, y2, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.05, 1.05)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 2, 3)
    plt.plot(t, x, 'k')
    plt.plot(t, y3, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.05, 1.05)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')

    plt.subplot(2, 2, 4)
    plt.plot(t, x, 'k')
    plt.plot(t, y4, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.05, 1.05)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
