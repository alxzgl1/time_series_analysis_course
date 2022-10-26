# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L06_main():
    
    # sampling parameters
    fs = 100   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # time variable
    t = np.linspace(0, T, N)
    
    # cutoff frequency
    f0 = 5
    
    # padding length
    L = fs // f0
    
    # signal
    np.random.seed(10)
    s = np.random.randn(N+2*L)
    s = s / np.max(s)
    x = s[L:(N+L)]
    
    # design filter in time domain
    fc = f0 / (fs/2) # cutoff frequency
    n = 16            # filter order
    
    # low-pass filter
    [b, a] = signal.butter(n, fc, 'low')

    # apply filter
    y = signal.filtfilt(b, a, x) 
    
    # zeros-padding
    u = np.concatenate((np.zeros(L), x, np.zeros(L)))
    fu = u
    u = signal.filtfilt(b, a, u) 
    u = u[L:(N+L)]
    
    #  signal-mirroring
    z = np.concatenate((x[L:0:-1], x, x[N:(N-L-1):-1]))
    fz = z
    z = signal.filtfilt(b, a, z) 
    z = z[L:(N+L)]
    
    # extend time variable
    p = np.concatenate((-t[L:0:-1], t, T+t[1:(L+1)]))
 
    # plot
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(p, s, color=(0.5,0.5,0.5))
    plt.plot(t, y, 'r')
    plt.plot([0, 0], [-1, 1], 'k--', linewidth=1.0)
    plt.plot([T, T], [-1, 1], 'k--', linewidth=1.0)
    plt.xlim(p[0], p[-1])
    plt.ylim(-1.05, 1.05)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    # plt.legend(['x', 'y'], loc='upper right')
    
    plt.subplot(2, 1, 2)
    plt.plot(p, fu, color=(0.5,0.5,0.5))
    plt.plot(t, u, 'b')
    plt.plot([0, 0], [-1, 1], 'k--', linewidth=1.0)
    plt.plot([T, T], [-1, 1], 'k--', linewidth=1.0)
    plt.xlim(p[0], p[-1])
    plt.ylim(-1.05, 1.05)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    # plt.legend(['x', 'u'], loc='upper right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    # plot
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(p, s, color=(0.5,0.5,0.5))
    plt.plot(t, y, 'r')
    plt.plot([0, 0], [-1, 1], 'k--', linewidth=1.0)
    plt.plot([T, T], [-1, 1], 'k--', linewidth=1.0)
    plt.xlim(p[0], p[-1])
    plt.ylim(-1.05, 1.05)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    # plt.legend(['x', 'y'], loc='upper right')
    
    plt.subplot(2, 1, 2)
    plt.plot(p, fz, color=(0.5,0.5,0.5))
    plt.plot(t, z, 'g')
    plt.plot([0, 0], [-1, 1], 'k--', linewidth=1.0)
    plt.plot([T, T], [-1, 1], 'k--', linewidth=1.0)
    plt.xlim(p[0], p[-1])
    plt.ylim(-1.05, 1.05)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    # plt.legend(['x', 'u'], loc='upper right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
