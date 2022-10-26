# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft
from scipy import signal

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

    # signal
    x = 0.5 * np.sin(2 * np.pi * 5 * t)  + \
        0.2 * np.sin(2 * np.pi * 10 * t) + \
        0.1 * np.sin(2 * np.pi * 20 * t) + \
        0.5 * np.random.randn(N)
    
    # filter parameters
    f0 = 25 
    fc = f0 / (fs/2) # cutoff frequency
    
    # low-pass IIR filter
    n = 4
    [b, a] = signal.butter(n, fc, 'low')
    
    # apply filter
    y = signal.lfilter(b, a, x) 
    
    # impulse response
    rlen = 200
    p = np.zeros(rlen)
    p[0] = 1
    h = signal.lfilter(b, a, p) 
    
    # convolution
    u = signal.convolve(x, h)
    u = u[0:N]
    
    # fft
    z = np.real(ifft(fft(x, nFFT) * fft(h, nFFT)))
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x,  'k')
    plt.plot(t, y,  'r')
    plt.xlim(0, T)
    plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['x', 'y'], loc='upper right')
    
    plt.subplot(2, 1, 2)
    plt.plot(t, y, 'r')
    plt.plot(t, u + 0.05, 'b')
    plt.plot(t, z + 0.1, 'g')
    plt.xlim(0, T)
    plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['filt', 'conv', 'fft'], loc='upper right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
