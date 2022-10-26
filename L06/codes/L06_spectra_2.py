# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft

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
    
    # remove above 25 Hz
    f0 = 25
    fx = fft(x)
    fx[np.arange(f0, nFFT-f0)] = 0
    y = np.real(ifft(fx))
    Y = np.abs(fft(y)) / N
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, x,  'k')
    plt.xlim(0, T)
    plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 2)
    plt.plot(f, X,  'k')
    plt.xlim(0, 50)
    plt.ylim(0, 0.3)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    
    plt.subplot(2, 2, 3)
    plt.plot(f, Y,  'r')
    plt.plot([f0, f0], [0, 0.1], 'k--', linewidth=0.5)
    plt.xlim(0, 50)
    plt.ylim(0, 0.3)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    
    plt.subplot(2, 2, 4)
    plt.plot(t, y,  'r')
    plt.xlim(0, T)
    plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
