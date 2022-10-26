# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft
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
    f = np.linspace(0, fs, nFFT)
    
    # signal
    x = 0.5 * np.sin(2 * np.pi * 5 * t)  + \
        0.2 * np.sin(2 * np.pi * 10 * t) + \
        0.1 * np.sin(2 * np.pi * 20 * t) + \
        0.5 * np.random.randn(N)
    
    # design filter in time domain
    f0 = 25 
    fc = f0 / (fs/2) # cutoff frequency
    n = 6            # filter order
    
    # low-pass filter
    [b, a] = signal.butter(n, fc, 'low')

    # apply filter
    y = signal.lfilter(b, a, x) 
    
    # spectra
    X = np.abs(fft(x)) / N
    Y = np.abs(fft(y)) / N
    
    # frequency response
    hlen = fs // 2
    w, h = signal.freqz(b, a, hlen)
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, x,  'k')
    plt.plot(t, y,  'r')
    plt.xlim(0, T)
    plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['x', 'y'], loc='upper right')
    
    plt.subplot(2, 2, 2)
    plt.plot(f, X, 'k')
    plt.plot(f, Y, 'r')
    plt.plot([f0, f0], [0, 0.1], 'k--', linewidth=0.5)
    plt.xlim(0, 50)
    plt.ylim(0, 0.3)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    plt.legend(['X', 'Y'], loc='upper right')
    
    plt.subplot(2, 2, 3)
    plt.plot(f[:hlen], np.abs(h), 'b')
    plt.plot([f0, f0], [0, 1], 'k--', linewidth=0.5)
    plt.xlim(0, 50)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 2, 4)
    plt.plot(f[:hlen], np.unwrap(np.angle(h)), 'g-.')
    plt.plot([f0, f0], [-9.5, -0.5], 'k--', linewidth=0.5)
    plt.xlim(0, 50)
    plt.ylim(-10, 0.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('phase')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
