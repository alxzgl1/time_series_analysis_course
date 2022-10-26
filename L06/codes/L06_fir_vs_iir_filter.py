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
    
    # filter parameters
    f0 = 25 
    fc = f0 / (fs/2) # cutoff frequency
    
    # design filter 
    n1 = 65 # FIR filter order
    n2 = 4  # IIR filter order
    
    # low-pass FIR filter
    a1 = 1
    b1 = signal.firwin(numtaps=n1, cutoff=fc)
    
    # low-pass IIR filter
    [b2, a2] = signal.butter(n2, fc, 'low')
    
    # apply filter
    y1 = signal.lfilter(b1, a1, x) 
    y2 = signal.lfilter(b2, a2, x) 
    
    # frequency response
    hlen = fs // 2
    w1, h1 = signal.freqz(b1, a1, hlen)
    w2, h2 = signal.freqz(b2, a2, hlen)
    
    # impulse response
    rlen = 200
    p = np.zeros(rlen)
    p[0] = 1
    z1 = signal.lfilter(b1, a1, p) 
    z2 = signal.lfilter(b2, a2, p) 

    plt.subplot(2, 2, 1)
    plt.plot(f[:hlen], np.abs(h1), 'b')
    plt.plot(f[:hlen], np.abs(h2), 'r')
    plt.plot([f0, f0], [0, 1], 'k--', linewidth=0.5)
    plt.xlim(0, 50)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    plt.legend(['FIR', 'IIR'], loc='upper right')
    
    plt.subplot(2, 2, 2)
    plt.plot(f[:hlen], np.unwrap(np.angle(h1)), 'b-.')
    plt.plot(f[:hlen], np.unwrap(np.angle(h2)), 'r-.')
    plt.plot([f0, f0], [-9.5, -0.5], 'k--', linewidth=0.5)
    plt.xlim(0, 50)
    plt.ylim(-10, 0.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('phase')
    plt.legend(['FIR', 'IIR'], loc='upper right')
    
    plt.subplot(2, 2, 3)
    plt.plot(t[:rlen], z1, 'b')
    plt.xlim(0, t[rlen])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.legend(['FIR'], loc='upper right')
    
    plt.subplot(2, 2, 4)
    plt.plot(t[:rlen], z2, 'r')
    plt.xlim(0, t[rlen])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.legend(['IIR'], loc='upper right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
