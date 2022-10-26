# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import ifft

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L05_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    nFFT = fs   # fft resolution = fs / nFFT, in Hz
    
    # time variable
    t = np.linspace(0, T, N)
    f = np.linspace(0, fs, nFFT)
    
    # frequency
    f0 = 10
    
    # cos
    y = np.zeros(N, 'complex')
    y[f0] = 100.0
    y[N-f0] = 100.0
    
    # sin
    u = np.zeros(N, 'complex')
    u[f0] = -1j * 100.0
    u[N-f0] = 1j * 100.0

    # inverse fourier transform
    x = ifft(y)
    z = ifft(u)

    # plot
    plt.subplot(2, 2, 1)
    plt.plot(f, np.real(y), 'r')
    plt.plot([nFFT/2, nFFT/2], [-100, 100], 'k--', linewidth=0.5)
    plt.xlim(0, nFFT)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.subplot(2, 2, 2)
    plt.plot(f, np.imag(u), 'r')
    plt.plot([nFFT/2, nFFT/2], [-100, 100], 'k--', linewidth=0.5)
    plt.xlim(0, nFFT)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.subplot(2, 2, 3)
    plt.plot(t, x, 'k')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 4)
    plt.plot(t, z, 'k')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
 
    plt.subplot(2, 2, 4)
    plt.plot(t, z, 'k')
    plt.xlim(0, T)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
