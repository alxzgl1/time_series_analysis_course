# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def fourier_tansform(x, nFFT):
    
    # fourier transform
    y = np.zeros(nFFT, 'complex') 
    t = np.arange(0, len(x))
    for k in range(0, nFFT):
      # relative frequency
      f = k / nFFT
      # complex exponent
      y[k] = np.sum(np.exp(-1j * 2 * np.pi * t * f) * x)
      
    return y

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L05_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 2      # duration, in seconds
    N  = T * fs # duration, in samples
    
    nFFT = 2 * fs   # fft resolution = fs / nFFT, in Hz
    
    # time variable
    t = np.linspace(0, T, N)
    
    # signal
    x1  = 0.5 * np.sin(2 * np.pi * 2 * t)
    x2a = 0.2 * np.sin(2 * np.pi * 5 * t)
    x2b = 0.2 * np.sin(2 * np.pi * 5.5 * t)
    x3  = 0.1 * np.sin(2 * np.pi * 10 * t)
    xA = x1 + x2a + x3
    xB = x1 + x2b + x3
    
    # fourier transform
    yA = fourier_tansform(xA, nFFT)
    yB = fourier_tansform(xB, nFFT)
    
    # power spectrum
    f = np.linspace(0, fs, nFFT)
    YA = np.abs(yA) / N
    YB = np.abs(yB) / N
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, xA, 'r')
    plt.xlim(0, T)
    plt.ylim(-1, 1)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 2)
    plt.plot(f, YA, 'r-.')
    plt.xlim(0, 15)
    plt.ylim(-0.1, 0.5)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    plt.legend(['5 Hz'], loc='upper right')
    
    plt.subplot(2, 2, 3)
    plt.plot(t, xB, 'k')
    plt.xlim(0, T)
    plt.ylim(-1, 1)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 4)
    plt.plot(f, YB, 'k-.')
    plt.xlim(0, 15)
    plt.ylim(-0.1, 0.5)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    plt.legend(['5.5 Hz'], loc='upper right')

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
