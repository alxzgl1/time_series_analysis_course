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
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    nFFT = fs   # fft resolution = fs / nFFT, in Hz
    
    # time and frequency variables
    t = np.linspace(0, T, N)
    f = np.linspace(0, fs, nFFT)
    
    # signal
    x1 = 0.5 * np.sin(2 * np.pi * 2 * t)
    x2 = 0.2 * np.sin(2 * np.pi * 5 * t)
    x3 = 0.1 * np.sin(2 * np.pi * 10 * t)
    x = x1 + x2 + x3
    
    # fourier transform
    y = fourier_tansform(x, nFFT)
    
    # magnitude
    Y = np.abs(y)
    
    # power spectral density
    U = (1 / (2 * np.pi * N)) * np.abs(y) ** 2

    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x, 'r')
    plt.xlim(0, T)
    plt.ylim(-1, 1)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 3)
    plt.plot(f, Y, 'b-.')
    plt.xlim(0, 15)
    plt.ylim(-10, 260)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.subplot(2, 2, 4)
    plt.plot(f, U, 'g-.')
    plt.xlim(0, 15)
    plt.ylim(-1, 11)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('PSD') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
