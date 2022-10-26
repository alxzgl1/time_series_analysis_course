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
def inverse_fourier_tansform(y, N):
    
    # length of y
    nFFT = len(y)

    # inverse fourier transform
    x = np.zeros(N, 'complex') 
    t = np.arange(0, N)
    for n in range(0, N):
      # relative frequency
      f = n / nFFT
      # complex exponent
      x[n] = (1 / N) * np.sum(np.exp(1j * 2 * np.pi * t * f) * y)
      
    return x

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
    
    # signal
    x1 = 0.5 * np.sin(2 * np.pi * 2 * t)
    x2 = 0.2 * np.cos(2 * np.pi * 5 * t)
    x3 = 0.1 * np.sin(2 * np.pi * 10 * t)
    x = x1 + x2 + x3
    
    # fourier transform
    y = fourier_tansform(x, nFFT)
    
    # power spectrum
    f = np.linspace(0, fs, N)
    
    # inverse fourier transform
    z = inverse_fourier_tansform(y, N)
    z = np.real(z)
    
    # plot
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t, x, 'r')
    plt.xlim(0, T)
    plt.ylim(-1, 1)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 1, 2)
    plt.plot(f, np.abs(y), 'b-.')
    plt.xlim(0, 15)
    plt.ylim(-10, 260)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(f, np.abs(y), 'b-.')
    plt.xlim(0, 15)
    plt.ylim(-10, 260)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.subplot(2, 1, 2)
    plt.plot(t, z, 'r')
    plt.plot(t, z, 'g-.')
    plt.xlim(0, T)
    plt.ylim(-1, 1)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['original', 'inverse'], loc='upper right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
