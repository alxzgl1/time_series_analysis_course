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
    T  = 3      # duration, in seconds
    N  = T * fs # duration, in samples
    L  = N // 3
    
    nFFT = fs   # fft resolution = fs / nFFT, in Hz
    
    # time variable
    t = np.linspace(0, T, N)
    
    # signal
    x1 = 0.7 * np.sin(2 * np.pi * 2 * t)
    x2 = 0.5 * np.sin(2 * np.pi * 5 * t)
    x3 = 0.3 * np.sin(2 * np.pi * 10 * t)
    x = np.concatenate((x1[:L], x2[:L], x3[:L]))
    
    # fourier transform
    M = 5
    Y = np.zeros((nFFT, M))
    for i in range(0, M):
        t1 = i * L//2
        t2 = t1 + L
        y = fourier_tansform(x[t1:t2], nFFT)
        Y[:, i] = np.abs(y)
    
    # power spectrum
    f = np.linspace(0, fs, nFFT)
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x, 'r')
    for i in range(1, M+1):
        plt.plot([i * t[L//2], i * t[L//2]], [-0.75, 0.75], 'k--', linewidth=0.5)
    plt.xlim(0, T)
    plt.ylim(-1, 1)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 3, 4)
    plt.plot(f, Y[:, 0], 'b')
    plt.plot(f, Y[:, 1], 'g')
    plt.xlim(0, 15)
    plt.ylim(-50, 400)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.subplot(2, 3, 5)
    plt.plot(f, Y[:, 2], 'b')
    plt.plot(f, Y[:, 3], 'g')
    plt.xlim(0, 15)
    plt.ylim(-50, 400)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.subplot(2, 3, 6)
    plt.plot(f, Y[:, 4], 'b')
    plt.xlim(0, 15)
    plt.ylim(-50, 400)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
