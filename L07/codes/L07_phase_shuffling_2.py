# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def phase_no_shuffling(x):

    s = fft(x)
    s = np.abs(s) * np.exp(1j * np.angle(s))
    y = np.real(ifft(s))
    
    return y

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L07_main():
    
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
    
    # phase-shuffling
    np.random.seed(7)
    y = phase_no_shuffling(x)
    
    # fourier transform
    X = fft(x, nFFT)
    Y = fft(y, nFFT)
    
    # amplitude/magnitude spectrum
    Ax = np.abs(X) / N
    Ay = np.abs(Y) / N
    
    # phase spectrum
    X[np.abs(X) < 0.9] = 0
    Px = np.angle(X)
    Y[np.abs(Y) < 0.9] = 0
    Py = np.angle(Y)
    
    # plot
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t, x, 'k')
    plt.xlim(0, T)
    plt.ylim(-1, 1)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 1, 2)
    plt.plot(t, y, 'r')
    plt.xlim(0, T)
    plt.ylim(-1, 1)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    # plot
    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.plot(f, Ax, 'k-.')
    plt.xlim(0, 15)
    plt.ylim(-0.05, 0.3)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')  
    
    plt.subplot(2, 2, 2)
    plt.plot(f, Px, 'k-.')
    plt.xlim(0, 15)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('phase')  
    
    plt.subplot(2, 2, 3)
    plt.plot(f, Ay, 'r-.')
    plt.xlim(0, 15)
    plt.ylim(-0.05, 0.3)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')  
    
    plt.subplot(2, 2, 4)
    plt.plot(f, Py, 'r-.')
    plt.xlim(0, 15)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('phase')  
 
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L07_main()
