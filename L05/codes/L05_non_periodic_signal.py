# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft
from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L05_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 3      # duration, in seconds
    N  = T * fs # duration, in samples
    L  = int(0.5 * fs)
    
    nFFT = fs   # fft resolution = fs / nFFT, in Hz
    
    # time variable
    t = np.linspace(0, T, N)
    
    # signal
    x = np.sin(2 * np.pi * 5 * t)
    x[:L] = 0
    x[(N-L):] = 0
    
    # window
    w = signal.hanning(N)
    z = x * w
    
    # fourier transform
    y = fft(x, nFFT)
    u = fft(z, nFFT)
    
    # power spectrum
    f = np.linspace(0, fs, nFFT)
    Y = np.abs(y)
    U = np.abs(u)
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, x, 'k')
    plt.xlim(0, T)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 2)
    plt.plot(f, Y, 'b')
    plt.xlim(0, 15)
    plt.ylim(-10, 260)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.subplot(2, 2, 3)
    plt.plot(t, w, 'g-.')
    plt.plot(t, z, 'k')
    plt.xlim(0, T)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 4)
    plt.plot(f, U, 'b')
    plt.xlim(0, 15)
    plt.ylim(-10, 260)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
