# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft

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
    
    # signal
    x = np.zeros(N)
    x[200:400] = 1.0
    x[600:800] = 1.0
    
    # fourier transform
    y = fft(x, nFFT)
    Y = np.abs(y)
    
    # cut
    M = 50
    y[M:(N-M)] = 0
    
    # magnitude
    U = np.abs(y)
    
    # inverse fourier transform
    z = ifft(y)

    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, x, 'k')
    plt.xlim(0, T)
    plt.ylim(-0.25, 1.25)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 

    plt.subplot(2, 2, 2)
    plt.plot(f, Y, 'r')
    plt.xlim(0, nFFT/4)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.subplot(2, 2, 3)
    plt.plot(f, U, 'r')
    plt.xlim(0, nFFT/4)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.subplot(2, 2, 4)
    plt.plot(t, z, 'k')
    plt.xlim(0, T)
    plt.ylim(-0.25, 1.25)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
