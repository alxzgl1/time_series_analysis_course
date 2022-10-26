# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft
from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L05_main():
    
    # generate gaussian noise
    N = 100
    x = np.random.randn(N) + 0.5 * np.sin(2 * np.pi * 10 * np.linspace(0, 1, N))

    # compute ACF
    rx = signal.correlate(x, x)
    rx = rx / np.max(rx)
    
    # compute ACF using FFT
    nFFT = 2 * N
    ry = np.real(ifft(fft(x, nFFT) * np.conj(fft(x, nFFT))))
    ry = np.concatenate((ry[N::-1], ry[1:N:]))
    ry = ry / np.max(ry)
    
    # plot gaussian noise
    plt.subplot(2, 2, 1) 
    plt.plot(x, 'k')
    plt.xlim(0, N)
    plt.xlabel('samples')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 2) 
    plt.plot(rx)
    plt.xlim(0, 2*N-1)
    plt.ylim(-0.5, 1.25)
    plt.xlabel('samples')
    plt.ylabel('correlation') 
    
    plt.subplot(2, 2, 4) 
    plt.plot(ry, 'g')
    plt.xlim(0, 2*N-1)
    plt.ylim(-0.5, 1.25)
    plt.xlabel('samples')
    plt.ylabel('correlation') 
    
    # set layout
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
