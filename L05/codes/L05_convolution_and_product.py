# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.fftpack import fft, ifft

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def my_convolution(x, w):
    
    # init
    N = len(x)
    M = len(w)

    # add zeros
    x = np.concatenate((np.zeros(M-1), x, np.zeros(M-1)))
    y = np.zeros(N+M-1)
    
    # over samples
    for n in range(0, (N+M-1)):
        y[n] = np.sum(x[n:(n + M)] * w[::-1]) # convolution
      
    return y

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L05_main():
    
    # sampling parameters
    N = 500  # data length
    M = 50    # window length
    
    # signal
    x = np.random.randn(N)
    # window
    w = signal.gaussian(M, 5)
    
    # convolution
    y = signal.convolve(x, w)
    u = my_convolution(x, w)

    # Fourier transform
    nFFT = N+M-1
    fx = fft(x, nFFT)
    fw = fft(w, nFFT)
    fy = np.real(ifft(fx * fw))

    # figure 1
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(M, N+M, N), x, 'k')
    plt.plot(2 * w, 'g')
    plt.plot([M, M], [-3, 3], 'k--', linewidth=0.5)
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.xlim(0, N+M-1)
    plt.ylim(-4, 6)
    plt.legend(['data', 'gausssian'], loc='upper right')
    
    plt.subplot(2, 1, 2)
    plt.plot(x, 'k')
    plt.plot(0.5 * y, 'r')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.xlim(0, N+M-1)
    plt.ylim(-4, 6)
    plt.legend(['data', 'conv'], loc='upper right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    # figure 2
    plt.figure(2)
    
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * y, 'r')
    plt.plot(0.5 * u + 0.2, 'b')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.xlim(0, N+M-1)
    plt.ylim(-4, 6)
    plt.legend(['conv', 'my_conv'], loc='best')
    
    plt.subplot(2, 1, 2)
    plt.plot(0.5 * y, 'r')
    plt.plot(0.5 * fy + 0.2, 'g')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.xlim(0, N+M-1)
    plt.ylim(-4, 6)
    plt.legend(['conv', 'fft'], loc='best')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
