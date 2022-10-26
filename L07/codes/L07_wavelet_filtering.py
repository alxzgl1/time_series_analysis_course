# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def init_wavelet(f0, m, fs):

    # init
    dt = 1.0 / fs
    p2 = 1.0 / (f0 / m * (2.0 * np.pi))
    p3 = int((p2 * 10) / dt)
    p3 = p3 + p3 % 2
    
    # carrier
    p4 = (np.arange(0, p3) * dt) - ((p3 * dt) / 2)
    
    # formulae
    p5 = (-1) * ((p4 ** 2) / ((p2 ** 2) * 2))
    p6 = p4 * (2 * np.pi * f0)
    
    # shape
    p7 = p5 + 1j * p6
    p8 = np.exp(p7) * (1 / np.sqrt(p2 * np.sqrt(np.pi)))
    p9 = p8 / (np.sum(np.abs(p8)) / 2)
    
    # split into halves
    p10 = int(p3 / 2)
    p11 = p9[:p10]
    p12 = p9[p10:]

    return p11, p12

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L06_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    nFFT = T * fs   # resolution
    
    # time variable
    t = np.linspace(0, T, N)
    f = np.linspace(0, fs, nFFT)
    
    # signal
    x = 0.5 * np.sin(2 * np.pi * 5 * t) + \
        0.5 * np.sin(2 * np.pi * 10 * t) + \
        0.5 * np.random.randn(N)
    X = np.abs(fft(x)) / N
    
    # init wavelet
    f0 = 10
    m = 5
    a, b = init_wavelet(f0, m, fs)
    
    # shape to draw
    L = (N - (len(a) + len(b))) // 2
    v = np.concatenate((np.zeros(L, 'complex'), a, b, np.zeros(L, 'complex')))
    V = np.abs(fft(v))
    
    # concatenate halves
    w = np.concatenate((b, np.zeros(2*L, 'complex'), a))
    
    # ifft
    y = ifft(fft(x) * fft(w))
    Y = np.abs(fft(y)) / N
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, np.real(v), 'b')
    plt.plot(t, np.imag(v), color=(1.0,0.5,0.0))
    plt.xlim(0.1, 0.9)
    # plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['real', 'imag'], loc='best')
    
    plt.subplot(2, 2, 2)
    plt.plot(f, V, 'g')
    plt.xlim(0, 30)
    # plt.ylim(0, 0.25)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    
    plt.subplot(2, 2, 3)
    plt.plot(t, x, 'k')
    plt.plot(t, y, 'r')
    plt.xlim(0, T)
    # plt.ylim(-3, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 4)
    plt.plot(f, X, 'k')
    plt.plot(f, Y, 'r')
    plt.xlim(0, 30)
    # plt.ylim(0, 0.25)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
