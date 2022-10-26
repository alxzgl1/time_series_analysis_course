# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L05_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 6      # duration, in seconds
    N  = T * fs # duration, in samples
    L  = 2 * N // T
    
    nFFT = fs   # fft resolution = fs / nFFT, in Hz
    
    # time variable
    t = np.linspace(0, T, N)
    
    # signal
    x1 = 0.5 * np.sin(2 * np.pi * 2 * t)
    x2 = 0.2 * np.sin(2 * np.pi * 5 * t)
    x3 = 0.1 * np.sin(2 * np.pi * 10 * t)
    xA = x1 + x2 + x3
    xB = np.concatenate((x1[:L], x2[:L], x3[:L]))
    
    # fourier transform
    step = 0.5
    duration = 1.0
    M = int(T / step) - 1
    YA = np.zeros((nFFT, M))
    YB = np.zeros((nFFT, M))
    for i in range(0, M):
        t1 = i * int(step * fs)
        t2 = t1 + int(duration * fs)
        yA = fft(xA[t1:t2], nFFT)
        yB = fft(xB[t1:t2], nFFT)
        YA[:, i] = np.abs(yA)
        YB[:, i] = np.abs(yB)
    
    # cut
    fmax = 20
    YA = YA[0:fmax, :]
    YB = YB[0:fmax, :]
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, xA, 'r')
    plt.plot(t, xB + 2, 'b')
    for i in range(1, M+1):
        plt.plot([i * t[int(step * fs)], i * t[int(step * fs)]], [-0.75, 2.75], 'k--', linewidth=0.5)
    plt.xlim(0, T)
    plt.ylim(-1, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    # image
    plt.subplot(2, 2, 3)
    plt.imshow(YA, extent=[0, M, 0, fmax], aspect='auto')
    plt.xlabel('windows')
    plt.ylabel('frequency (Hz)') 
    plt.gca().set_yticklabels([20, 15, 10, 5, 0])
    plt.colorbar()
    plt.show()
    
    # image
    plt.subplot(2, 2, 4)
    plt.imshow(YB, extent=[0, M, 0, fmax], aspect='auto')
    plt.xlabel('windows')
    plt.ylabel('frequency (Hz)') 
    plt.gca().set_yticklabels([20, 15, 10, 5, 0])
    plt.colorbar()
    plt.show()
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
