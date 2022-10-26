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
    L  = int(N / 3)
    
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
    YA = np.zeros((nFFT, 3))
    YB = np.zeros((nFFT, 3))
    for i in range(0, 3):
        t1 = i * L
        t2 = (i + 1) * L
        yA = fourier_tansform(xA[t1:t2], nFFT)
        yB = fourier_tansform(xB[t1:t2], nFFT)
        YA[:, i] = np.abs(yA)
        YB[:, i] = np.abs(yB)
    
    # power spectrum
    f = np.linspace(0, fs, nFFT)
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, xA, 'r')
    plt.plot(t, xB + 2, 'b')
    plt.plot([t[L], t[L]], [-0.75, 2.75], 'k--', linewidth=0.5)
    plt.plot([t[2*L], t[2*L]], [-0.75, 2.75], 'k--', linewidth=0.5)
    plt.xlim(0, T)
    plt.ylim(-1, 3)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    
    plt.subplot(2, 2, 3)
    plt.plot(f, YA[:, 0], 'r-.')
    plt.plot(f, YA[:, 1] + 300, 'r-.')
    plt.plot(f, YA[:, 2] + 600, 'r-.')
    plt.xlim(0, 15)
    plt.ylim(-50, 900)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.subplot(2, 2, 4)
    plt.plot(f, YB[:, 0], 'b-.')
    plt.plot(f, YB[:, 1] + 300, 'b-.')
    plt.plot(f, YB[:, 2] + 600, 'b-.')
    plt.xlim(0, 15)
    plt.ylim(-50, 900)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
