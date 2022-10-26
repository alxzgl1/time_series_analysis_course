# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from spectrum import arburg, arma2psd

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L04_main():
    
    # sampling parameters
    fs = 1000    # sampling rate, in Hz
    T  = 1       # duration, in seconds
    N  = T * fs  # duration, in samples
    NFFT = fs
    
    # time variable
    t = np.linspace(0, T, N)
    f = np.linspace(0, fs, NFFT)
    
    # signal
    SNR = 0.1 
    x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 150 * t) + SNR * np.random.randn(N)

    # estimate AR model
    p = 8
    AR, P, k = arburg(x, p)
    
    # compute power spectrum
    PSD = arma2psd(AR, NFFT=NFFT)
    PSD = PSD / np.sum(PSD)

    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x, 'k')
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    
    plt.subplot(2, 1, 2)
    plt.plot([0, 0])
    plt.plot(f[:NFFT/4], PSD[:NFFT/4])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('power')
     
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L04_main()
