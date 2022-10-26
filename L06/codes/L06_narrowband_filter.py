# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft
from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L06_main():
    
    # sampling parameters
    fs = 100   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    nFFT = T * fs   # resolution
    
    # time variable
    t = np.linspace(0, T, N)
    f = np.linspace(0, fs, nFFT)
    
    # signal
    x = 0.4 * np.sin(2 * np.pi * 8 * t)  + \
        0.3 * np.sin(2 * np.pi * 10 * t) + \
        0.2 * np.sin(2 * np.pi * 12 * t)
    
    # design filter
    fl = 9.5
    fh = 10.5
    
    # band-pass filter
    n = 8
    [bb, ab] = signal.butter(n, [fl/(fs/2), fh/(fs/2)], 'bandpass')
    
    # low-pass and high-pass filters
    n = 16
    [bl, al] = signal.butter(n, fh/(fs/2), 'lowpass')
    [bh, ah] = signal.butter(n, fl/(fs/2), 'highpass')

    # apply band-pass filter
    y = signal.filtfilt(bb, ab, x) 
    
    # apply low-pass and high-pass filters
    u = signal.filtfilt(bl, al, x) 
    u = signal.filtfilt(bh, ah, u) 
    
    # spectra
    X = np.abs(fft(x)) / N
    Y = np.abs(fft(y)) / N
    U = np.abs(fft(u)) / N
    
    # frequency response
    w, hb = signal.freqz(bb, ab, nFFT//2)
    w, hl = signal.freqz(bl, al, nFFT//2)
    w, hh = signal.freqz(bh, ah, nFFT//2)
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, x,  'k')
    plt.plot(t, 2 * y,  'r')
    plt.xlim(0, T)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['x', 'y'], loc='upper right')
    
    plt.subplot(2, 2, 2)
    plt.plot(f, X, 'k')
    plt.plot(f, Y, 'r')
    plt.plot(f[:nFFT//2], np.abs(hb) * 0.25, 'b-.')
    plt.xlim(5, 15)
    plt.ylim(0, 0.3)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    plt.legend(['X', 'Y', 'H'], loc='upper right')
    
    plt.subplot(2, 2, 3)
    plt.plot(t, x,  'k')
    plt.plot(t, 2 * u,  'r')
    plt.xlim(0, T)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['x', 'y'], loc='upper right')
    
    plt.subplot(2, 2, 4)
    plt.plot(f, X, 'k')
    plt.plot(f, U, 'r')
    plt.plot(f[:nFFT//2], np.abs(hl) * 0.25, 'b-.')
    plt.plot(f[:nFFT//2], np.abs(hh) * 0.25, 'b-.')
    plt.xlim(5, 15)
    plt.ylim(0, 0.3)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude')
    plt.legend(['X', 'Y', 'H'], loc='upper right')
   
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
