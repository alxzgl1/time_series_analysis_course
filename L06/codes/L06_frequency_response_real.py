# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L06_main():
    
    # parameters
    fc = 25
    fl = 20
    fh = 30
    fs = 100
    
    # frequency variable
    f = np.linspace(0, fs, fs)
    
    # design low-pass IIR filter
    order = 8
    [b, a] = signal.butter(order, fc/(fs/2), 'low')
    [w1, h1] = signal.freqz(b, a, fs//2)
    
    # design high-pass IIR filter
    order = 8
    [b, a] = signal.butter(order, fc/(fs/2), 'high')
    [w2, h2] = signal.freqz(b, a, fs//2)
    
    # design band-pass IIR filter
    order = 8
    [b, a] = signal.butter(order, [fl/(fs/2), fh/(fs/2)], 'bandpass')
    [w3, h3] = signal.freqz(b, a, fs//2)
    
    # design stop-band IIR filter
    order = 8
    [b, a] = signal.butter(order, [fl/(fs/2), fh/(fs/2)], 'bandstop')
    [w4, h4] = signal.freqz(b, a, fs//2);
    
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(f[:fs//2], np.abs(h1), 'b')
    plt.xlim(0, fs/2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    plt.legend(['low-\npass'], loc='upper right')

    plt.subplot(2, 2, 2)
    plt.plot(f[:fs//2], np.abs(h2), 'b')
    plt.xlim(0, fs/2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    plt.legend(['high-\npass'], loc='lower right')
    
    plt.subplot(2, 2, 3)
    plt.plot(f[:fs//2], np.abs(h3), 'b')
    plt.xlim(0, fs/2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    plt.legend(['band-\npass'], loc='upper right')
    
    plt.subplot(2, 2, 4)
    plt.plot(f[:fs//2], np.abs(h4), 'b')
    plt.xlim(0, fs/2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    plt.legend(['stop-\nband'], loc='lower right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
