# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L05_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # signal parameters
    A = 1 # signal amplitude
    f = 2 # signal frequency, in Hz
    
    # time variable
    t = np.linspace(0, T, N)
    
    # signal
    z = A * np.exp(-1j * 2 * np.pi * f * t)
    x = np.imag(z)
    y = np.real(z)
    a = np.abs(z)
    p = np.angle(z) / (2 * np.pi)
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.plot(t, y)
    plt.xlim(0, T)
    plt.ylim(-2, 2)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['imag', 'real'], loc='lower left')
    
    plt.subplot(2, 1, 2)
    plt.plot(t, a)
    plt.plot(t, p)
    plt.xlim(0, T)
    plt.ylim(-2, 2)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['abs', 'angle'], loc='lower left')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
