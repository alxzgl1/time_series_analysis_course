# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L03_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # time variable
    t = np.linspace(0, T, N)
    
    # signal
    f = 5
    x1 = np.sin(2 * np.pi * f * t)
    x2 = np.cos(2 * np.pi * f * t)
    
    x3 = np.imag(np.exp(1j * 2 * np.pi * f * t))
    x4 = np.real(np.exp(1j * 2 * np.pi * f * t))
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x1)
    plt.plot(t, x2)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, x3)
    plt.plot(t, x4)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
