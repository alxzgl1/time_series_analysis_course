# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L03_main():
    
    # parameters
    fs = 1000   # sampling rate, in Hz
    N  = 1000   # duration, in samples
    T  = N / fs # duration, in seconds

    # time variable
    t = np.linspace(0, T, N)
    
    # chirp signal
    f0 = 1
    f1 = 10
    t1 = T
    y = signal.chirp(t, f0, t1, f1) 
    
    # non-periodic signal
    f0 = 10
    u = np.sin(2 * np.pi * f0 * t) 
    u[int(N/4):] = 0
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, u)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
