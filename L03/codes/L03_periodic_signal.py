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
    
    # signal parameters
    A1   = 1 # signal amplitude
    f1   = 5 # signal frequency, in Hz
    phi1 = 0 # signal phase
    
    A2   = 1 # signal amplitude
    f2   = 5 # signal frequency, in Hz
    phi2 = np.pi / 2 # signal phase
    
    # time variable
    t = np.linspace(0, T, N)
    
    # signal
    x1 = A1 * np.sin(2 * np.pi * f1 * t + phi1)
    x2 = A2 * np.sin(2 * np.pi * f2 * t + phi2)
    
    # plot
    plt.plot(t, x1)
    plt.plot(t, x2)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
