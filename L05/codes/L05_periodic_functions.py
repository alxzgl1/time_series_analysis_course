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
    A1 = 1 # signal amplitude
    f1 = 1 # signal frequency, in Hz
    
    A2 = 0.5 # signal amplitude
    f2 = 10  # signal frequency, in Hz
    
    # time variable
    t = np.linspace(0, T, N)
    
    # signal
    x1 = A1 * np.sin(2 * np.pi * f1 * t)
    y1 = A1 * np.cos(2 * np.pi * f1 * t)
    x2 = A2 * np.sin(2 * np.pi * f2 * t)
    y2 = A2 * np.cos(2 * np.pi * f2 * t)
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x1)
    plt.plot(t, y1)
    plt.xlim(0, T)
    plt.ylim(-2, 2)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['sin (1 Hz)', 'cos (1 Hz)'], loc='lower left')
    
    plt.subplot(2, 1, 2)
    plt.plot(t, x2)
    plt.plot(t, y2)
    plt.xlim(0, T)
    plt.ylim(-2, 2)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['sin (10 Hz)', 'cos (10 Hz)'], loc='lower left')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L05_main()
