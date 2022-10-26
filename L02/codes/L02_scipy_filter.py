# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L02_main():

    from scipy import signal
    # Number of sample points
    N = 600
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N*T, N)
    y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
    
    # butterworth filter
    fs = 1.0 / T
    b, a = signal.butter(4, [30.0 / (fs / 2), 60.0 / (fs / 2)] , 'bandpass')
    u = signal.filtfilt(b, a, y)
    
    # plot
    plt.plot(x, y)
    plt.plot(x, u)
    plt.show()

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L02_main()
