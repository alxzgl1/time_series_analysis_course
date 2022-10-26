# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L02_main():

    from scipy.fftpack import fft
    # Number of sample points
    N = 600
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N*T, N)
    y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    
    # plot
    plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
    plt.grid()
    plt.show()

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L02_main()
