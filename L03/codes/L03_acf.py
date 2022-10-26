# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L03_main():
    
    # generate gaussian noise
    N = 100
    x = np.random.randn(N)
    
    # smooth signal by 4 neighboring points
    y = signal.filtfilt(np.ones(4) / 4, 1, x)

    # compute ACF
    rx = signal.correlate(x, x)
    rx = rx / np.max(rx)
    ry = signal.correlate(y, y)
    ry = ry / np.max(ry)
    
    # plot gaussian noise
    plt.subplot(2, 2, 1) 
    ax = plt.plot(x)
    plt.setp(ax, color='k', linewidth=0.5)
    plt.xlim(0, N)
    
    plt.subplot(2, 2, 2) 
    ax = plt.plot(rx)
    plt.setp(ax, color='k', linewidth=0.5)
    plt.xlim(0, 2*N-1)
    plt.ylim(-0.25, 1.25)
    
    # plot smoothed noise
    plt.subplot(2, 2, 3) 
    ax = plt.plot(y)
    plt.setp(ax, color='k', linewidth=0.5)
    plt.xlim(0, N)
    
    plt.subplot(2, 2, 4) 
    ax = plt.plot(ry)
    plt.setp(ax, color='k', linewidth=0.5)
    plt.xlim(0, 2*N-1)
    plt.ylim(-0.25, 1.25)
    
    # set layout
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
