# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L03_main():
    
    # sampling parameters
    N = 100
    
    # gaussian noise
    e = np.random.randn(N)
    e = e / np.max(e)
    
    # AR model
    a = [0.3, -0.5] 
    p = len(a)
    x = np.zeros(N)
    for i in range(p, N):
        x[i] = a[0] * x[i-2] + a[1] * x[i-1] + e[i]
    
    # compute ACF
    re = signal.correlate(e, e)
    re = re / np.max(re)
    re = np.concatenate((re[::-1], re[1:]))
    rx = signal.correlate(x, x)
    rx = rx / np.max(rx)
    rx = np.concatenate((rx[::-1], rx[1:]))
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(e, linewidth=1)
    plt.plot(x, linewidth=1)
    plt.xlim(0, N)
    plt.ylim(-2, 2)
    plt.xlabel('samples')
    plt.ylabel('e, x') 
    plt.legend(['noise', 'AR(2)'], loc='best')
    
    plt.subplot(2, 1, 2)
    plt.plot(re, linewidth=1)
    plt.plot(rx, linewidth=1)
    plt.xlim(0, 2*N)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('samples')
    plt.ylabel('re, rx')
    plt.legend(['noise', 'AR(2)'], loc='best')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
