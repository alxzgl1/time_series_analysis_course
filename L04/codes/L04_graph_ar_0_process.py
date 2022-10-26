# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L04_main():
    
    # sampling parameters
    N = 100
    
    # gaussian noise
    e = np.random.randn(N)
    e = e / np.max(e)
    
    # AR model
    a = []
    p = len(a)
    x = np.zeros(N)
    for i in range(p, N):
        x[i] = 0.1 + e[i]
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(e, linewidth=1)
    plt.plot(x, linewidth=1)
    plt.xlim(0, N)
    plt.ylim(-2, 2)
    plt.xlabel('samples')
    plt.ylabel('e, x') 
    plt.legend(['noise', 'AR(0)'], loc='best')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L04_main()
