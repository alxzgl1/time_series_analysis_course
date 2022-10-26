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
    a1 = [0.1] # close to 0
    a2 = [0.9] # close to 1
    p = len(a1)
    x1 = np.zeros(N)
    x2 = np.zeros(N)
    for i in range(p, N):
        x1[i] = a1[0] * x1[i-1] + e[i]
        x2[i] = a2[0] * x2[i-1] + e[i]
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(e, linewidth=1)
    plt.plot(x1, linewidth=1)
    plt.xlim(0, N)
    plt.ylim(-2, 2)
    plt.xlabel('samples')
    plt.ylabel('e, x') 
    plt.legend(['noise', 'AR(1)'], loc='best')
    
    plt.subplot(2, 1, 2)
    plt.plot(e, linewidth=1)
    plt.plot(x2, linewidth=1)
    plt.xlim(0, N)
    plt.ylim(-2, 2)
    plt.xlabel('samples')
    plt.ylabel('e, x')
    plt.legend(['noise', 'AR(1)'], loc='best')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L04_main()
