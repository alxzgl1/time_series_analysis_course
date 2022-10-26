# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L04_main():
    
    # sampling parameters
    N = 100
    
    # random data
    x = np.random.randn(N)
    
    # linear temporal dependency (via moving average)
    y = np.zeros(N)
    for i in range(2, N):
        y[i] = 0.5 * x[i] - 0.2 * x[i-1] + 0.1 * x[i-2] 

    # linear approximation
    p = np.polyfit(x, y, 1) 
    fy = p[0] * x + p[1]
    
    # non-linear temporal dependency
    u = np.zeros(N)
    for i in range(2, N):
        u[i] = 0.5 * x[i] ** 2 - 0.2 * x[i-1] ** 3 + 0.1 * x[i-2] ** 4
    
    # linear approximation
    p = np.polyfit(x, u, 1) 
    fu = p[0] * x + p[1]
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(x, linewidth=1)
    plt.plot(y, linewidth=1)
    plt.xlim(0, N)
    plt.xlabel('samples')
    plt.ylabel('x, y')
    
    plt.subplot(2, 2, 2)
    plt.plot(x, y, 'r.')
    plt.plot(x, fy, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.subplot(2, 2, 3)
    plt.plot(x, linewidth=1)
    plt.plot(u, linewidth=1)
    plt.xlim(0, N)
    plt.xlabel('samples')
    plt.ylabel('x, u')
    
    plt.subplot(2, 2, 4)
    plt.plot(x, u, 'r.')
    plt.plot(x, fu, 'k', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('u')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L04_main()
