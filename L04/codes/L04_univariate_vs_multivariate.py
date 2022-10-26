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
    
    # univariate time series (via moving avarage)
    y = np.zeros(N)
    for i in range(2, N):
        y[i] = 0.5 * x[i] - 0.2 * x[i-1] + 0.1 * x[i-2] 
    
    # multivariate time series
    u = np.zeros([N, 3])
    for i in range(2, N):
        u[i, 0] = 0.5 * x[i] 
        u[i, 1] = 0.5 * x[i] - 0.2 * x[i-1] 
        u[i, 2] = 0.5 * x[i] - 0.2 * x[i-1] + 0.1 * x[i-2]
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(y, linewidth=1)
    plt.xlim(0, N)
    plt.ylim(-4, 4)
    plt.xlabel('samples')
    plt.ylabel('y')

    plt.subplot(2, 1, 2)
    plt.plot(u[:, 2] - 2, linewidth=1)
    plt.plot(u[:, 1] + 0, linewidth=1)
    plt.plot(u[:, 0] + 2, linewidth=1)
    plt.xlim(0, N)
    plt.ylim(-4, 4)
    plt.xlabel('samples')
    plt.ylabel('u1, u2, u3')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L04_main()
