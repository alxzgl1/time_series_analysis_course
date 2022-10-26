# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L04_main():
    
    # init
    N = 100
    
    # AR model parameters
    a = np.array([0.5, -0.3])  # model coefficients  
    p = len(a)                 # model order
    c = 0                      # constant
    e = np.random.randn(N)     # noise
    e = e / np.max(e)
    
    # data
    X1 = np.zeros(N)
    X2 = np.zeros(N)

    # over samples
    for n in range(p, N):
        # loop
        SUM = 0
        for i in range(0, p):
            SUM += a[i] * X1[n-i-1]
        X1[n] = c + SUM + e[n]
        # sum
        X2[n] = c + np.sum(a * X2[np.arange((n-1), (n-p-1), -1)]) + e[n]
        
    # plot results
    plt.plot(e, 'k', linewidth=1, label="noise")
    plt.plot(X1, 'r',  linewidth=1, label='AR(2) process')
    plt.plot(X2, 'b-.', linewidth=1, label='AR(2) process')
    plt.xlim(0, N)
    plt.ylim(-2, 2)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L04_main()
