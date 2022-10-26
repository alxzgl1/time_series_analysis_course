# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def get_chain(p, N):
    x = np.zeros(N)
    state = 0
    for i in range(0, N):
        if np.random.rand(1) > p[state, state]:
            state = 1 - state
        x[i] = state
    return x

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L11_main():
    
    # sampling parameters
    fs = 200    # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # time variable
    t = np.linspace(0, T, N)
    
    # generate data
    np.random.seed(5)
    P00 = 0.9
    P11 = 0.3
    p = np.array([[P00, (1 - P00)], [(1 - P11), P11]])
    x = get_chain(p, N) 
    
    P00 = 0.9
    P11 = 0.9
    p = np.array([[P00, (1 - P00)], [(1 - P11), P11]])
    y = get_chain(p, N) 
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.xlim(0, T)
    plt.xlabel('time')
    plt.ylabel('amplitude')
    #plt.title('p$_{00}$ = 0.9, p$_{11}$  = 0.3') 
    
    plt.subplot(2, 1, 2)
    plt.plot(t, y)
    plt.xlim(0, T)
    plt.xlabel('time')
    plt.ylabel('amplitude')
    #plt.title('p$_{00}$ = 0.9, p$_{11}$  = 0.9') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L11_main()
