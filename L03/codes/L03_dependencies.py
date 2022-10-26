# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L03_main():
    
    # generate distributions
    N = 100
    x = np.random.randn(N)
    y = np.random.randn(N)
    
    # dependencies between variables
    u = (x + y) / 2           # linear
    z = (x ** 2 + y ** 2) / 4 # non-linear

    # correlation coefficient using numpy
    rxx = np.corrcoef(x, x)[0, 1]
    rxy = np.corrcoef(x, y)[0, 1]
    rxu = np.corrcoef(x, u)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    print('%1.2f, %1.2f, %1.2f, %1.2f\n' % (rxx, rxy, rxu, rxz))
    
    # linear fit
    p = np.polyfit(x, x, 1) 
    fx = p[0] * x + p[1]
    p = np.polyfit(x, y, 1)
    fy = p[0] * x + p[1]
    p = np.polyfit(x, u, 1)
    fu = p[0] * x + p[1]
    p = np.polyfit(x, z, 1)
    fz = p[0] * x + p[1]
   
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(x, x, '.')
    plt.plot(x, fx)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, y, '.')
    plt.plot(x, fy)
    
    plt.subplot(2, 2, 3)
    plt.plot(x, u, '.')
    plt.plot(x, fu)
    
    plt.subplot(2, 2, 4)
    plt.plot(x, z, '.')
    plt.plot(x, fz)
    
        
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
