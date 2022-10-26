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
    
    # mean value
    x_mu = 0.0
    y_mu = 0.0
    for i in range(0, N):
        x_mu += x[i] / N
        y_mu += y[i] / N
    
    # standard deviation 
    x_sd = 0.0
    y_sd = 0.0
    for i in range(0, N):
        x_sd += ((x[i] - x_mu) ** 2) / N
        y_sd += ((y[i] - y_mu) ** 2) / N
    x_sd = np.sqrt(x_sd)
    y_sd = np.sqrt(y_sd)
    
    # correlation coefficient
    r = 0.0
    for i in range(0, N):
        r += ((x[i] - x_mu) * (y[i] - y_mu)) / (x_sd * y_sd) / N
        
    # correlation coefficient using numpy
    r_np = np.corrcoef(x, y)[0, 1]
         
    # print
    print([r, r_np])
        
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
