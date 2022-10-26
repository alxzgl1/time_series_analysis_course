# -*- coding: utf-8 -*-

# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

# by Andrew D. Straw
# http://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html

# modified by A. Zhigalov (see, Haykin, 2009, Book)

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L07_main():

    N = 100 # number of observations
    
    np.random.seed(11)
    
    # parameters
    x = -0.375 # truth value / state at time n (process)
    y = np.random.normal(x, 0.1, size=N) # observations (measurements) at time n
    
    Q = 1e-5   # process variance
    R = 0.01   # measurement variance
    
    # allocate space for arrays
    xhm = np.zeros(N) # predicted estimate of the state at time n
                      # given the observations y[1], y[2], ..., y[n-1]    
    xh  = np.zeros(N) # filtered estimate of the state at time n, 
                      # given the observations y[1], y[2], ..., y[n]
    Pm = np.zeros(N)  # prediction-error covariance matrix                           
    P  = np.zeros(N)  # filtering-error covariance matrix  
    G = np.zeros(N)   # Kalman gain at time n
    
    # intial guesses
    xh[0] = 0.0
    P[0] = 1.0
    for n in range(1, N):
        
        # prediction
        xhm[n] = xh[n-1]
        Pm[n]  = P[n-1] + Q # posterior at previous step becomes a prior at current step
    
        # measurement update
        G[n] = Pm[n] / (Pm[n] + R)
        a = y[n] - xhm[n]
        xh[n] = xhm[n] + G[n] * a
        P[n] = (1 - G[n]) * Pm[n]
    
    # plot
    plt.subplot(1, 2, 1)
    plt.plot(y, 'k+')
    plt.plot(xh, 'b-')
    plt.plot([0, N], [x, x], 'g')
    plt.xlabel('observation')
    plt.ylabel('voltage')
    plt.xlim(0, N)
    plt.legend(['measurements', 'a posterior estimate', 'truth value'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(Pm)
    plt.xlabel('observation')
    plt.ylabel('voltage$^2$')
    plt.xlim(0, N)
    plt.ylim(0, 0.01)
    plt.legend(['a prior estimate'], loc='upper right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L07_main()
