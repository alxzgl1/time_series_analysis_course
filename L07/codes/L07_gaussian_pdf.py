# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L07_main():
    
    # parameters
    mu = 3.0
    
    # binning
    N = 1000
    xmin = -6
    xmax = 6
    b = np.linspace(xmin+mu, xmax+mu, N)

    # gaussian pdf
    sigma = 0.5
    p1 = norm.pdf(b, mu, sigma)
    p1 = p1 / np.sum(p1)
    
    sigma = 1.0
    p2 = norm.pdf(b, mu, sigma)
    p2 = p2 / np.sum(p2)
    
    sigma = 2.0
    p3 = norm.pdf(b, mu, sigma)
    p3 = p3 / np.sum(p3)
    
    # plot histogram
    plt.plot(b, p1, 'k')
    plt.plot(b, p2, 'b')
    plt.plot(b, p3, 'g')
    plt.plot([mu, mu], [0, 0.01], 'r-.')
    plt.xlim(xmin+mu, xmax+mu)
    plt.ylim(0, 0.010)
    plt.xlabel('x')
    plt.ylabel('probability')
    plt.legend(['sigma=0.5', 'sigma=1.0', 'sigma=2.0', 'mean'], loc='best')
    
    # set layout
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L07_main()
