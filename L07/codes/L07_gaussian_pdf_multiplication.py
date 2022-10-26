# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L07_main():
    
    # true mean
    mu = 3.0
    
    # binning
    N = 1000
    xmin = -6
    xmax = 10
    b = np.linspace(xmin, xmax, N)

    # prior
    mu1 = 1.0
    sigma1 = 2.0
    p1 = norm.pdf(b, mu1, sigma1)
    p1 = p1 / np.sum(p1)
    
    # likelihood
    mu2 = 4.0
    sigma2 = 1.0
    p2 = norm.pdf(b, mu2, sigma2)
    p2 = p2 / np.sum(p2)
    
    # product
    p3 = p1 * p2
    p3 = p3 / np.sum(p3)
    mu = np.argmax(p3)
    
    # plot histogram
    plt.plot(b, p1, 'k')
    plt.plot(b, p2, 'k-.')
    plt.plot(b, p3, 'b')
    plt.plot([b[mu], b[mu]], [0, 0.008], 'r--')
    plt.xlim(xmin, xmax)
    plt.ylim(0, 0.010)
    plt.xlabel('x')
    plt.ylabel('probability')
    plt.legend(['prior', 'likelihood', 'posterior', 'mean'], loc='best')
    
    # set layout
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L07_main()
