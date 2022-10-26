# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import lognorm

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L03_main():
    
    # generate data for specific probability density function (PDF)
    N = 10000
    x0 = np.random.randn(N)
    x1 = np.random.lognormal(0.5, 0.5, size=N)
    
    # plot
    plt.subplot(2, 2, 1) 
    ax = plt.plot(x0)
    plt.setp(ax, color='k', linewidth=0.5)
    plt.xlim(0, N)
    
    plt.subplot(2, 2, 3)
    ax = plt.plot(x1)
    plt.setp(ax, color='k', linewidth=0.5)
    plt.xlim(0, N)
    
    # histogram - normal
    xmin = np.min(x0) 
    xmax = np.max(x0)
    bx = np.linspace(xmin, xmax, 100)
    hx, bx = np.histogram(x0, bx)
    hx = hx / np.sum(hx)
    bx = bx[:-1] + (bx[1] - bx[0]) / 2 # get center of bins
    
    # fit histogram
    mu, std = norm.fit(x0)
    px = norm.pdf(bx, mu, std)
    px = px / np.sum(px)
    
    # plot histogram
    plt.subplot(2, 2, 2) 
    plt.bar(bx, hx, color='k', width=0.05)
    plt.plot(bx, px, 'r')
    
    # histogram - lognormal
    xmin = np.min(x1) 
    xmax = np.max(x1)
    bx = np.linspace(xmin, xmax, 100)
    hx, bx = np.histogram(x1, bx)
    hx = hx / np.sum(hx)
    bx = bx[:-1] + (bx[1] - bx[0]) / 2 # get center of bins
    
    # fit histogram
    p0, p1, p2 = lognorm.fit(x1)
    px = lognorm.pdf(bx, p0, p1, p2)
    px = px / np.sum(px)
    
    # plot histogram
    plt.subplot(2, 2, 4) 
    plt.bar(bx, hx, color='k', width=0.05)
    plt.plot(bx, px, 'r')
    
    # set layout
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
