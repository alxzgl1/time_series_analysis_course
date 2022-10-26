# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L02_main():

    from scipy.interpolate import interp1d

    # generate time series
    x = np.linspace(0, 10, num=11, endpoint=True)
    y = np.cos(-x**2/9.0)
    
    # interpolate
    f = interp1d(x, y, kind='cubic')

    # new time variable
    xnew = np.linspace(0, 10, num=41, endpoint=True)
    
    # draw
    plt.plot(x, y, 'o', xnew, f(xnew), '-')
    plt.legend(['data', 'cubic'], loc='best')
    plt.show()

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L02_main()
