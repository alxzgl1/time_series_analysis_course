# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L11_main():
    
    # sampling parameters
    fs = 200    # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # generate data
    np.random.seed(5)
    X = np.sort(5 * np.random.rand(N, 1), axis=0)
    y = np.sin(X).ravel()
    y[::10] += 3 * (0.5 - np.random.rand(20))
    
    # train classifier 
    model_svr = SVR(kernel='linear', C=1e3)
    model_svr.fit(X, y)
    support_vector_indices = model_svr.support_
    
    model_lin = LinearRegression()
    model_lin.fit(X, y)
    
    # test
    u = model_svr.predict(X)
    v = model_lin.predict(X)
    
    # plot
    b_fig_1 = 1
    if b_fig_1 == 1:
        plt.subplot(2, 2, 1)
        plt.plot(X, y , 'r.')
        plt.plot(X, u, linestyle='-', color='gray')
        plt.xlim(0, 5)
        plt.ylim(-3, 3)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear SVR')
    
        plt.subplot(2, 2, 2)
        plt.plot(X, y , 'g.')
        plt.plot(X, v, linestyle='-', color='gray')
        plt.xlim(0, 5)
        plt.ylim(-3, 3)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear regression')
        
        plt.subplot(2, 2, 3)
        plt.plot(X, y, 'r.')
        plt.plot(X[support_vector_indices], y[support_vector_indices], 'b.')
        plt.legend(['data', 'support vectors'], loc='best')
        plt.xlim(0, 5)
        plt.ylim(-3, 3)
        plt.xlabel('X')
        plt.ylabel('y')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return

    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L11_main()
