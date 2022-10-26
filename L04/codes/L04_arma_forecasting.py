# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARMA

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L04_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    L = int(0.50 * N) # 50% to fit model and 50% to forecast 

    # time variable
    t = np.linspace(0, T, N)
    
    # signal
    A = 0.05 
    X = np.sin(2 * np.pi * 5 * t) + A * np.random.randn(N)
    
    # split dataset
    x = X[:L] # data to fit
    y = X[L:] # data to test
    
    # autoregressive model
    p = 2 # AR model order
    q = 2 # MA model order
    model = ARMA(x, (p, q))
    model_fit = model.fit() 
    
    # make predictions
    start = len(x)
    stop  = len(x) + len(y) - 1
    u = model_fit.predict(start, stop) # Kalman filter 
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t[:start], x, 'k', linewidth=2, label="original")
    plt.plot(t[start:(stop + 1)], y, 'k-.', linewidth=2, label='expected')
    plt.plot(t[start:(stop + 1)], u, 'r', linewidth=1, label='predicted')
    plt.plot([t[start], t[start]], [-1, 1], 'r--', linewidth=0.5)
    plt.xlim(t[0], t[-1])
    plt.ylim(-2, 2)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L04_main()
