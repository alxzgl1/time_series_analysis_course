# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

#------------------------------------------------------------------------------
# Reference:
# Granger, C.W.J., 1969. "Investigating causal relations by econometric
#     models and cross-spectral methods". Econometrica 37 (3), 424438.
# Original MATLAB code:
#    Chandler Lutz, UCR 2009
# Problem:
#    Does Y Granger Cause X?
#------------------------------------------------------------------------------
def granger_causality(x, y, alpha, max_lag):
 
    # fit restricted RSS model
    x_lag, RSS_R = fit_restricted_rss_model(x, max_lag)
    
    # fit full RSS model
    y_lag, RSS_F = fit_full_rss_model(x, y, x_lag, max_lag)
    
    # compare models
    F_stat, F_crit = compare_rss_models(x_lag, y_lag, RSS_R, RSS_F, len(x), alpha)
      
    return F_stat, F_crit, (y_lag+1)

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def fit_restricted_rss_model(x, max_lag):
    
    # init
    T = len(x)
    
    # RSS (restricted model), residual sum of squares
    RSS_R = np.zeros(max_lag)
    BIC = np.zeros(max_lag) # model comparison
    
    # over lags
    for i in range(1, (max_lag+1)):
        
        # init Y from x[i:], init X by ones(:,1) and zeros(:,i)
        Y = x[i:T]
        X = np.concatenate((np.ones((T-i, 1)), np.zeros((T-i, i))), axis=1)

        # populate X matrix with corresponding vectors of lags of x
        for j in range(1, (i+1)):
            X[:, j] = x[(i-j):(T-j)]
    
        # compute residuals
        b = np.linalg.lstsq(X, Y)
        r = Y - np.dot(X, b[0])
        
        # compute the bayesian information criterion
        BIC[i-1] = T*np.log(np.cov(r)*((T-2)/T)) + (i+1)*np.log(T)

        # init model
        RSS_R[i-1] = np.cov(r)*(T-2)
        
    # get best model
    x_lag = np.argmin(BIC)
    
    return x_lag, RSS_R
    
#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def fit_full_rss_model(x, y, x_lag, max_lag):
    
    # init
    T = len(x)
    
    # RSS (full model) 
    RSS_F = np.zeros(max_lag)
    BIC = np.zeros(max_lag)
    
    # over lags
    for i in range(1, (max_lag+1)):
        
        # init Y from x[i+x_lag:], init X by ones(:,1) and zeros(:,i)
        Y = x[(i+x_lag):T]
        X = np.concatenate((np.ones((T-(i+x_lag), 1)), np.zeros((T-(i+x_lag), x_lag+i))), axis=1)
        
        # populate X matrix with corresponding vectors of lags x
        for j in range(1, (x_lag+1)):
            X[:, j] = x[(i+x_lag-j):(T-j)]
        
        # populate X matrix with corresponding vectors of lags y
        for j in range(1, (i+1)):
            X[:, (x_lag+j)] = y[(i+x_lag-j):(T-j)]

        # compute residuals
        b = np.linalg.lstsq(X, Y)
        r = Y - np.dot(X, b[0])
        
        # compute the bayesian information criterion
        BIC[i-1] = T*np.log(np.cov(r)*((T-2)/T)) + (i+1)*np.log(T)
        
        # init model
        RSS_F[i-1] = np.cov(r)*(T-2)
        
    # get best model 
    y_lag = np.argmin(BIC)
    
    return y_lag, RSS_F

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def compare_rss_models(x_lag, y_lag, RSS_R, RSS_F, T, alpha):
    
    F_num = ((RSS_R[x_lag] - RSS_F[y_lag]) / (y_lag + 1))
    F_den = RSS_F[y_lag] / (T - (x_lag + y_lag + 1))
    F_stat = F_num / F_den
    F_crit = stats.f.isf(alpha, (y_lag + 1), (T - (x_lag + y_lag + 1)))
    
    return F_stat, F_crit
    
#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L08_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # time variable
    t = np.linspace(0, T, N)
    np.random.seed(5)
    
    # generate signal
    lag = 5
    X = 0.1 * np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(N)
    Y = 1.0 * np.concatenate((X[lag:], X[:lag])) + 0.1 * np.random.randn(N)

    # U = np.zeros(N)          
    # for i in range(lag, N):
    #     U[i-lag] = X[i]
        
    # linear relationship between X and Y
    p = np.polyfit(X, Y, 1) 
    U = p[0] * X + p[1]
    
    # correlation
    r = np.corrcoef(X, Y)[0, 1] 
    print(r)
    
    # does y granger cause x
    M = 10
    p = np.zeros((3, M))
    for max_lag in range(1, M):
        F_stat, F_crit, y_lag = granger_causality(X, Y, 1e-10, max_lag)
        p[0, max_lag] = F_stat 
        p[1, max_lag] = F_crit 
        p[2, max_lag] = y_lag
        
    # does x granger cause y
    q = np.zeros((3, M))
    for max_lag in range(1, M):
        F_stat, F_crit, y_lag = granger_causality(Y, X, 1e-10, max_lag)
        q[0, max_lag] = F_stat 
        q[1, max_lag] = F_crit 
        q[2, max_lag] = y_lag
 
    # plot
    plt.figure(1)
    
    plt.subplot(2, 2, 1)
    plt.plot(t, X, 'k')
    plt.plot(t, Y + 0.9, 'r')
    plt.xlim(0, T)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('X, Y') 
    
    plt.subplot(2, 2, 2)
    plt.plot(X, Y, 'g.')
    plt.plot(X, U, 'b')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(2, 2, 3)
    plt.plot(p[0, :], 'b')
    plt.plot(p[1, :], 'g-.')
    plt.xlim(0, M)
    # plt.grid(color=(0.7,0.7,0.7), linestyle='--', linewidth=0.5)
    plt.xlabel('lags')
    plt.ylabel(r'F(Y$\leftarrow$X)')
    plt.legend([r'F$_{stat}$', r'F$_{crit}$'], loc='best')
    
    plt.subplot(2, 2, 4)
    plt.plot(q[0, :], 'b')
    plt.plot(q[1, :], 'g-.')
    plt.xlim(0, M)
    # plt.grid(color=(0.7,0.7,0.7), linestyle='--', linewidth=0.5)
    plt.xlabel('lags')
    plt.ylabel(r'F(X$\leftarrow$Y)')
    plt.legend([r'F$_{stat}$', r'F$_{crit}$'], loc='best')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L08_main()
