# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
    
#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def generate_data(process_model, measurement_model, N):
    
    # set seed
    np.random.seed(18)
    
    # variables
    X = np.zeros(N)
    Y = np.zeros(N)
    
    # process model
    dx, process_var = process_model
    process_noise = np.sqrt(process_var)
    
    # measurement model
    _, sensor_var = measurement_model
    measurement_noise = np.sqrt(sensor_var)
 
    # init 
    X[0] = 0.0 + dx + np.random.randn(1) * process_noise

    # generate data
    for i in range(1, N):
        
        # process
        X[i] = X[i-1] + dx + np.random.randn(1) * process_noise

        # measurement
        Y[i] = X[i] + np.random.randn(1) * measurement_noise

    return X, Y

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def update(prior, measurement):
    
    x, P = prior        # mean and variance of prior
    y, R = measurement  # mean and variance of measurement
    
    a = y - x           # residual
    G = P / (P + R)     # Kalman gain

    x = x + G*a         # posterior
    P = (1 - G) * P     # posterior variance
    
    return x, P

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def predict(posterior, process):
    x, P = posterior # mean and variance of posterior
    dx, Q = process  # mean and variance of process
    
    x = x + dx
    P = P + Q
    
    return x, P

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L07_main():

    # process model
    process_var = 1.0  # variance in the process
    dx = 1.0 
    process_model = (dx, process_var) 
    
    # measurement model
    sensor_var = 100.0 # variance in the sensor
    measurement_model = (0, sensor_var)

    # generate data
    N = 10
    X, Y = generate_data(process_model, measurement_model, N)
    
    # allocate space for estimates of X
    N = len(Y)
    E = np.zeros(N) 

    # initial condition
    x_mu = 0.0
    x_var = 100.0
    x = (x_mu, x_var) # Gaussian process (mean and variance)
    
    # run Kalman filter
    for i in range(0, N):
        
        # predict
        prior_mu, prior_var = predict((x_mu, x_var), (dx, process_var))
        prior = (prior_mu, prior_var) # Gaussian distribution
        
        # update 
        y_mu = Y[i] # measurement
        likelihood = (y_mu, sensor_var) # Gaussian distribution
        x_mu, x_var = update(prior, likelihood)
        
        # process estimates
        E[i] = x_mu
    
    # plot
    plt.plot(X, '.')
    plt.plot(Y, '+')
    plt.plot(E, '-.')
    plt.legend(['process (X)', 'measurements (Y)', 'estimates (E)'], loc='best')
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L07_main()
