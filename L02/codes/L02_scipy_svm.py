# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L02_main():

    # generate data
    N = 200
    M = 2
    X = np.random.randn(N, M)
    y = np.random.randn(N)
    y = y > 0.0
    
    # create noisy labels for data
    SNR = 1.0
    X = X + SNR * np.transpose(np.tile(y, [M, 1]))
   
    # train
    model = SVC(kernel='linear')
    model.fit(X[:100, :], y[:100])
    
    # test
    y = y[100:]
    u = model.predict(X[100:, :])
    u = u > 0.5
    
    # accuracy
    a = np.mean(y == u)
    print('accuracy: %1.2f' % (a))
    
    # plot
    plt.plot(y)
    plt.plot(u)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L02_main()
