# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def get_sequence(min_len, threshold, N):
    x = np.zeros(N)
    r = np.random.rand(N)
    count = 0
    state = 0
    for i in range(0, N):
        
        if count > min_len:
            if r[i] > threshold:
                state = 1 - state
                count = 0
        count = count + 1
        x[i] = state
    return x

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L10_main():
    
    # sampling parameters
    fs = 200    # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    M = 2       # channels
    
    # time variable
    t = np.linspace(0, T, N)
    
    # generate data
    np.random.seed(5)
    X = np.random.randn(M, N)
    y = get_sequence(5, 0.8, N) # e.g., stimulus type
    
    # induce some correlation between X and y
    w = 1.0
    X = X + w * np.tile(y, (M, 1))
    X = X * 0.5
    
    # divide data into 2 parts: training and testing datasets
    L = N // 2
    Y = y[:L] # training labels
    U = y[L:] # testing labels
    XY = X[:, :L] # training data
    XU = X[:, L:] # testing data
    TY = t[:L]
    TU = t[L:]
    
    # train classifier 
    model = SVC(kernel='linear')
    model.fit(XY.T, Y)
    
    # classifier outcome
    coef = model._get_coef()
    intercept = model.intercept_
    
    # what is decision function?
    # Z = np.zeros(N)
    # for i in range(0, N):
    #     Z[i] = np.sum(X[:, i] * coef) + intercept
        
    # decision function
    z = model.decision_function(X.T)

    # test
    v = U
    u = model.predict(XU.T)
    u = u > 0.5
    
    # accuracy
    a = np.mean(v == u)
    print('accuracy: %1.2f' % (a))
    
    # plot
    b_fig_1 = 0
    if b_fig_1 == 1:
        plt.subplot(2, 2, 1)
        for i in range(0, M):
            plt.plot(t, X[i, :] + 5*i, 'k')
        plt.plot(TY, Y * 2 + 5*M, linestyle='-.', color=(0.8,0.8,0.8))
        plt.plot(TY[Y == 0], Y[Y == 0] * 2 + 5*M , 'b.')
        plt.plot(TY[Y == 1], Y[Y == 1] * 2 + 5*M , 'g.')
        plt.plot(TU, U * 2 + 5*M, linestyle='-.', color=(0.8,0.8,0.8))
        plt.plot([t[L], t[L]], [-5, 15], 'k--', linewidth=1.0)
        plt.xlim(0, T)
        plt.ylim(-2, 15)
        plt.xlabel('time')
        plt.ylabel('amplitude / response')
    
        plt.subplot(2, 2, 2)
        plt.plot(XY[0, Y == 0], XY[1, Y == 0], 'b.')
        plt.plot(XY[0, Y == 1], XY[1, Y == 1], 'g.')
        plt.xlabel('X0 (train)')
        plt.ylabel('X1 (train)')
        plt.legend(['class 0', 'class 1'], loc='upper left')
        
        plt.subplot(2, 2, 3)
        plt.plot(XU[0, :], XU[1, :], color=(0.7,0.7,0.7), marker='.', linestyle='')
        plt.xlabel('X0 (test)')
        plt.ylabel('X1 (test)')
        plt.legend(['class 0/1'], loc='upper left')
        
        plt.subplot(2, 2, 4)
        plt.plot(XU[0, U == 0], XU[1, U == 0], 'b.')
        plt.plot(XU[0, U == 1], XU[1, U == 1], 'g.')
        # create grid to evaluate model
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 10)
        yy = np.linspace(ylim[0], ylim[1], 10)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        # plot decision boundary and margins
        Z = model.decision_function(xy).reshape(XX.shape)
        plt.contour(XX, YY, Z, colors='m', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
        # plot support vectors
        plt.plot(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 'ko')
        plt.show()
        plt.xlabel('X0 (test)')
        plt.ylabel('X1 (test)')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
    
    # plot
    b_fig_2 = 1
    if b_fig_2 == 1:
        plt.subplot(2, 1, 1)
        plt.plot(t, z, 'm')
        plt.plot([t[L], t[L]], [-5, 5], 'k--', linewidth=1.0)
        plt.plot(t[L:], np.zeros(L), 'g')
        plt.xlim(0, T)
        plt.ylim(-5, 5)
        plt.xlabel('time')
        plt.ylabel('decision function')
    
        plt.subplot(2, 1, 2)
        plt.plot(TY, Y, 'r')
        plt.plot(TU, U, linestyle='-.', color=(0.8,0.8,0.8))
        plt.plot([t[L], t[L]], [-0.1, 1.1], 'k--', linewidth=1.0)
        plt.plot(t[L:], u, 'r.')
        plt.xlim(0, T)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('time')
        plt.ylabel('response')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
    
        plt.subplot(2, 2, 2)
        plt.plot(XY[0, Y == 0], XY[1, Y == 0], 'b.')
        plt.plot(XY[0, Y == 1], XY[1, Y == 1], 'g.')
        plt.xlabel('X0 (train)')
        plt.ylabel('X1 (train)')
        plt.legend(['class 0', 'class 1'], loc='upper left')
        
        plt.subplot(2, 2, 3)
        plt.plot(XU[0, :], XU[1, :], color=(0.7,0.7,0.7), marker='.', linestyle='')
        plt.xlabel('X0 (test)')
        plt.ylabel('X1 (test)')
        plt.legend(['class 0/1'], loc='upper left')
        
        plt.subplot(2, 2, 4)
        plt.plot(XU[0, U == 0], XU[1, U == 0], 'b.')
        plt.plot(XU[0, U == 1], XU[1, U == 1], 'g.')
        # create grid to evaluate model
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 10)
        yy = np.linspace(ylim[0], ylim[1], 10)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        # plot decision boundary and margins
        Z = model.decision_function(xy).reshape(XX.shape)
        plt.contour(XX, YY, Z, colors='m', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
        # plot support vectors
        plt.plot(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 'ko')
        plt.show()
        plt.xlabel('X0 (test)')
        plt.ylabel('X1 (test)')
        
        
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
