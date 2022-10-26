# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn import mixture

from matplotlib import patches, mlab

from scipy.stats import norm

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L10_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples 
    
    # clustering parameters
    K = 2       # number of clusters
    
    # time variable
    t = np.linspace(0, T, N)
    
    # generate data
    np.random.seed(0)
    S0 = np.concatenate((np.random.randn(1, N) * 0.5 + 0, np.random.randn(1, N) * 0.5 + 0))
    S1 = np.concatenate((np.random.randn(1, N) * 1.0 + 4, np.random.randn(1, N) * 1.0 + 2))
    X = np.concatenate((S0, S1), axis=1).T
    
    # fit model
    model = mixture.GaussianMixture(n_components=K)
    model.fit(X)
    
    # model properties
    Y = model.predict(X)
    model_mu = model.means_
    model_cov = model.covariances_

    
    # plot
    b_fig_1 = 0
    if b_fig_1 == 1:
        plt.subplot(2, 2, 1)
        Z = np.concatenate((np.random.randn(1, N) * 0.5 + 0.0, np.random.randn(1, N) * 1.25 + 4.0))
        plt.plot(t, Z[0], 'k')
        plt.plot(t, Z[1], 'r')
        plt.xlim(0, T)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 2)
        plt.scatter(Z[0], Z[1], 1.0, color=(0.7,0.7,0.7))
        plt.plot([-4, 4], [np.mean(Z[1]), np.mean(Z[1])], linestyle='--', color='g', linewidth=0.7)
        plt.plot([np.mean(Z[0]), np.mean(Z[0])], [0, 8], linestyle='--', color='g', linewidth=0.7)
        plt.xlim(-4, 4)
        plt.ylim(0, 8)
        plt.xlabel('Z0')
        plt.ylabel('Z1')
        
        plt.subplot(2, 2, 3)
        b = np.linspace(-3, 10, 1000)
        p0 = norm.pdf(b, np.mean(Z[0]), np.std(Z[0])) * 4
        p1 = norm.pdf(b, np.mean(Z[1]), np.std(Z[1])) * 4
        plt.plot(b, p0, 'k')
        plt.plot(-p1 + 4, b, 'r')
        plt.plot([-4, 4], [np.mean(Z[1]), np.mean(Z[1])], linestyle='--', color='g', linewidth=0.7)
        plt.plot([np.mean(Z[0]), np.mean(Z[0])], [0, 8], linestyle='--', color='g', linewidth=0.7)
        plt.xlim(-4, 4)
        plt.ylim(0, 8)
        plt.xlabel('distribution Z0')
        plt.ylabel('distribution Z1')
        
        plt.subplot(2, 2, 4)
        delta = 0.1
        x = np.arange(-10.0, 10.0, delta)
        y = np.arange(-10.0, 10.0, delta)
        x, y = np.meshgrid(np.arange(-10.0, 10.0, delta), np.arange(-10.0, 10.0, delta))
        z = mlab.bivariate_normal(x, y, np.std(Z[0]), np.std(Z[1]), np.mean(Z[0]), np.mean(Z[1]))
        plt.scatter(Z[0], Z[1], 1.0, color=(0.7,0.7,0.7))
        plt.contour(x, y, z)
        plt.plot([-4, 4], [np.mean(Z[1]), np.mean(Z[1])], linestyle='--', color='g', linewidth=0.7)
        plt.plot([np.mean(Z[0]), np.mean(Z[0])], [0, 8], linestyle='--', color='g', linewidth=0.7)
        plt.xlim(-4, 4)
        plt.ylim(0, 8)
        plt.xlabel('Z0')
        plt.ylabel('Z1')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
    
    b_fig_2 = 1
    if b_fig_2 == 1:
        plt.subplot(2, 2, 1)
        plt.plot(X[:, 0], 'k')
        plt.plot(X[:, 1], 'r')
        plt.xlim(0, 2*N)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], 1.0, color=(0.7,0.7,0.7))
        plt.xlim(-2, 8)
        plt.ylim(-2, 8)
        plt.xlabel('X0')
        plt.ylabel('X1')
        
        s = plt.subplot(2, 2, 3)
        colors = 'bgmc'
        for i in range(0, K):
            
            # predict
            plt.scatter(X[Y == i, 0], X[Y == i, 1], 1.0, color=(0.7,0.7,0.7))
     
            # plot an ellipse to show the Gaussian component
            v, w = np.linalg.eigh(model_cov[i])
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ellipse = patches.Ellipse(model_mu[i], v[0], v[1], 180.0 + angle, color=colors[i])
            ellipse.set_clip_box(s.bbox)
            ellipse.set_alpha(0.5)
            s.add_artist(ellipse)
        plt.xlim(-2, 8)
        plt.ylim(-2, 8)
        plt.xlabel('X0')
        plt.ylabel('X1')
        
        plt.subplot(2, 2, 4)

        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
