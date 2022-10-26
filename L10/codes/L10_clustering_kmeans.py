# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

from sklearn import cluster

#------------------------------------------------------------------------------
# Function
#------------------------------------------------------------------------------
def get_colors(M, R):
    
    colors_base = 'rgbcmyk'
    colors_base = colors_base[:M]
    colors = ''
    for i in range(0, M):
        colors += colors_base[i] * R
    
    return colors

#------------------------------------------------------------------------------
# Function, main()
# Parameters, 1) K, 2) SNR
#------------------------------------------------------------------------------
def L10_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    M  = 5      # number of sources
    R  = 3      # number of copies 
    
    # clustering parameters
    K = 5       # number of clusters
    # noise parameters
    SNR = 0.5
    
    # time variable
    t = np.linspace(0, T, N)
    
    # generate sources
    np.random.seed(5)
    S = np.zeros((M, N))
    S[0, :] = np.sin(2 * np.pi * t * 7)
    S[1, :] = signal.sawtooth(2 * np.pi * t * 5)
    S[2, :] = np.abs(np.cos(2 * np.pi * t * 3)) - 0.5
    S[3, :] = np.sign(np.sin(2 * np.pi * t * 8))
    S[4, :] = np.random.randn(N)
    S[0, :] = S[0, :] / np.max(S[0, :])
    S[1, :] = S[1, :] / np.max(S[1, :])
    S[2, :] = S[2, :] / np.max(S[2, :])
    S[3, :] = S[3, :] / np.max(S[3, :])
    S[4, :] = S[4, :] / np.max(S[4, :])
    
    # add noise
    X0 = np.tile(S[0, :], (R, 1)) + np.random.randn(R, N) * SNR
    X1 = np.tile(S[1, :], (R, 1)) + np.random.randn(R, N) * SNR
    X2 = np.tile(S[2, :], (R, 1)) + np.random.randn(R, N) * SNR
    X3 = np.tile(S[3, :], (R, 1)) + np.random.randn(R, N) * SNR
    X4 = np.tile(S[4, :], (R, 1)) + np.random.randn(R, N) * SNR
    
    # combine
    X = np.concatenate((X0, X1, X2, X3, X4))

    # re-order
    Y = X[np.random.permutation(M*R), :]

    # clustering  using sklearn
    model = cluster.KMeans(n_clusters=K)
    model.fit(Y)
    
    # clustering outcome
    labels = model.labels_
    Z = model.cluster_centers_
    inertia = model.inertia_
    print(inertia)
    
    # re-order data in accordance to labels
    # indices = np.squeeze(np.argsort(labels))
    # U = Y[indices, :]
    
    # plot
    b_fig_1 = 1
    if b_fig_1 == 1:
        colors = get_colors(M, 1)
        plt.subplot(2, 2, 1)
        for i in range(0, M):
            plt.plot(t, S[i, :] + 3 * i, color=colors[i])
        plt.xlim(0, T)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title('original')
            
        plt.subplot(2, 2, 2)
        colors = get_colors(M, R)
        for i in range(0, M*R):
            plt.plot(t, X[i, :] + 3 * i, color=colors[i])
        plt.xlim(0, T)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title('copies')
            
        plt.subplot(2, 2, 3)
        colors = get_colors(M, R)
        for i in range(0, M*R):
            plt.plot(t, Y[i, :] + 3 * i, color=colors[i])
        plt.xlim(0, T)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title('measurements')
        
        plt.subplot(2, 2, 4)
        colors = get_colors(K, 1)
        for i in range(0, K):
            plt.plot(t, Z[i, :] + 3 * i, color=colors[i])
        plt.xlim(0, T)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title('clusters')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
    
    b_fig_2 = 1
    if b_fig_2 == 1:
        colors = get_colors(M, 1)
        plt.subplot(2, 2, 1)
        for i in range(0, M):
            plt.plot(t, S[i, :] + 3 * i, color=colors[i])
        plt.xlim(0, T)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title('original')
             
        plt.subplot(2, 2, 2)
        colors = get_colors(M, R)
        for i in range(0, M*R):
            plt.plot(t, Y[i, :] + 4 * i, color=colors[i])
        plt.xlim(0, T)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title('measurements')
        
        plt.subplot(2, 2, 3)
        colors = get_colors(K, 1)
        for i in range(0, K):
            plt.plot(t, Z[i, :] + 4 * i, color=colors[i])
        plt.xlim(0, T)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title('clusters')
        
        plt.subplot(2, 2, 4)
        colors = get_colors(K, 1)
        for i in range(0, K):
            plt.plot(t, Z[i, :] + 4 * i, color=colors[i])
        plt.xlim(0, T)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title('clusters')
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
