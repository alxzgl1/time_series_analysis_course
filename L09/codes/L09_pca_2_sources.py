# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L10_main():
    
    # parameters
    fs = 1000
    T = 1
    N = T * fs
    t = np.linspace(0, T, N)
    np.random.seed(5)
    
    # sources
    M = 2
    S0 = np.sin(2 * np.pi * 7 * t)
    S1 = np.random.randn(N)
    
    S = np.zeros((M, N))
    S[0, :] = S0 / np.std(S0)
    S[1, :] = S1 / np.std(S1)

    # mixing matrix
    A = np.array([[0.6, 0.4], \
                  [0.4, 0.6]])

    # mix sources
    X = np.dot(A, S) # mixing
    
    # subtract mean
    X = X - np.tile(np.mean(X, axis=1), (N, 1)).transpose()
    
    X0 = X[0, :]
    X1 = X[1, :]
    
    # scilearn
    # pca = PCA()
    # H = pca.fit_transform(X)
    
    # covariance
    C = np.cov(X)
    
    # eigen-decomposition
    [D, V] = np.linalg.eigh(C)
    
    # compute explained variance by components
    explained_variance = (D ** 2) / (N - 1)
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    
    print(explained_variance_ratio)
    
    # print(D)
    # print(V)
    
    # project data into orth space
    W = np.dot(np.diag(D), V.T)
    # Z = np.dot(V, X) 
    Z = np.dot(W, X)
    Z0 = Z[0, :]
    Z1 = Z[1, :]
    
    # plot
    b_pca_time = 0
    if b_pca_time == 1:
        plt.subplot(2, 1, 1)
        plt.plot(t, X0, 'k')
        plt.plot(t, X1 + 3, 'r')
        plt.xlim(0, T) 
        plt.xlabel('time (s)')
        plt.ylabel('amplitude') 
        
        plt.subplot(2, 1, 2)
        plt.plot(t, Z0, 'k')
        plt.plot(t, Z1 + 3, 'r')
        plt.xlim(0, T) 
        plt.xlabel('time (s)')
        plt.ylabel('amplitude') 
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
        
    b_pca_scatter = 0
    if b_pca_scatter == 1:
        plt.subplot(2, 2, 1)
        plt.plot(X0, X1, '.') 
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xlabel('amplitude (X0)')
        plt.ylabel('amplitude (X1)') 
        
        plt.subplot(2, 2, 2)
        plt.plot(X0, X1, '.') 
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.plot([0, V[0, 0]], [0, V[1, 0]], color='r') 
        plt.plot([0, V[0, 1]], [0, V[1, 1]], color='r')  
        plt.xlabel('amplitude (X0)')
        plt.ylabel('amplitude (X1)') 
        
        plt.subplot(2, 2, 3)
        plt.plot(X0, X1, '.') 
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.plot([0, V[0, 0] * D[0] * 2.5], [0, V[1, 0] * D[0] * 2.5], color='r') 
        plt.plot([0, V[0, 1] * D[1] * 2.5], [0, V[1, 1] * D[1] * 2.5], color='r') 
        plt.xlabel('amplitude (X0)')
        plt.ylabel('amplitude (X1)') 

        plt.subplot(2, 2, 4)
        plt.plot(Z0, Z1, '.') 
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.plot([0, 1 * D[0] * 2.5], [0, 0], color='r') 
        plt.plot([0, 0], [0, -1 * D[1] * 2.5], color='r') 
        plt.xlabel('amplitude (Z0)')
        plt.ylabel('amplitude (Z1)') 

        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
    
    b_explained_var = 1
    if b_explained_var == 1:
        plt.subplot(2, 1, 1)
        plt.plot(t, Z0, 'k')
        plt.plot(t, Z1 + 3, 'r')
        plt.xlim(0, T) 
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        
        plt.subplot(2, 2, 3)
        plt.plot(Z0, Z1, '.') 
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.plot([0, 1 * D[0] * 2.5], [0, 0], color='r') 
        plt.plot([0, 0], [0, -1 * D[1] * 2.5], color='r') 
        plt.xlabel('amplitude (Z0)')
        plt.ylabel('amplitude (Z1)') 
        
        plt.subplot(2, 2, 4)
        indices = np.arange(0, M)
        plt.bar(indices, explained_variance_ratio)
        plt.xticks(indices, ('var(Z0)', 'var(Z1)'))
        
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
        
        return
        
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L10_main()
