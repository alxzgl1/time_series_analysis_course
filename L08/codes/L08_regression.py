# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

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
    
    # smooth signal
    np.random.seed(25)
    S0 = np.random.randn(1, N) * 2.0
    S1 = np.random.randn(1, N) * 2.0
    S = np.concatenate((S0, S1))
    [b, a] = signal.butter(4, [10.0/(fs/2), 20.0/(fs/2)], 'bandpass')
    S = signal.filtfilt(b, a, S)
    S0 = S[0, :]
    S1 = S[1, :]
    
    # mixing matrix
    A = np.array([[1.0, 0.7], \
                  [0.3, 1.0]])
    
    # mixing
    Z = np.dot(A, S)
    Z0 = Z[0, :]
    Z1 = Z[1, :]
 
    # what is dot? 
    # Z[0, :] =  A[0, 0] * Z[0, :] + A[0, 1] * Z[1, :]
    # Z[1, :] =  A[1, 0] * Z[0, :] + A[1, 1] * Z[1, :]
    
    # linear relationship between Z0 and Z1
    p = np.polyfit(Z0, Z1, 1) 
    Z2 = p[0] * Z0 + p[1]
    
    # correlation
    r = np.corrcoef(Z0, Z1)[0, 1]
    print(r)
    r = np.corrcoef(Z0, Z2)[0, 1] 
    print(r)
    r = np.corrcoef(Z1, Z2)[0, 1] 
    print(r)
    
    # plot
    plt.subplot(2, 2, 1)
    plt.plot(t, Z0, 'k')
    plt.xlim(0, T)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('time (s)')
    plt.ylabel('Z0') 
    
    plt.subplot(2, 2, 2)
    plt.plot(t, Z1, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('time (s)')
    plt.ylabel('Z1') 
    
    plt.subplot(2, 2, 3)
    plt.plot(Z0, Z1, 'g.')
    plt.plot(Z0, Z2, 'b')
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.xlabel('Z0')
    plt.ylabel('Z1') 
    
    plt.subplot(2, 2, 4)
    plt.plot(t, Z2, 'b')
    plt.xlim(0, T)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('time (s)')
    plt.ylabel('Z2') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L08_main()
