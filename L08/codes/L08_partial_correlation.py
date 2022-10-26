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
    np.random.seed(55)
    S0 = np.random.randn(1, N) * 2.0
    S1 = np.random.randn(1, N) * 2.0
    S2 = np.random.randn(1, N) * 2.0
    S = np.concatenate((S0, S1, S2))
    [b, a] = signal.butter(4, [10.0/(fs/2), 20.0/(fs/2)], 'bandpass')
    S = signal.filtfilt(b, a, S)
    
    # mixing matrix
    A = np.array([[1.0, 0.7, 0.3], \
                  [0.3, 1.0, 0.1], \
                  [0.5, 0.5, 1.0]])
    
    # mixing
    U = np.dot(A, S)
    X = U[0, :]
    Y = U[1, :]
    Z = U[2, :]
    
    # correlation
    r = np.corrcoef(X, Y)[0, 1]
    print(r)
    r = np.corrcoef(X, Z)[0, 1]
    print(r)
    r = np.corrcoef(Y, Z)[0, 1]
    print(r)
    
    # fitting and regressing out
    p = np.polyfit(Z, X, 1) 
    fX = X - p[0] * Z + p[1]
    
    p = np.polyfit(Z, Y, 1) 
    fY = Y - p[0] * Z + p[1]
    
    # correlation
    r = np.corrcoef(fX, fY)[0, 1]
    print(r)
    r = np.corrcoef(fX, Z)[0, 1]
    print(r)
    r = np.corrcoef(fY, Z)[0, 1]
    print(r)
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, X, 'b')
    plt.plot(t, Y + 1.5, 'g')
    plt.plot(t, Z + 3.0, 'k')
    plt.xlim(0, T)
    plt.ylim(-1, 4)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    # plt.legend(['X', 'Y', 'Z'], loc='best')

    plt.subplot(2, 1, 2)
    plt.plot(t, fX, 'b')
    plt.plot(t, fY + 1.5, 'g')
    plt.plot(t, Z + 3.0, 'k')
    plt.xlim(0, T)
    plt.ylim(-1, 4)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
   
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L08_main()
