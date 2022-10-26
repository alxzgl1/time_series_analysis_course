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
    T  = 4      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # time variable
    t = np.linspace(0, T, N)
    
    # smooth signal
    np.random.seed(9)
    S0 = np.random.randn(1, N) * 2.0
    S1 = np.random.randn(1, N) * 2.0
    S = np.concatenate((S0, S1))
    fl = 8.0
    fh = 12.0
    [b, a] = signal.butter(4, [fl / (fs/2), fh / (fs/2)], 'bandpass')
    L = int(fs / fl)
    M = np.shape(S)[0]
    S = signal.filtfilt(b, a, np.concatenate((np.zeros((M, L)), S, np.zeros((M, L))), axis=1))
    S = S[:, L:(N+L)]
    
    # mixing matrix
    A = np.array([[1.0, 0.7], \
                  [0.3, 1.0,]])
    
    # mixing
    Z = np.dot(A, S)
    X = Z[0, :] * 0.75
    Y = Z[1, :] * 0.75
    
    # amplitude
    AX = np.abs(signal.hilbert(X))
    AY = np.abs(signal.hilbert(Y))
    
    # linear relationship between AX and AY
    p = np.polyfit(AX, AY, 1) 
    AU = p[0] * AX + p[1]
    
    # linear relationship between AX and AY
    p = np.polyfit(X, Y, 1) 
    U = p[0] * X + p[1]
    
    # correlation
    r = np.corrcoef(AX, AY)[0, 1]
    print(r)
    r = np.corrcoef(X, Y)[0, 1] 
    print(r)
 
    # phase
    # PX = np.angle(signal.hilbert(X)) / (4 * np.pi)
    # PY = np.angle(signal.hilbert(Y)) / (4 * np.pi)

    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, X, color=(0.7, 0.7, 0.7), linewidth=0.75)
    plt.plot(t, AX, 'k')
    plt.plot(t, Y + 1, color=(0.7, 0.7, 0.7), linewidth=0.75)
    plt.plot(t, AY + 1, 'r')
    plt.xlim(0, T)
    plt.ylim(-0.75, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude (X, Y)') 

    plt.subplot(2, 2, 3)
    plt.plot(AX, AY, 'g.')
    plt.plot(AX, AU, 'b')
    plt.xlim(-0.1, 0.6)
    plt.ylim(-0.1, 0.6)
    plt.xlabel('amplitude (X)')
    plt.ylabel('amplitude (Y)') 
    
    plt.subplot(2, 2, 4)
    plt.plot(X, Y, color=(0.7, 0.7, 0.7), marker='.')
    plt.plot(X, U, color=(0.3, 0.3, 0.3))
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    plt.xlabel('X')
    plt.ylabel('Y') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L08_main()
