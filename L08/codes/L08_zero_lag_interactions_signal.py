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
    
    # amplitude
    AX = np.abs(signal.hilbert(X))
    # phase
    PX = np.angle(signal.hilbert(X)) / (4 * np.pi)
    
    # plot
    plt.plot(t, X, 'k')
    plt.plot(t, AX + 0.5, 'r')
    plt.plot(t, PX - 1, 'b')
    plt.xlim(1.0, 3.0)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude') 
    plt.legend(['signal', 'amplitude', 'phase'], loc='best')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L08_main()
