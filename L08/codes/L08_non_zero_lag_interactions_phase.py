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
    np.random.seed(21)
    S0 = np.random.randn(1, N) 
    S1 = np.random.randn(1, N) 
    S2 = np.random.randn(1, N)
    S = np.concatenate((S0, S1, S2))
    fl = 8.0
    fh = 12.0
    [b, a] = signal.butter(4, [fl / (fs/2), fh / (fs/2)], 'bandpass')
    L = int(fs / fl)
    M = np.shape(S)[0]
    S = signal.filtfilt(b, a, np.concatenate((np.zeros((M, L)), S, np.zeros((M, L))), axis=1))
    S = S[:, L:(N+L)]
    
    # mixing matrix
    A = np.array([[1.0, 0.0, 2.0], \
                  [0.0, 1.0, 2.0], \
                  [0.0, 0.0, 1.0]])
    
    # mixing
    Z = np.dot(A, S) / 1.25
    X = Z[0, :]
    Y = Z[1, :]
    
    # phase
    PX = np.angle(signal.hilbert(X)) 
    PY = np.angle(signal.hilbert(Y)) 
    
    # phase-locking value
    p = np.abs(np.sum(np.exp(1j * (PX - PY))) / N)
    print(p)
    p = np.real(np.sum(np.exp(1j * (PX - PY))) / N)
    print(p)
    p = np.imag(np.sum(np.exp(1j * (PX - PY))) / N)
    print(p)
    
 
    # plot
    PCOS = np.cos(PX - PY)
    PSIN = np.sin(PX - PY)
    PX = PX / (4 * np.pi)
    PY = PY / (4 * np.pi)
    
    plt.figure(1)
    
    plt.subplot(2, 1, 1)
    plt.plot(t, X, 'k')
    plt.plot(t, Y + 1, 'r')
    plt.xlim(1, 3)
    plt.ylim(-0.75, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('phase (X, Y)') 

    plt.subplot(2, 1, 2)
    plt.plot(t, X, color=(0.7, 0.7, 0.7), linewidth=0.75)
    plt.plot(t, PX, 'k')
    plt.plot(t, Y + 1, color=(0.7, 0.7, 0.7), linewidth=0.75)
    plt.plot(t, PY + 1, 'r')
    plt.xlim(1, 3)
    plt.ylim(-0.75, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('phase (X, Y)') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    plt.figure(2)

    plt.subplot(2, 1, 1)
    plt.plot(t, PX, 'k')
    plt.plot(t, PY + 1, 'r')
    plt.xlim(1, 3)
    plt.ylim(-0.75, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('phase (X, Y)') 
    plt.grid(color=(0.7,0.7,0.7), linestyle='--', linewidth=0.5)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, PCOS, 'b')
    plt.plot(t, PSIN, 'g')
    plt.xlim(1, 3)
    #plt.ylim(-0.75, 1.5)
    plt.xlabel('time (s)')
    plt.ylabel('cos, sin') 
    plt.legend(['cos', 'sin'], loc='best')
    plt.grid(color=(0.7,0.7,0.7), linestyle='--', linewidth=0.5)
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L08_main()
