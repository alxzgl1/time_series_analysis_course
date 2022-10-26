# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function, direct PAC estimate (Ozkurt and Schnitzler, 2011)
#------------------------------------------------------------------------------
def modulation_index(A, P):
 
  N = len(A)
  Z = A * np.exp(1j * P)
  m = (1.0 / np.sqrt(N)) * np.abs(np.mean(Z)) / np.sqrt(np.mean(A**2))
  
  return m * 32

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L08_main():
    
    b_sine_signal = 1
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 1      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # time variable
    t = np.linspace(0, T, N)
    
    # generate signal
    if b_sine_signal == 1:
        X = np.sin(2 * np.pi * 5 * t)
        Y = np.sin(2 * np.pi * 50 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
        V = np.sin(2 * np.pi * 50 * t) * (1 + 0.5 * np.sin(2 * np.pi * 6 * t))
    else:
        # signal realistic signal
        np.random.seed(9)
        L = 200
        S = np.random.randn(N) * 2.0
        # low frequency
        [b, a] = signal.butter(4, [8.0 / (fs/2), 12.0 / (fs/2)], 'bandpass')
        X = signal.filtfilt(b, a, np.concatenate((np.zeros(L), S, np.zeros(L))))
        X = X[L:(N+L)]
        # high frequency
        [b, a] = signal.butter(4, [18.0 / (fs/2), 22.0 / (fs/2)], 'bandpass')
        Y = signal.filtfilt(b, a, np.concatenate((np.zeros(L), S, np.zeros(L))))
        Y = Y[L:(N+L)]
    
    # slow signal phase
    PX = np.angle(signal.hilbert(X))
    
    # fast signal amplitude
    AY = np.abs(signal.hilbert(Y))
    AV = np.abs(signal.hilbert(V))
    
    # phase of fast amplitude
    PA = np.angle(signal.hilbert(AY))
    PB = np.angle(signal.hilbert(AV))
    
    # L = 200
    # [b, a] = signal.butter(4, [4.0 / (fs/2), 6.0 / (fs/2)], 'bandpass')
    # PA = signal.filtfilt(b, a, np.concatenate((np.zeros(L), AY, np.zeros(L))))
    # PA = PA[L:(N+L)]
    # PA = np.angle(signal.hilbert(PA))
    # [b, a] = signal.butter(4, [5.0 / (fs/2), 7.0 / (fs/2)], 'bandpass')
    # PB = signal.filtfilt(b, a, np.concatenate((np.zeros(L), AV, np.zeros(L))))
    # PB = PB[L:(N+L)]
    # PB = np.angle(signal.hilbert(PB))

    # compute phase-locking value
    p = np.abs(np.sum(np.exp(1j * (PX - PA))) / N)
    print(p)
    
    p = np.abs(np.sum(np.exp(1j * (PX - PB))) / N)
    print(p)
    
    m =  modulation_index(AY, PX)
    print(m)
    
    m =  modulation_index(AV, PX)
    print(m)
    
    # linear relationship between X and Y
    p = np.polyfit(X, Y, 1) 
    U = p[0] * X + p[1]
    
    # correlation
    r = np.corrcoef(X, Y)[0, 1] 
    print(r)
 
    # plot
    plt.figure(1)
    
    plt.subplot(2, 2, 1)
    plt.plot(t, X, 'k')
    plt.plot(t, Y + 3, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.5, 5)
    plt.xlabel('time (s)')
    plt.ylabel('X, Y') 
    
    plt.subplot(2, 2, 2)
    plt.plot(t, X, color=(0.7,0.7,0.7), linewidth=0.5)
    plt.plot(t, PX * 0.3, 'k')
    plt.plot(t, Y + 3, color=(0.7,0.7,0.7), linewidth=0.5)
    plt.plot(t, AY + 3, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.5, 5)
    plt.xlabel('time (s)')
    plt.ylabel('X, Y') 

    plt.subplot(2, 2, 3)
    plt.plot(X, Y, 'g.')
    plt.plot(X, U, 'b')
    plt.xlabel('X')
    plt.ylabel('Y')    
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    plt.figure(2)

    plt.subplot(2, 2, 1)
    plt.plot(t, X, color=(0.7,0.7,0.7), linewidth=0.5)
    plt.plot(t, PX * 0.3, 'k')
    plt.plot(t, Y + 3, color=(0.7,0.7,0.7), linewidth=0.5)
    plt.plot(t, AY + 3, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.5, 5)
    plt.xlabel('time (s)')
    plt.ylabel('X, Y') 
    
    plt.subplot(2, 2, 2)
    plt.plot(t, PX * 0.3, 'k')
    plt.plot(t, PA * 1.5 + 3, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.5, 5)
    plt.xlabel('time (s)')
    plt.ylabel('X, Y') 
    
    plt.subplot(2, 2, 3)
    plt.plot(t, X, color=(0.7,0.7,0.7), linewidth=0.5)
    plt.plot(t, PX * 0.3, 'k')
    plt.plot(t, V + 3, color=(0.7,0.7,0.7), linewidth=0.5)
    plt.plot(t, AV + 3, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.5, 5)
    plt.xlabel('time (s)')
    plt.ylabel('X, Y') 
    
    plt.subplot(2, 2, 4)
    plt.plot(t, PX * 0.3, 'k')
    plt.plot(t, PB * 1.5 + 3, 'r')
    plt.xlim(0, T)
    plt.ylim(-1.5, 5)
    plt.xlabel('time (s)')
    plt.ylabel('X, Y') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L08_main()
