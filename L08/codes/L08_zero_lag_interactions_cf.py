# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def cf_coupling(LF_data, HF_data, ratio):
    
    # init
    N = len(LF_data)
    
    # get phase
    LF_phase = np.angle(signal.hilbert(LF_data))
    HF_phase = np.angle(signal.hilbert(HF_data))
    
    # unwrap phases
    LF_unwrap_phase = np.unwrap(LF_phase)
    HF_unwrap_phase = np.unwrap(HF_phase)
    
    # rescale phase
    LF_wrap_phase = (LF_unwrap_phase % (2 * np.pi / ratio)) * ratio
    HF_wrap_phase = (HF_unwrap_phase % (2 * np.pi))
    
    # compute phase-locking value
    p = np.abs(np.sum(np.exp(1j * (LF_wrap_phase - HF_wrap_phase))) / N)
    
    return p

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L08_main():
    
    b_sine_signal = 1
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    T  = 4      # duration, in seconds
    N  = T * fs # duration, in samples
    
    # time variable
    t = np.linspace(0, T, N)
    
    # generate signal
    if b_sine_signal == 1:
        ratio = 2
        f0 = 10
        X = np.sin(2 * np.pi * f0 * t)
        Y = np.sin(2 * np.pi * (f0 * ratio) * t)
    else:
        # signal realistic signal
        ratio = 2
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
    
    # cross-frequency coupling
    p = cf_coupling(X, Y, ratio)
    print(p)
    
    # linear relationship between AX and AY
    p = np.polyfit(X, Y, 1) 
    U = p[0] * X + p[1]
    
    # correlation
    r = np.corrcoef(X, Y)[0, 1] 
    print(r)
    
    # CFC: get phase
    LF_phase = np.angle(signal.hilbert(X))
    HF_phase = np.angle(signal.hilbert(Y))
    
    # CFC: unwrap phases
    LF_unwrap_phase = np.unwrap(LF_phase)
    HF_unwrap_phase = np.unwrap(HF_phase)
    
    # CFC: rescale phase
    LF_wrap_phase = (LF_unwrap_phase % (2 * np.pi / ratio)) * ratio
    HF_wrap_phase = (HF_unwrap_phase % (2 * np.pi))
 
    # plot
    plt.figure(1)
    
    plt.subplot(2, 1, 1)
    plt.plot(t, X, 'k')
    plt.plot(t, Y + 2.5, 'r')
    plt.xlim(0, T)
    #plt.ylim(-1.5, 4)
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
    plt.plot(t, X, 'k')
    plt.plot(t, Y + 2.5, 'r')
    plt.xlim(1, 3)
    plt.ylim(-1.5, 4)
    plt.xlabel('time (s)')
    plt.ylabel('X, Y') 
    
    plt.subplot(2, 2, 2)
    plt.plot(t, LF_phase * 0.25, 'k')
    plt.plot(t, HF_phase * 0.25 + 2.5, 'r')
    plt.xlim(1, 3)
    plt.ylim(-1.5, 4)
    plt.xlabel('time (s)')
    plt.ylabel('phases (X, Y)') 
    
    plt.subplot(2, 2, 3)
    plt.plot(t, LF_unwrap_phase * 0.25, 'k')
    plt.plot(t, HF_unwrap_phase * 0.25 + 2.5, 'r')
    plt.xlim(1, 3)
    # plt.ylim(-1.5, 4)
    plt.xlabel('time (s)')
    plt.ylabel('unwrapped phases (X, Y)') 
    
    plt.subplot(2, 2, 4)
    plt.plot(t, LF_wrap_phase * 0.25 - 1, 'k')
    plt.plot(t, HF_wrap_phase * 0.25 - 1 + 2.5, 'r')
    plt.xlim(1, 3)
    plt.ylim(-1.5, 4)
    plt.xlabel('time (s)')
    plt.ylabel('rescaled phases (X, Y)') 
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L08_main()
