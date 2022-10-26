# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L03_main():
    
    # parameters
    fs = 1000   # sampling rate, in Hz
    N  = 1000   # duration, in samples
    T  = N / fs # duration, in seconds
    
    f0 = 3      # signal frequency, in Hz
    fc = 30     # carrier frequency, in Hz
    AM = 0.5    # modulation factor
    
    # time variable
    t = np.linspace(0, T, N)
    
    # amplitude modulated signal
    y = 1 + AM * np.cos(2 * np.pi * f0 * t)
    s = np.sin(2 * np.pi * fc * t) * y
    
    # detect amplitude and phase
    A = np.abs(signal.hilbert(s))
    P = np.angle(signal.hilbert(s)) / (2 * np.pi)
    
    # plot
    plt.subplot(2, 1, 1)
    plt.plot(t, s)
    plt.plot(t, y, linestyle='--')
    plt.xlim(0, T)
    plt.ylim(-2.0, 2.0)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, P)
    plt.plot(t, A, linestyle='--')
    plt.xlim(0, T)
    plt.ylim(-2.0, 2.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
