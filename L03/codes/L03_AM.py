# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

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
    
    # plot
    plt.plot(t, s)
    plt.plot(t, y, linestyle='--')
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L03_main()
