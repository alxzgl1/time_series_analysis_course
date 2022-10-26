# -*- coding: utf-8 -*-

from scipy import signal

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L07_main():
    
    # sampling parameters
    fs = 1000   # sampling rate, in Hz
    
    # design filter in time domain
    f0 = 25
    [b, a] = signal.butter(4, f0 / (fs/2), 'low')
    
    # transform to state space representation of the system
    [A, B, C, D] = signal.tf2ss(b, a)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    L07_main()
