# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Function, main()
#------------------------------------------------------------------------------
def L06_main():
    
    # plot
    plt.subplot(2, 2, 1)
    fc = 25
    fs = 100
    plt.plot([0, fc], [1, 1], 'b')
    plt.plot([fc, fc], [1, 0], 'b')
    plt.plot([fc, fs/2], [0, 0], 'b')
    plt.xlim(0, fs/2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    plt.legend(['low-\npass'], loc='upper right')

    plt.subplot(2, 2, 2)
    fc = 25
    fs = 100
    plt.plot([0, fc], [0, 0], 'b')
    plt.plot([fc, fc], [0, 1], 'b')
    plt.plot([fc, fs/2], [1, 1], 'b')
    plt.xlim(0, fs/2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    plt.legend(['high-\npass'], loc='lower right')
    
    plt.subplot(2, 2, 3)
    fl = 20
    fh = 30
    fs = 100
    plt.plot([0, fl], [0, 0], 'b')
    plt.plot([fl, fl], [0, 1], 'b')
    plt.plot([fl, fh], [1, 1], 'b')
    plt.plot([fh, fh], [0, 1], 'b')
    plt.plot([fh, fs/2], [0, 0], 'b')
    plt.xlim(0, fs/2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    plt.legend(['band-\npass'], loc='upper right')
    
    plt.subplot(2, 2, 4)
    fl = 20
    fh = 30
    fs = 100
    plt.plot([0, fl], [1, 1], 'b')
    plt.plot([fl, fl], [1, 0], 'b')
    plt.plot([fl, fh], [0, 0], 'b')
    plt.plot([fh, fh], [1, 0], 'b')
    plt.plot([fh, fs/2], [1, 1], 'b')
    plt.xlim(0, fs/2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    plt.legend(['stop-\nband'], loc='lower right')
    
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
#------------------------------------------------------------------------------
if __name__ == '__main__':
    L06_main()
