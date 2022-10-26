# -*- coding: utf-8 -*-
"""
Spyder Editor

L01: basic libraries and examples
"""

import numpy as np
import matplotlib.pyplot as plt

# time variable
t = np.linspace(0, 2, 200)
N = np.shape(t)[0] # size() MATLAB

# frequency
f = 2

# signal
y = np.sin(2 * np.pi * f * t)
u = np.random.randn(N)

# plot
plt.plot(t, u)
plt.plot(t, y)