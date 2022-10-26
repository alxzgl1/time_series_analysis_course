# -*- coding: utf-8 -*-

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

"""
np.random.seed(42)

model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.8, 0.2])
model.transmat_ = np.array([[0.8, 0.2], \
                            [0.2, 0.8]])
model.means_ = np.array([[0.0, 0.0], [3.0, -3.0]])
model.covars_ = np.tile(np.identity(2), (3, 1, 1))
X, Z = model.sample(100)

plt.plot(X)
"""

chain_length = 100
chain = np.zeros(chain_length, dtype=int)
chain[0] = 0

transition_probabilities = np.array([[0.7, 0.3], [0.3, 0.7]])

for i in range(1, chain_length): 

    this_step_distribution = transition_probabilities[chain[i-1], :]

    cumulative_distribution = np.cumsum(this_step_distribution)

    print(cumulative_distribution)

    r = np.random.rand(1)

    t = np.squeeze(np.where(cumulative_distribution > r))
    
    if t.all() == 0:
        t = 0
    else:
        if len(t) > 1:
            t = t[0]

    chain[i] = t


plt.plot(chain)
#  provides chain = 1 2 1 2 1 2 1 2 1 1 2 1 2 1 2....