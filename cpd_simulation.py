#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:24:14 2019
===============================================================
Examples of CPD
===============================================================
 
@author: Yongjie
"""

import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt

# Make dataset.
I, J, K, R = 100, 100, 100, 4
X = tt.rand_ktensor((I, J, K), rank=R)

# Add noise.
Xn = np.maximum(0, X.full() + .1*np.random.randn(I, J, K))

# Fit ensemble of unconstrained tensor decompositions.
methods = (
  'cp_als',
  'ncp_bcd',
  'ncp_hals',
)


ensembles = {}
for m in methods:
    ensembles[m] = tt.Ensemble(fit_method=m, fit_options=dict(tol=1e-4))
    ensembles[m].fit(Xn, ranks=range(1, 9), replicates=3)

# Plotting options for the unconstrained and nonnegative models.
plot_options = {
  'cp_als': {
    'line_kw': {
      'color': 'black',
      'label': 'cp_als',
    },
    'scatter_kw': {
      'color': 'black',
    },
  },
  'ncp_hals': {
    'line_kw': {
      'color': 'blue',
      'alpha': 0.5,
      'label': 'ncp_hals',
    },
    'scatter_kw': {
      'color': 'blue',
      'alpha': 0.5,
    },
  },
  'ncp_bcd': {
    'line_kw': {
      'color': 'red',
      'alpha': 0.5,
      'label': 'ncp_bcd',
    },
    'scatter_kw': {
      'color': 'red',
      'alpha': 0.5,
    },
  },
}

# Plot similarity and error plots.
plt.figure()
for m in methods:
    tt.plot_objective(ensembles[m], **plot_options[m])
plt.legend()

plt.figure()
for m in methods:
    tt.plot_similarity(ensembles[m], **plot_options[m])
plt.legend()

plt.show()
