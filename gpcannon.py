# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import os
import pickle
import numpy as np
import matplotlib.pyplot as pl
from scipy.linalg import cho_factor, cho_solve
import scipy.optimize as op

f_all = pickle.load(open("data/norm_tr_fluxes.p", "r"))
l_all = pickle.load(open("data/tr_label_vals.p", "r"))
ivar_all = pickle.load(open("data/norm_tr_ivars.p", "r"))

pix = 8230
label = 0  # teff

f = f_all[:, pix]
l = l_all[:, label]
ivar = ivar_all[:, pix]

m = f > 0.0
f = f[m]
f -= np.mean(f)
l = l[m]
var = 1./ivar[m]


# Covariance function
def kernel(theta, l_i, l_j, white_noise=True):
    a, tau, s = np.exp(theta)
    K = a**2 * np.exp(-0.5*(l_i-l_j)**2/tau**2)
    if white_noise:
        K[np.diag_indices_from(K)] += s**2
    return K


# Calculate alpha
def compute_l_and_alpha(theta):
    # compute the kernel
    K = kernel(theta, l[:, None], l[None, :])
    # add in the uncertainties in the fluxes to the kernel
    K[np.diag_indices_from(K)] += var
    # factorize the kernel
    L, flag = cho_factor(K, overwrite_a=True)
    return (L, flag), cho_solve((L, flag), f)


# Likelihood function
def ln_likelihood(theta):
    (L, flag), alpha = compute_l_and_alpha(theta)
    # compute half of the log determinant
    lndet = np.sum(np.log(np.diag(L)))
    return -0.5*np.dot(f, alpha)-lndet


# Evaluate the predictive mean
def predictive_mean(theta, test_labels):
    K_star = kernel(theta, test_labels[:, None], l[None, :], white_noise=False)
    alpha = compute_l_and_alpha(theta)[1]
    return np.dot(K_star, alpha)

# Evaluate the likelihood of

hyperfn = "best_hyperparams.p"
if not os.path.exists(hyperfn):
    nll = lambda theta: -ln_likelihood(theta)

    # Optimize
    bounds = np.log([(0.001, 0.1), (10, 1000), (0.0001, 0.1)])
    output = op.minimize(nll, np.log([0.0075, 400, 0.005]), method="L-BFGS-B",
                         bounds=bounds)
    best_hyperparams = output.x
    pickle.dump(best_hyperparams, open(hyperfn, "wb"), -1)

    test_labels = np.linspace(np.min(l), np.max(l), 2000)
    mu = predictive_mean(best_hyperparams, test_labels)
    pl.scatter(l, f, c="k")
    pl.plot(test_labels, mu, "g", lw=2)
    pl.show()

else:
    best_hyperparams = pickle.load(open(hyperfn, "r"))


