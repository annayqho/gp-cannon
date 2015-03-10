# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import pickle
import numpy as np
from itertools import product
import matplotlib.pyplot as pl
from scipy.linalg import cho_factor, cho_solve

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
def kernel(theta, l_i, l_j):
    a, tau = theta
    K = a**2 * np.exp(-0.5*(l_i-l_j)**2/tau**2)
    K[np.diag_indices_from(K)] += 0.004
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
    K_star = kernel(theta, test_labels[:, None], l[None, :])
    alpha = compute_l_and_alpha(theta)[1]
    return np.dot(K_star, alpha)

a = np.linspace(-5, 0, 50)
a = 10**a
taus = np.linspace(-3, 3, 51)
taus = 10**taus
likelihoods = np.zeros((len(a), len(taus)))

for xi, yi in product(range(len(a)), range(len(taus))):
    likelihoods[xi, yi] = ln_likelihood([a[xi], taus[yi]])

pl.pcolor(a, taus, np.log(likelihoods.T), cmap="gray_r")
pl.xscale("log")
pl.yscale("log")
pl.colorbar()
pl.show()


# pl.figure()
# test_labels = np.linspace(np.min(l), np.max(l), 2000)

# i, j = np.unravel_index(np.argmax(likelihoods), likelihoods.shape)
# amax = a[i]
# taumax = taus[j]
# print(amax, taumax)

# mu = predictive_mean([amax, taumax], test_labels)
# pl.scatter(l, f, c="k")
# pl.plot(test_labels, mu, "g", lw=2)
