# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import os
import pickle
import numpy as np
import matplotlib.pyplot as pl
from scipy.linalg import cho_factor, cho_solve
from itertools import product
import scipy.optimize as op

f_all = pickle.load(open("data/norm_tr_fluxes.p", "r"))
l_all = pickle.load(open("data/tr_label_vals.p", "r"))
ivar_all = pickle.load(open("data/norm_tr_ivars.p", "r"))

pix = 1383 

f = f_all[:, pix]
ivar = ivar_all[:, pix]

# identify bad pixels. flux == 0 or flux == 1
m = np.logical_and(f != 0.0, f != 1.0)
f = f[m]
f -= np.mean(f)
l_all = l_all[m,:]
var = 1./ivar[m]


# Covariance function
def kernel(theta, l_i, l_j, white_noise=True):
    a, tau0, tau1, tau2, s = np.exp(theta)
    K = a**2 * np.exp(-0.5*(((l_i-l_j)[:,:,0])**2/tau0**2 + 
                           ((l_i-l_j)[:,:,1])**2/tau1**2 + 
                           ((l_i-l_j)[:,:,2])**2/tau2**2))
    if white_noise:
        K[np.diag_indices_from(K)] += s**2
    return K


# Calculate alpha
def compute_l_and_alpha(theta):
    # compute the kernel
    K = kernel(theta, l_all[:, None], l_all[None, :])
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
    K_star = kernel(theta, test_labels[:, None], 
                    l_all[None, :], white_noise=False)
    alpha = compute_l_and_alpha(theta)[1]
    f_star_bar = np.dot(K_star, alpha)
    return f_star_bar


# Evaluate the likelihood of a data point given model
def test_ln_likelihood(label, flux, flux_var, theta, (L,flag), alpha):
    # Calculate the mean prediction.
    K_star = kernel(theta, label[None,None,:], l_all[None,:,:], 
                    white_noise=False)
    f_star = np.dot(K_star, alpha)

    # Calculate the variance on the prediction.
    label_array = np.array(label[None,None,:])  # Hack so kernel works w scalar.
    pred_var = flux_var + float(kernel(theta, label_array, label_array))
    pred_var -= np.dot(K_star, cho_solve((L,flag), K_star[0]))

    return -0.5 * ((flux - f_star) ** 2 / pred_var + np.log(pred_var))

hyperfn = "best_hyperparams.p"
if not os.path.exists(hyperfn):
    print("fitting for hyperparams")
    nll = lambda theta: -ln_likelihood(theta)

    # Optimize
    bounds = np.log([(0.001, 0.1), (100, 10000), (0.01, 10), (0.01, 10),
                     (0.0001, 0.1)])
    output = op.minimize(nll, np.log([0.01, 100, 0.5, 0.05, 0.01]), 
                         method="L-BFGS-B", bounds=bounds)
    best_hyperparams = output.x
    print(np.exp(best_hyperparams))
    pickle.dump(best_hyperparams, open(hyperfn, "wb"), -1)

else:
    print("loading")
    best_hyperparams = pickle.load(open(hyperfn, "r"))

# Now optimize over the test label space
obj = 0
(L,flag), alpha = compute_l_and_alpha(best_hyperparams)
tll = lambda labels: -test_ln_likelihood(labels, f[obj], var[obj], 
                                         best_hyperparams, (L,flag), alpha)

p0 = np.array([np.mean(l_all[:,i]) for i in range(l_all.shape[1])])
bounds = np.array([(3500,5500), (0,5), (-2.5,0.5)]) # training set dist

print("fitting for labels")
# want to minimize the negative log likelihood (maximize lnp)
output = op.minimize(tll, p0, method="L-BFGS-B", bounds=bounds)

# test_labels = np.zeros((2000,3)) + np.max(l_all, axis=0)[None,:]
# axis = 2 
# test_labels[:,axis] = np.linspace(np.min(l_all[:,axis]), 
#                                   np.max(l_all[:,axis]), 
#                                   len(test_labels[:,axis]))
# mu = predictive_mean(best_hyperparams, test_labels)
# pl.scatter(l_all[:,axis], f, c="k")
# pl.plot(test_labels[:,axis], mu, "g", lw=2)
# pl.show()

# (L,flag), alpha = compute_l_and_alpha(best_hyperparams)
# pl.figure()
# for obj in range(5,9):
#     lls = [test_ln_likelihood(test_labels[i,:], f[obj], var[obj], 
#            best_hyperparams, (L,flag), alpha) for i in range(0,len(test_labels))]
#     pl.plot(test_labels[:,axis], lls, label="%s"%np.round(l_all[obj,axis],2))
#     pl.legend()
#     pl.gca().axvline(l_all[obj,axis])
# pl.show()

