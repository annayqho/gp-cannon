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

def kernel(theta, l_i, l_j, white_noise=True):
    a, tau0, tau1, tau2, s = np.exp(theta)
    K = a**2 * np.exp(-0.5*(((l_i-l_j)[:,:,0])**2/tau0**2 + 
                           ((l_i-l_j)[:,:,1])**2/tau1**2 + 
                           ((l_i-l_j)[:,:,2])**2/tau2**2))
    if white_noise:
        K[np.diag_indices_from(K)] += s**2
    return K


# Calculate alpha
def compute_l_and_alpha(theta, f, var, l_all):
    # compute the kernel
    K = kernel(theta, l_all[:, None], l_all[None, :])
    # add in the uncertainties in the fluxes to the kernel
    K[np.diag_indices_from(K)] += var
    # factorize the kernel
    L, flag = cho_factor(K, overwrite_a=True)
    return (L, flag), cho_solve((L, flag), f)


# Likelihood function
def ln_likelihood(theta, f, var, l_all):
    (L, flag), alpha = compute_l_and_alpha(theta, f, var, l_all)
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
def test_ln_likelihood_single_pix(label, flux, flux_var, theta, (L,flag), alpha, l_all):
    # Calculate the mean prediction.
    K_star = kernel(theta, label[None,None,:], l_all[None,:,:], 
                    white_noise=False)
    f_star = np.dot(K_star, alpha)

    # Calculate the variance on the prediction.
    label_array = np.array(label[None,None,:])  # Hack so kernel works w scalar.
    pred_var = flux_var + float(kernel(theta, label_array, label_array))
    pred_var -= np.dot(K_star, cho_solve((L,flag), K_star[0]))

    return -0.5 * ((flux - f_star) ** 2 / pred_var + np.log(pred_var))


def test_ln_likelihood(label, f, var, theta_all, f_all, var_all, l_all):
    #print(label)
    npix = len(f)
    npix = 10 
    ln_likelihoods = np.zeros(npix)
    for pix in range(100,npix):
        (L,flag), alpha = compute_l_and_alpha(
                theta_all[pix,:], f_all[:,pix], var_all[:,pix], l_all)
        ln_likelihoods[pix] = test_ln_likelihood_single_pix(
                label, f[pix], var[pix], theta_all[pix], (L,flag), alpha, l_all)
    return sum(ln_likelihoods)


def pick_good_obj(pix, f_all, ivar_all, l_all):
    f = f_all[:, pix]
    ivar = ivar_all[:, pix]

    # this pixel is bad for the following objects: flux == 0 or flux == 1
    m = np.logical_and(f != 0.0, f != 1.0)
    f = f[m]
    f -= np.mean(f)
    l = l_all[m,:]
    var = 1./ivar[m]
    return f,var,l


def train_single_pix(pix, f_all, ivar_all, l_all):
    f, var, l = pick_good_obj(pix, f_all, ivar_all, l_all)
    nll = lambda theta: -ln_likelihood(theta, f, var, l)
    bounds = np.log([(0.001, 0.1), (100, 10000), (0.01, 10), (0.01, 10),
                     (0.0001, 0.1)])
    output = op.minimize(nll, np.log([0.01, 100, 0.5, 0.05, 0.01]), 
                         method="L-BFGS-B", bounds=bounds)
    best_hyperparams = output.x
    print("best hyperparams: ")
    print(ln_likelihoods[pix])
    print(np.exp(best_hyperparams))
    return best_hyperparams


def infer_labels_single_pix(pix, best_hyperparams, f_all, var_all, l_all):
    f, var, l = pick_good_obj(pix, f_all, ivar_all, l_all)
    hyperparams = best_hyperparams[pix,:]
    (L,flag), alpha = compute_l_and_alpha(hyperparams, f, var, l)
    tll = lambda labels: -test_ln_likelihood(labels, f[obj], var[obj], 
                                             hyperparams, (L,flag), alpha, l)
    p0 = np.array([np.mean(l[:,i]) for i in range(l.shape[1])])
    bounds = np.array([(3500,5500), (0,5), (-2.5,0.5)]) # training set dist

    print("fitting for labels")
    # want to minimize the negative log likelihood (maximize lnp)
    output = op.minimize(tll, p0, method="L-BFGS-B", bounds=bounds)
    return output.x


def infer_labels_single_obj(obj, hyperparams_all, f_all, var_all, l_all):
    f = f_all[obj,:]
    var = var_all[obj,:]
    # l = l_all[obj,:]
    tll = lambda labels: -test_ln_likelihood(
            labels, f, var, hyperparams_all, f_all, var_all, l_all)
    p0 = np.array([np.mean(l_all[:,i]) for i in range(l_all.shape[1])])
    #p0 = l_all[obj,:]
    bounds = np.array([(3500,5500), (0,5), (-2.5,0.5)]) # training set dist
    print("fitting for labels")
    output = op.minimize(tll, p0, method="L-BFGS-B", bounds=bounds)
    return output


f_all = pickle.load(open("data/norm_tr_fluxes.p", "r"))
l_all = pickle.load(open("data/tr_label_vals.p", "r"))
ivar_all = pickle.load(open("data/norm_tr_ivars.p", "r"))

# "bad" pixels are those with var == 0 
keep = np.var(f_all, axis=0) > 0
f_all = f_all[:,keep]
ivar_all = ivar_all[:,keep]

npix = f_all.shape[1]

# Covariance function

hyperfn = "best_hyperparams.p"
if not os.path.exists(hyperfn):
    print("fitting for hyperparams")
    # Optimize for each pixel independently
    npix = f_all.shape[1]
    best_hyperparams_all = map(
            train_single_pix(pix, f_all, ivar_all, l_all), np.linspace(0,npix,npix+1))
    pickle.dump(best_hyperparams_all, open(hyperfn, "wb"), -1)

else:
    print("loading")
    best_hyperparams = pickle.load(open(hyperfn, "r"))

##### TEST STEP #####

# Now optimize over the test label space
obj = 0
print(infer_labels_single_obj(obj, best_hyperparams, f_all, ivar_all, l_all))


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

