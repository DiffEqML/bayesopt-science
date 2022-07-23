import sys; sys.path.append('../')

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import math
import random
import numpy as np
import scipy.io
from scipy.interpolate import interp1d

from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import CosineKernel, RBFKernel, RQKernel, MaternKernel

# Top-K specific dependencies
from src.alg.algorithms import MyTopK
from src.utils.domain_util import unif_random_sample_domain
from src.utils.misc_util import dict_to_namespace
from argparse import Namespace, ArgumentParser
from src.acq.acquisition import MyBaxAcqFunction
from src.acq.acqoptimize import AcqOptimizer


nice_fonts = {
    # Use LaTex to write all text
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    # Thesis use 16 and 14, respectively
    "axes.labelsize": 18,
    "font.size": 18,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.figsize": [12,8],
}

mpl.rcParams.update(nice_fonts)

# device = torch.device("cpu")
dtype = torch.double

# Set function
def f(x):

    return torch.sin(8*x)/x.abs()**0.5

# Interpolation for torch dependencies to maintain grad operations
# From https://gist.github.com/chausies
def interp_func(x, y):
  "Returns integral of interpolating function"
  if len(y)>1:
    m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  def f(xs):
    if len(y)==1: # in the case of 1 point, treat as constant function
      return y[0] + torch.zeros_like(xs)
    I = torch.searchsorted(x[1:], xs)
    dx = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dx)
    return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

  return f

def interp(x, y, xs):
  return interp_func(x,y)(xs)

def h_poly_helper(tt):
  A = torch.tensor([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=tt[-1].dtype)
  return [
    sum( A[i, j]*tt[j] for j in range(4) )
    for i in range(4) ]

def h_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = 1
  for i in range(1, 4):
    tt[i] = tt[i-1]*t
  return h_poly_helper(tt)

# Different top-k flavors
def naive_topk(xvals, yvals, k):
    """
    Function to return the top-k local maxima.
    """
    glob_indices = np.argsort(-yvals).cpu().numpy()[:k]

    return xvals[glob_indices], yvals[glob_indices]

def naive_topk_eps(xvals, yvals, k, eps):
    """
    Function to return the top-k local maxima that are eps apart from each other.
    """
    glob_indices = np.argsort(-yvals).cpu().numpy()

    topk_indices = [glob_indices[0]]

    i = 1
    while i < len(glob_indices) and len(topk_indices) < k:
        temp = np.zeros(len(topk_indices))
        for ii in range(len(topk_indices)):
            temp[ii] = abs(xvals[glob_indices[i]] - xvals[[topk_indices[ii]]])
        
        if  ((temp >= eps).sum() == temp.size).astype(np.int):
            topk_indices.append(glob_indices[i])
        i += 1

    return xvals[topk_indices], yvals[topk_indices]

def particle_topk(xvals, yvals, k, lr, func=interp, n_particles=1000, inner_iters=1000, eps=1e-1):
    """
    Function to return the top-k local maxima using a particle gradient descent approach. 
    The algorithm will randomly summon n_particles within your domain, calculate the gradient, 
    check the values, and only retain the peaks.
    """
    dom_min, dom_max = xvals.min(), xvals.max()
    
    # sample `n_particles` in uniform [dom_min, dom_max]
    particles = (dom_max - dom_min) * torch.rand(n_particles) + dom_min

    particles = particles.requires_grad_(True)
    
    for i in range(inner_iters):
        old_particles = particles
        y = func(xvals, yvals, particles)
        grad = torch.autograd.grad(y.sum(), particles, retain_graph=True)[0]
        particles = particles + lr*grad
        particles = torch.clamp(particles, min=dom_min, max=dom_max)
        if (particles - old_particles).abs().max() < eps/10:
            # print("Particles are not changing by much, exiting inner loop...")
            # print("Algorithm took", i, "iterations to converge within", eps/10)
            # print("")
            break

    y = func(xvals, yvals, particles)
    
    # pick top-k
    x_ = torch.zeros(1)
    
    vals, indices = y.sort(descending=True)
    values = []
    values_x = []
    
    for i in range(k):
        if len(particles) == 0:
            print("Breaking early, not enough significant top-k elements...")
            break      
        
        v = 0
        while True:
            if v >= len(particles) or v >= len(vals) or v >= len(indices):
                print("Breaking early, not enough significant top-k elements...")
                return torch.stack(values), torch.stack(values_x)
            
            candidate = vals[v]
            candidate_x = particles[indices[v]]
            v = v + 1
            if (candidate_x - x_).abs() >= eps: break

        values.append(candidate)
        values_x.append(candidate_x)
        
        # pop chosen element
        particles = torch.cat([particles[:v], particles[v+1:]])
        indices = torch.cat([indices[:v], indices[v+1:]])
        vals = torch.cat([vals[:v], vals[v+1:]])
        
        # remove all elements eps close 
        idxs_to_keep = []
        for j, el in enumerate(particles):
            if (particles[j] - x_).abs() >= eps:
                idxs_to_keep.append(j)
        
        particles = particles[idxs_to_keep]
        y = func(xvals, yvals, particles)
        vals, indices = y.sort(descending=True)
        
        x_ = candidate_x
        
    return torch.stack(values_x), torch.stack(values)

# Define the domain resolution and bounds, number of BO iterations, 
# number of posterior samples, number of peaks to detect, type of detection
# algorithm, and convergence criteria
N = 1000
low_bound = -1
upp_bound = 1

n_iter = 20
n_samples = 12
k = 2
normalize = False

# Currently accepted options are: "particle" or "offset"
# topk = "particle"
topk = "offset"

# Initialize the system and find the top-k maxima
plot_x = torch.linspace(low_bound, upp_bound, N)
plot_y = f(plot_x)

# Normalized
if normalize == True:
    plot_x = (plot_x - low_bound)/(upp_bound - low_bound)

if topk == "offset":
    # Choose your desired offset level, i.e., separation between the peaks
    topk_algo = naive_topk_eps
    if normalize == True:
        buffer = 0.1
    else:
        buffer = 0.1*(upp_bound - low_bound)
    print("")
    print("--Executing a naive top-k with an offset of", buffer, "to detect the peaks--")
    topk_xvals, topk_yvals = naive_topk_eps(plot_x, plot_y, k, buffer)
if topk == "particle":
    topk_algo = particle_topk
    print("")
    print("--Executing a particle search top-k to detect the peaks--")
    # Need a step size that is <= 1/L, where L is the largest Lipschitz constant, for convergence
    L = (plot_y[1:] - plot_y[:-1])/(plot_x[1:] - plot_x[:-1])
    buffer = 0.5/L.max()
    topk_xvals, topk_yvals = particle_topk(plot_x, plot_y, k, buffer)

print("")
print("Actual top-k values (x-array, y-array):")
print(topk_xvals.detach().numpy(), topk_yvals.detach().numpy())
print("")

# Construct your training data set
indx_list = [0, N//4, N//2, 3*N//4, N-1]
train_X = [[plot_x[0]], [plot_x[N//4]], [plot_x[N//2]], [plot_x[3*N//4]], [plot_x[N-1]]]
train_Y = [[plot_y[0]],  [plot_y[N//4]], [plot_y[N//2]], [plot_y[3*N//4]], [plot_y[N-1]]]
train_X = torch.tensor(train_X, dtype=dtype)
train_Y = torch.tensor(train_Y, dtype=dtype)

# Display an interactive plot to monitor system behavior
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(plot_x, plot_y, 'k-', label='f(x)')
ax1.plot(topk_xvals.detach(), topk_yvals.detach(), 'r1', markersize=20, label='Actual Top-k Max')
ax1.plot(train_X, train_Y, linestyle=' ', color='k', marker='o', mfc='None', markersize=8, label='Observations')
ax1.set_ylim(-3, 4.5)
ax1.legend(loc='upper left', frameon=False)

ax2.set(xlabel='x', ylabel='EIG')
plt.tight_layout()
plt.pause(0.4)

# Main BO iteration loop
i = 1
tol = 1e-4
err = 1
while (i <= n_iter and err > tol):
    if normalize == True:
        train_Y = (train_Y - train_Y.mean())/train_Y.std()

    # Fitting a GP model
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # Getting the posterior and some samples
    posterior = gp.posterior(plot_x)

    # Calculating the entropy of the posterior
    std_arr = posterior.variance.sqrt()
    entropy = torch.log(std_arr) + torch.log(torch.tensor(2*torch.pi).sqrt()) + 0.5

    samples = posterior.sample(sample_shape=torch.Size([n_samples]))

    # For each sample in samples, run top_k and get x and y indices of top-k maxima
    # Use those along with training data to fit another gp (SingleTaskGP/FixedNoiseGP)
    # Find the posterior of the resulting GP, then get the standard deviation
    # Find the entropy using the std, append one global list, and average to get the expectation
    # Subtract from entropy to get the acquisiton function -> maximum = next step
    new_entropy = torch.zeros(N, 1) 

    for j in range(n_samples):
        sample_xvals, sample_yvals = topk_algo(plot_x, samples[j].squeeze(1), k, buffer)
        new_X = torch.cat((train_X, sample_xvals.detach().unsqueeze(1)))
        new_Y = torch.cat((train_Y, sample_yvals.detach().unsqueeze(1)))

        if normalize == True:
            new_Y = (new_Y - new_Y.mean())/new_Y.std()

        new_gp = SingleTaskGP(new_X, new_Y)
        new_mll = ExactMarginalLogLikelihood(new_gp.likelihood, new_gp)
        fit_gpytorch_model(new_mll)

        new_posterior = new_gp.posterior(plot_x)

        new_std = new_posterior.variance.sqrt()

        new_entropy += torch.log(new_std) + torch.log(torch.tensor(2*torch.pi).sqrt()) + 0.5

    EIG = entropy - new_entropy/n_samples
    EIG_vals = torch.argsort(EIG.squeeze(1).detach(), descending=True)
    # Need to ensure that chosen index is not already sampled
    for jj in EIG_vals:
        if jj not in indx_list:
            next_indx = jj
            break

    x_next = plot_x[next_indx]
    if normalize == True:
        y_next = f(x_next*(upp_bound - low_bound) + low_bound)
    else:
        y_next = f(x_next)

    train_X = torch.cat((train_X, torch.tensor([x_next]).unsqueeze(1)))
    train_Y = torch.cat((train_Y, torch.tensor([y_next]).unsqueeze(1)))
    indx_list.append(next_indx.detach().numpy().tolist())

    if i == 1:
        mylabel = 'Posterior'
        mylabel2 = 'BAX Top-k'
        mylabel3 = 'BO Moves'
    else:
        mylabel = None
        mylabel2 = None
        mylabel3 = None

    avg_posterior = posterior.mean
    if i == 1:
        avg_xvals, avg_yvals = topk_algo(plot_x, avg_posterior.squeeze(1).detach(), k, buffer)
    else:
        old_xvals, old_yvals = avg_xvals, avg_yvals
        avg_xvals, avg_yvals = topk_algo(plot_x, avg_posterior.squeeze(1).detach(), k, buffer)
        err = (avg_xvals - old_xvals).abs().sum()
        print("Iteration, next x, error, topk max")
        print(i, x_next.detach().numpy(), err.detach().numpy(), avg_xvals.detach().numpy())
        

    # Plot to see the BO moves
    ax1.plot(x_next, y_next, 'bs', markersize=8, label=mylabel3)
    ax1.legend(loc='upper left', frameon=False)
    ax1.plot(plot_x, avg_posterior.detach(), linestyle='--', color='m', linewidth=1.0, label=mylabel)
    ax1.plot(avg_xvals.detach(), avg_yvals.detach(), 'y^', markersize=10, label=mylabel2)
    plt.draw()
    
    ax2.plot(plot_x, EIG.detach(), 'g', alpha=0.5)
    ax2.set(xlabel='x', ylabel='EIG')
    plt.draw()
    plt.tight_layout()
    plt.pause(0.25)
    
    if err > tol:
        plt.cla()

    i += 1

if i-1 != 0:
    print("Converged in", i-1, "iterations with error:")
    ov_err = np.divide((avg_xvals.detach().numpy() - topk_xvals.detach().numpy()), topk_xvals.detach().numpy())
    print(np.round(abs(ov_err)*100, 1))
plt.ioff()
plt.show()
