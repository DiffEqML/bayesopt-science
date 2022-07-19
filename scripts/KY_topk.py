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

from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

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

    return torch.sin(10*x)/x.abs()**0.5

# def f(x):
#     return torch.sin(x)/x.abs()**0.7

def naive_topk(y_vals, k):
    """
    Function to return the top-k local maxima.
    """
    glob_indices = np.argsort(-y_vals).cpu().numpy()[:k]

    return glob_indices

def naive_topk_eps(y_vals, domain, k, eps):
    """
    Function to return the top-k local maxima that are eps apart from each other.
    """
    glob_indices = np.argsort(-y_vals).cpu().numpy()

    topk_indices = [glob_indices[0]]

    i = 1
    while i < len(glob_indices) and len(topk_indices) < k:
        temp = np.zeros(len(topk_indices))
        for ii in range(len(topk_indices)):
            temp[ii] = abs(domain[glob_indices[i]] - domain[[topk_indices[ii]]])
        
        if  ((temp >= eps).sum() == temp.size).astype(np.int):
            topk_indices.append(glob_indices[i])
        i += 1

    return topk_indices

def naive_topk_weights(y_vals, domain, k, alpha):
    """
    Function to return the top-k local maxima based on weights; the weights take 
    into account the distance and the y-value of each point. Alpha is a hyperparameter.
    """

    glob_indices = np.argsort(-y_vals).cpu().numpy()
    topk_indices = [glob_indices[0]]

    weights = []
    for i in range(len(glob_indices)):
        weight = y_vals[glob_indices[i]] + alpha*abs(domain[glob_indices[i]] - domain[[topk_indices[0]]])
        weights.append(weight.detach().numpy()[0])
    weights = np.array(weights)
    hi = np.max(weights[1:])
    hello = np.where(weights == np.max(weights[1:]))[0]

    topk_indices.append(glob_indices[hello][0])

    return topk_indices

def particle_topk(dom, f, k, lr, n_particles=1000, inner_iters=1000, eps=1e-1):
    dom_min, dom_max = dom
    
    # sample `n_particles` in uniform [dom_min, dom_max]
    particles = (dom_max - dom_min) * torch.rand(n_particles) + dom_min

    particles = particles.requires_grad_(True)
    
    for i in range(inner_iters):
        y = f(particles)
        grad = torch.autograd.grad(y.sum(), particles)[0]
        particles = particles + lr * grad
        particles = torch.clamp(particles, min=dom_min, max=dom_max)
        
    y = f(particles)
    
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
            if v >= len(particles) or v >= len(vals)  or v >= len(indices):
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
        y = f(particles)
        vals, indices = y.sort(descending=True)
        
        x_ = candidate_x
        
    return torch.stack(values), torch.stack(values_x)

N = 100
n_iter = 0
n_samples = 12
k = 2
eps = 0.5

# plot_x = np.linspace(-1, 1, N)
plot_x = torch.linspace(-1, 1, N)
L = (f(plot_x)[1:] - f(plot_x)[:-1]) / (plot_x[1:] - plot_x[:-1])
# need a step size that is <= 1 / L, where L is the largest Lipschitz const of f
print(1 / L.max())
# tensor_x = torch.linspace(-8, 8, N)
plot_y = f(plot_x)
# indices = naive_topk_eps(plot_y, plot_x, k, eps)
yvals, xvals = particle_topk([-1, 1], f, k, 0.5/ L.max())
print(xvals.detach().numpy(), yvals.detach().numpy())
# print(indices)

# Pick which samples to start with
indx_list = [0, N//4, N//2, 3*N//4, N-1]

train_X = [[plot_x[0]], [plot_x[N//4]], [plot_x[N//2]], [plot_x[3*N//4]], [plot_x[-1]]]
train_Y = [[plot_y[0]],  [plot_y[N//4]], [plot_y[N//2]], [plot_y[3*N//4]], [plot_y[-1]]]

# print(train_X, train_Y)
train_X = torch.tensor(train_X, dtype=dtype)
train_Y = torch.tensor(train_Y, dtype=dtype)


plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
# ax1.title.set_text('T = ' + str(temperature) + '$\\degree$C')
ax1.plot(plot_x, plot_y, 'k-', label='f(x)')
# ax1.plot(plot_x[indices], plot_y[indices], 'r1', markersize=20, label='Actual Top-k Max')
ax1.plot(xvals.detach(), yvals.detach(), 'r1', markersize=20, label='Actual Top-k Max')
ax1.plot(train_X, train_Y, linestyle=' ', color='k', marker='o', mfc='None', markersize=8, label='Observations')
# ax1.set(ylabel='$f(x)$')
# ax1.set_ylim(-1, 1.5)
ax1.legend(loc='upper left', frameon=False)

ax2.set(xlabel='x', ylabel='EIG')
plt.tight_layout()
plt.pause(0.4)

i = 1
while i <= n_iter:
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
        sample_indices = naive_topk_eps(samples[j].squeeze(1), plot_x, k, eps)
        temp_X = plot_x[sample_indices].unsqueeze(1)
        temp_Y = plot_y[sample_indices].unsqueeze(1)
        new_X = torch.cat((train_X, temp_X))
        new_Y = torch.cat((train_Y, temp_Y))

        new_gp = SingleTaskGP(new_X, new_Y)
        new_mll = ExactMarginalLogLikelihood(new_gp.likelihood, new_gp)
        fit_gpytorch_model(new_mll)

        new_posterior = new_gp.posterior(plot_x)

        new_std = new_posterior.variance.sqrt()

        new_entropy += torch.log(new_std) + torch.log(torch.tensor(2*torch.pi).sqrt()) + 0.5

    EIG = entropy - new_entropy/n_samples
    next_indx = torch.argmax(EIG)

    x_next = plot_x[next_indx]
    y_next = f(x_next)

    train_X = torch.cat((train_X, torch.tensor([x_next]).unsqueeze(1)))
    train_Y = torch.cat((train_Y, torch.tensor([y_next]).unsqueeze(1)))

    if i == 1:
        mylabel = 'Posterior'
        mylabel2 = 'BAX Top-k'
        mylabel3 = 'BO Moves'
    else:
        mylabel = None
        mylabel2 = None
        mylabel3 = None

    avg_posterior = posterior.mean
    avg_indices = naive_topk_eps(avg_posterior.squeeze(1).detach(), plot_x, k, eps)
    # print(plot_x[avg_indices].cpu().numpy(), avg_indices)

    # Plot to see the BO moves
    ax1.plot(x_next, y_next, 'bs', markersize=8, label=mylabel3)
    ax1.legend(loc='upper left', frameon=False)
    ax1.plot(plot_x, avg_posterior.detach().numpy(), linestyle='--', color='m', linewidth=1.0, label=mylabel)
    ax1.plot(plot_x[avg_indices], avg_posterior[avg_indices].detach().numpy(), 'y^', markersize=10, label=mylabel2)
    plt.draw()
    
    ax2.plot(plot_x, EIG.detach().numpy(), 'g', alpha=0.5)
    ax2.set(xlabel='x', ylabel='EIG')
    plt.draw()
    plt.tight_layout()
    plt.pause(0.25)
    
    if i < n_iter:
        plt.cla()

    i += 1

# plt.figure(1)
# plt.plot(plot_x, plot_y, color='k')
# plt.plot(plot_x[indices], plot_y[indices], linestyle=' ', color='b', marker='s', markersize=8)
# plt.plot(train_X, train_Y, linestyle=' ', color='k', marker='o', markersize=8)
# plt.plot(plot_x, EIG.detach().numpy(), color='g')
# for i in range(n_samples):
#     plt.plot(plot_x, samples[i].cpu().numpy(), linestyle='--', color='r')
plt.ioff()
plt.show()
