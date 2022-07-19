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
    x = torch.tensor(x)*8

    return torch.sin(x)/x.abs()**0.5

# WIP; not fully working
def naive_topk_eps(func, domain, k, eps):
    """
    Function to return the top-k local maxima that are eps apart.
    """
    temp = -func(domain)
    glob_indices = np.argsort(temp).cpu().numpy()

    topk_indices = [glob_indices[0]]

    i = 1
    j = 0
    while i < len(glob_indices) and len(topk_indices) < k:
        if abs(domain[glob_indices[i]] - domain[[glob_indices[j]]]) > eps:
            topk_indices.append(glob_indices[i])
            j = i
        i += 1

    return topk_indices

def naive_topk(y_vals, k):
    """
    Function to return the top-k local maxima.
    """
    glob_indices = np.argsort(-y_vals).cpu().numpy()[:k]


    return glob_indices

N = 100
n_iter = 10
n_samples = 12
k = 2

plot_x = np.linspace(-1, 1, N)
plot_y = f(plot_x).cpu().numpy()
# norm_x = (plot_x - plot_x.min())/(plot_x.max() - plot_x.min())
indices = naive_topk(torch.tensor(plot_y), k)
print(indices)

# Pick which samples to start with
indx_list = [0, N//4, N//2, 3*N//4, N-1]
print(indx_list) 

train_X = [[plot_x[0]], [plot_x[N//4]], [plot_x[N//2]], [plot_x[3*N//4]], [plot_x[-1]]]
train_Y = [[plot_y[0]],  [plot_y[N//4]], [plot_y[N//2]], [plot_y[3*N//4]], [plot_y[-1]]]

train_X = torch.tensor(train_X, dtype=dtype)
train_Y = torch.tensor(train_Y, dtype=dtype)


plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
# ax1.title.set_text('T = ' + str(temperature) + '$\\degree$C')
ax1.plot(plot_x, plot_y, 'k-', label='f(x)')
ax1.plot(plot_x[indices], plot_y[indices], 'r1', markersize=20, label='Actual Top-k Max')
ax1.plot(train_X, train_Y, linestyle=' ', color='k', marker='o', mfc='None', markersize=8, label='Observations')
# ax1.set(ylabel='$f(x)$')
ax1.set_ylim(-1, 1.5)
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
    posterior = gp.posterior(torch.tensor(plot_x))

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
        sample_indices = naive_topk(samples[j].squeeze(1), k)
        temp_X = torch.tensor(plot_x[sample_indices]).unsqueeze(1)
        temp_Y = torch.tensor(plot_y[sample_indices]).unsqueeze(1)
        new_X = torch.cat((train_X, temp_X))
        new_Y = torch.cat((train_Y, temp_Y))

        new_gp = SingleTaskGP(new_X, new_Y)
        new_mll = ExactMarginalLogLikelihood(new_gp.likelihood, new_gp)
        fit_gpytorch_model(new_mll)

        new_posterior = new_gp.posterior(torch.tensor(plot_x))

        new_std = new_posterior.variance.sqrt()

        new_entropy += torch.log(new_std) + torch.log(torch.tensor(2*torch.pi).sqrt()) + 0.5

    EIG = entropy - new_entropy/n_samples
    next_indx = torch.argmax(EIG)

    x_next = plot_x[next_indx]
    y_next = f(x_next)
    x_next = torch.tensor([x_next]).unsqueeze(1)
    y_next = torch.tensor([y_next]).unsqueeze(1)

    train_X = torch.cat((train_X, x_next))
    train_Y = torch.cat((train_Y, y_next))

    if i == 1:
        mylabel = 'Posterior'
        mylabel2 = 'BAX Top-k'
        mylabel3 = 'BO Moves'
    else:
        mylabel = None
        mylabel2 = None
        mylabel3 = None

    avg_posterior = posterior.mean
    avg_indices = naive_topk(avg_posterior.squeeze(1).detach(), k)
    print(avg_indices)

    # Plot to see the BO moves
    ax1.plot(x_next, y_next.cpu().numpy(), 'bs', markersize=8, label=mylabel3)
    ax1.plot(plot_x, avg_posterior.detach().numpy(), linestyle='--', color='m', linewidth=0.75, label=mylabel)
    ax1.plot(plot_x[avg_indices], avg_posterior[avg_indices].detach().numpy(), 'y^', markersize=10, label=mylabel2)
    ax1.legend(loc='upper left', frameon=False)
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
