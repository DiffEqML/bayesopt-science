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

def loadmatlab(filename):
    mat = scipy.io.loadmat('NoiseFree/'+filename, appendmat=True)
    mat_contents = mat[filename]
    N = len(mat_contents)
    P = np.zeros(N)
    corr_length = np.zeros(N)
    dens_fluc = np.zeros(N)

    for i in range(N):
        P[-1-i] = mat_contents[i][0]
        corr_length[-1-i] = mat_contents[i][1]
        dens_fluc[-1-i] = mat_contents[i][2]

    return P, corr_length, dens_fluc


# Loading the data
temperature = 30
print(temperature)
if type(temperature) == int:
    P, corr_length, dens_fluc = loadmatlab('co2_t'+str(temperature)+'C')
else:
    if temperature == 30.5:
        P, corr_length, dens_fluc = loadmatlab('B2')
    if temperature == 31.5:
        P, corr_length, dens_fluc = loadmatlab('B1')
    if temperature == 32.5:
        P, corr_length, dens_fluc = loadmatlab('B3')


# Creating a cubic interpolation based on the gathered data
def f(x):
    f_interp = interp1d(P, corr_length, kind='cubic')

    return f_interp(x)

def get_init(num_samples):

    return random.sample(range(len(P)), num_samples)


# For plotting purposes only
plot_x = np.linspace(P.min(), P.max(), 1001)
plot_y = f(plot_x)
actual_max = plot_x[np.where(plot_y == plot_y.max())[0]]

# Normalized pressure values
norm_P = (P - P.min())/(P.max() - P.min())
# norm_corr_length = (corr_length - corr_length.min())/(corr_length.max() - corr_length.min())

bounds = torch.stack([torch.zeros(1, dtype=dtype), torch.ones(1, dtype=dtype)])

# Pick which samples to start with
train_X = [[norm_P[0]], [norm_P[-1]]]
train_Y = [[corr_length[0]], [corr_length[-1]]]
# train_X = [[norm_P[0]], [norm_P[len(P)//2]]]
# train_Y = [[corr_length[0]], [corr_length[len(P)//2]]]
train_X = torch.tensor(train_X, dtype=dtype)
train_Y = torch.tensor(train_Y, dtype=dtype)

# Interactive plot to visualize BO moves
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.title.set_text('T = ' + str(temperature) + '$\\degree$C')
ax1.plot(plot_x, plot_y, 'k-', label='f(P)')
ax1.plot(actual_max, plot_y.max(), 'r1', markersize=20, label='Actual Max')
ax1.plot(train_X.cpu().numpy()*(P.max() - P.min()) + P.min(), train_Y.cpu().numpy(), 'ko', mfc='None', markersize=8, label='Samples')
ax1.set( ylabel='$L~[\\AA]$')
ax1.set_xlim(70, 82)
ax1.legend(frameon=False)
# ax1.pause(0.1)

ax2.set(xlabel='$P~$[bar]', ylabel='Exp. Improvement')
ax2.set_xlim(70, 82)
plt.tight_layout()
plt.pause(0.4)


i = 1
err = 1
tol = 1e-2
n_iter = 20
x_span = torch.linspace(0, 1, 1001)[:, None, None] # batch, 1 (q), 1 (feature dimension)
noise_variance = 1e-8

# Main BO Loop
while i <= n_iter and abs(err) > tol:
    # Two different ways of normalizing; keep training (generated) data stored separately
    norm_Y = (train_Y - train_Y.mean())/train_Y.std() # Mean and Std
    # norm_Y = (train_Y - train_Y.min())/(train_Y.max() - train_Y.min()) # Min Max

    # Unnormalized
    # norm_Y = train_Y
    
    # Fitting a GP model
    # likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-7))
    # likelihood = GaussianLikelihood(noise_constraint=GreaterThan(0.7))
    # gp = SingleTaskGP(train_X, norm_Y, covar_module=RQKernel(), likelihood=likelihood)
    # gp = SingleTaskGP(train_X, norm_Y, likelihood=likelihood)
    # gp = FixedNoiseGP(train_X, norm_Y, covar_module=RQKernel(), train_Yvar=torch.full_like(norm_Y, 1e-8))
    gp = FixedNoiseGP(train_X, norm_Y, train_Yvar=torch.full_like(norm_Y, noise_variance))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # Getting an acquisition function
    EI = qExpectedImprovement(gp, norm_Y.max(), maximize=True)

    # Optimizing the acquisition function to get its max 
    candidate, _ = optimize_acqf(EI, bounds=bounds, q=1, num_restarts=5, raw_samples=1000)

    acq_eval = EI(x_span)

    # Calculate by how much the BO guess has moved by
    unnorm_candidate = candidate*(P.max() - P.min()) + P.min()
    # prev_candidate = train_X[-1]*(P.max() - P.min()) + P.min()

    # err = (unnorm_candidate - prev_candidate)/prev_candidate
    err = candidate - train_X[-1]

    print(i, unnorm_candidate.cpu().numpy(), f(unnorm_candidate), abs(err).cpu().numpy())

    # norm_f_candidate = (f(unnorm_candidate) - corr_length.min())/(corr_length.max() - corr_length.min())
    # norm_f_candidate = (f(unnorm_candidate) - train_Y.mean().cpu().numpy())/train_Y.std().cpu().numpy()
    # Append new torch tensors
    train_X = torch.cat((train_X, candidate))

    # Normalized
    # train_Y = torch.cat((train_Y, torch.tensor(norm_f_candidate)))

    # Unnormalized
    train_Y = torch.cat((train_Y, torch.tensor(f(unnorm_candidate), dtype=dtype)))

    if i == 1:
        mylabel = 'BO Step'
        mylabel2 = 'Acq. Func.'
        mylabel3 = 'Target'
    else:
        mylabel = None
        mylabel2 = 'Acq. Func.'
        mylabel3 = None
    i += 1

    # Plot to see the BO moves
    ax1.plot(unnorm_candidate, f(unnorm_candidate), 'bs', markersize=8, label=mylabel)
    ax1.legend(frameon=False)
    plt.draw()
    # ax1.pause(0.05)


    ax2.plot(plot_x, acq_eval.detach().numpy(), 'g', alpha=0.5, label=mylabel2)
    ax2.set(xlabel='$P~$[bar]', ylabel='Exp. Improvement')
    # ax2.plot(plot_x[np.where(acq_eval.detach().numpy() == acq_eval.detach().numpy().max())[0]], 1, 'rx', markersize=8, label=mylabel3)
    ax2.legend(frameon=False)
    plt.draw()
    plt.tight_layout()
    # ax2.tight_layout()
    plt.pause(0.25)

    if abs(err) > tol:
        plt.cla()

error = ((unnorm_candidate - actual_max)*100/actual_max).cpu().numpy()
print("Error from actual max is:", round(error[0][0],2),"%") 
print("Actual max is:", actual_max, round(plot_y.max(),2))
plt.ioff()
plt.show()