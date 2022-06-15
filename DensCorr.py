import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import math
import numpy as np
import scipy.io
from scipy.interpolate import interp1d

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf


nice_fonts = {
        # Use LaTex to write all text
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        # Thesis use 16 and 14, respectively
        "axes.labelsize": 16,
        "font.size": 16,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 14,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.figsize": [7,5],
}

mpl.rcParams.update(nice_fonts)

dtype = torch.float

def loadmatlab(filename):
    mat = scipy.io.loadmat(filename, appendmat=True)
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
P, corr_length, dens_fluc = loadmatlab('B1')

# Creating a cubic interpolation based on the gathered data
def f(x):
    f_interp = interp1d(P, corr_length, kind='cubic')

    return f_interp(x)

# For plotting purposes only
plot_x = np.linspace(P.min(), P.max(), 1001)
plot_y = f(plot_x)
actual_max = plot_x[np.where(plot_y == plot_y.max())[0]]

# Normalized pressure values
norm_P = (P - P.min())/(P.max() - P.min())

bounds = torch.stack([torch.zeros(1), torch.ones(1)])

# Pick which samples to start with
train_X = [[norm_P[0]], [norm_P[len(P)//4]], [norm_P[len(P)//2]], [norm_P[-1]]]
train_Y = [[corr_length[0]], [corr_length[len(P)//4]], [corr_length[len(P)//2]], [corr_length[-1]]]
train_X = torch.tensor(train_X)
train_Y = torch.tensor(train_Y)

# Interactive plot to visualize BO moves
# plt.ion()
# plt.figure(1)
# plt.plot(plot_x, plot_y, 'k-', label='f(P)')
# plt.plot(actual_max, plot_y.max(), 'r1', markersize=20, label='Actual Max')
# plt.plot(train_X.cpu().numpy()*(P.max() - P.min()) + P.min(), train_Y.cpu().numpy(), 'ko', mfc='None', markersize=8, label='Samples')
# plt.xlabel('$P~$[bar]')
# plt.ylabel('$L~[\\AA]$')
# plt.xlim(70, 82)
# plt.legend(frameon=False)
# plt.tight_layout()
# plt.pause(0.1)

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
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
plt.pause(0.1)


i = 1
err = 1
tol = 1e-3
n_iter = 20
x_span = torch.linspace(0, 1, 1001)[:, None, None] # batch, 1 (q), 1 (feature dimension)

# Main BO Loop
while i <= n_iter and abs(err) > tol:
    # Fitting a GP model
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # Getting an acquisition function
    EI = qExpectedImprovement(gp, train_Y.max(), maximize=True)

    # Optimizing the acquisition function to get its max 
    candidate, acq_value = optimize_acqf(EI, bounds=bounds, q=1, num_restarts=5, raw_samples=512)
    
    # Evaluate ACQ function at the candidate at a linspace
    # Use this to plot BO moves
    # Note that ExpectedImprovement has an analytic form that uses GP posteriors
    # whereas qExpectedImprovement uses a Monte Carlo approximation
    # https://botorch.org/v/0.1.0/docs/acquisition
    acq_eval = EI(x_span)


    # Calculate by how much the BO guess has moved by
    unnorm_candidate = candidate*(P.max() - P.min()) + P.min()
    prev_candidate = train_X[-1]*(P.max() - P.min()) + P.min()

    # err = (unnorm_candidate - prev_candidate)/prev_candidate
    err = candidate - train_X[-1]

    print(i, unnorm_candidate, f(unnorm_candidate), abs(err).cpu().numpy())

    # Append new torch tensors
    train_X = torch.cat((train_X, candidate))
    train_Y = torch.cat((train_Y, torch.tensor(f(unnorm_candidate))))

    if i == 1:
        mylabel = 'BO Step'
        mylabel2 = 'Acq. Func.'
        mylabel3 = 'Target'
    else:
        mylabel = None
        mylabel2 = None
        mylabel3 = None
    i += 1

    # Plot to see the BO moves
    ax1.plot(unnorm_candidate, f(unnorm_candidate), 'bs', markersize=8, label=mylabel)
    ax1.legend(frameon=False)
    plt.draw()
    # ax1.pause(0.05)

    ax2.plot(plot_x, acq_eval.detach().numpy(), 'g', alpha=0.5, label=mylabel2)
    ax2.plot(plot_x[np.where(acq_eval.detach().numpy() == acq_eval.detach().numpy().max())[0]], acq_eval.detach().numpy().max(), 'rx', markersize=8, label=mylabel3)
    ax2.legend(frameon=False)
    plt.draw()
    # ax2.tight_layout()
    plt.pause(0.05)

error = ((unnorm_candidate - actual_max)*100/actual_max).cpu().numpy()
print("Error from actual max is:", round(error[0][0],2),"%") 
print("Actual max is:", actual_max, round(plot_y.max(),2))
plt.ioff()
plt.show()