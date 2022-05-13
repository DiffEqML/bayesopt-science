#### TODO: finish and parametrize from CLI


import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch

from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
from src.sim import MixtureModel
import matplotlib.pyplot as plt

import math
from dataclasses import dataclass

import glob
from PIL import Image
import matplotlib as mpl

nice_fonts = {
        # Use LaTex to write all text
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        # Thesis use 16 and 14, respectively
        #"axes.labelsize": 16,
        "font.size": 16,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 14,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
}

mpl.rcParams.update(nice_fonts)


##############
# Utils
##############

def make_bo_gif(run_id, frame_folder):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/exploration*.png"))]
    frame_one = frames[0]
    frame_one.save(f"bo_run_{run_id}.gif", format="GIF", append_images=frames,
               save_all=True, duration=800, loop=0)

    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/density*.png"))]
    frame_one = frames[0]
    frame_one.save(f"density_{run_id}.gif", format="GIF", append_images=frames,
               save_all=True, duration=800, loop=0)


frame_folder = 'artifacts/plots/bo_exploration'


##############
# Define hyperparams and run BO
#############


##############
# Generate gif
##############

sim = MixtureModel(concentrations=[1., 0.])

for k in range(n_init, n_iters):
    fig, ax = plt.subplots(1,1,figsize=(20,10))
    im = ax.contourf(T_span, P_span, res, levels=300, cmap='bone')
    im1 = ax.scatter(
        unnormalize(X_ei[:n_init, 1], (lb_T, ub_T)), unnormalize(X_ei[:n_init, 0], (lb_P, ub_P)),
        color='black', marker='1', s=100
       )
    P, T = X_ei[k, 0], X_ei[k, 1]
    im2 = ax.scatter(
        unnormalize(T, (lb_T, ub_T)), unnormalize(P, (lb_P, ub_P)),
        color='blue', alpha=0.9, marker='p', s=200
        )

    P_pre, T_pre = X_ei[n_init:k, 0], X_ei[n_init:k, 1]
    im3 = ax.scatter(
        unnormalize(T_pre, (lb_T, ub_T)), unnormalize(P_pre, (lb_P, ub_P)),
        color='gray', alpha=1, marker='p', s=50
        )
    ax.set_xlim(lb_T - 1, ub_T + 1)
    ax.set_ylim(lb_P - 1, ub_P + 1)
    #plt.colorbar(im)   

    # find and color best so far
    best_y_idx = Y_ei[:k].argmax()
    P_best, T_best = unnormalize(X_ei[best_y_idx, 0], (lb_P, ub_P)), unnormalize(X_ei[best_y_idx, 1], (lb_T, ub_T))
    im4 = ax.scatter(T_best, P_best, alpha=0.9, marker='s', s=300, facecolors='none', edgecolors='r')

    # plot last move
    P_move, T_move = X_ei[k-1:k+1, 0], X_ei[k-1:k+1, 1]
    im3 = ax.plot(
        unnormalize(T_move, (lb_T, ub_T)), unnormalize(P_move, (lb_P, ub_P)),
        color='gray', alpha=0.6, linewidth=2,
        )
    ax.set_xlim(lb_T - 1, ub_T + 1)
    ax.set_ylim(lb_P - 1, ub_P + 1) 

    # plt.grid()
    plt.savefig(f'artifacts/plots/bo_exploration/exploration_{k}.png', dpi=200, bbox_inches='tight', )
    plt.close()

    # plot actual density curve (according to PC-SAFT)
    fig, ax = plt.subplots(1,1,figsize=(7,4))
    T_span_fine = np.linspace(lb_T, ub_T, 100)
    best_curve = np.array(sim.get_density(T_best.numpy(), T_span_fine)) 

    ax.set_ylim(best_curve.min(), best_curve.max())  
    ax.vlines(T_best, ymin=best_curve.min(), ymax=best_curve.max(), color='red', linestyle='--') # should be T_best, all inverted -_-
    ax.plot(T_span_fine, best_curve, alpha=1, color='black')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Density (kg / m3)")

    #plt.grid()
    plt.savefig(f'artifacts/plots/bo_exploration/density_{k}.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.close()


make_bo_gif(run_id=run_id, frame_folder=frame_folder)