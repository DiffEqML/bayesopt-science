import numpy as np
from src.Species import get_properties
from src.PCSAFT import get_fugacity
import matplotlib

import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from datetime import timedelta


class MixtureModel:
	def __init__(self, mixture = ['CO2', 'Xe'], concentrations = [1., 0.]):
		assert sum(concentrations) == 1.

		S = len(mixture)
		Tc = np.zeros(S)
		Pc = np.zeros(S)
		MW = np.zeros(S)
		m = np.zeros(S)
		sigma = np.zeros(S)
		eps_k = np.zeros(S)
		kappa_AB = np.zeros(S)
		epsAB_k = np.zeros(S)
		Nassoc = np.zeros(S)

		for s, species in enumerate(mixture):
			Tc[s], Pc[s], MW[s], m[s], sigma[s], eps_k[s], kappa_AB[s], epsAB_k[s], Nassoc[s] = get_properties(species)

		self.S = S
		self.Tc = Tc
		self.Pc = Pc
		self.MW = MW
		self.m = m
		self.sigma = sigma
		self.eps_k = eps_k
		self.kappa_AB = kappa_AB
		self.epsAB_k = epsAB_k
		self.Nassoc = Nassoc

		self.concentrations = concentrations


	def get_density(self, P_span=100, T_span=np.arange(250, 350+1, 1)):

		density_PCSAFT = []
		density_CP = []
		enthalpy_PCSAFT = []
		enthalpy2_PCSAFT = []
		cp_PCSAFT = []
		entropy_PCSAFT = []
		gibbs_PCSAFT = []
		fugacity_PCSAFT = []

		for ii in range(len(T_span)):

			# P in bar
			P = P_span*10**5
			T = T_span[ii]

			assert self.concentrations[0] == 1, "Simulator tested only for pure CO2 mixtures"

			# The code runs twice, each time with a different guess and chooses the physical result
			dens1, fug1 = get_fugacity(T, P, self.MW, self.m, self.sigma, self.eps_k, self.kappa_AB, self.epsAB_k, self.Nassoc, self.concentrations, 10**(-10))
			dens2, fug2 = get_fugacity(T, P, self.MW, self.m, self.sigma, self.eps_k, self.kappa_AB, self.epsAB_k, self.Nassoc, self.concentrations, 0.5)
			if fug1[0] < fug2[0]: 
				density_PCSAFT.append(dens1)
			else:
				density_PCSAFT.append(dens2)

		return density_PCSAFT
