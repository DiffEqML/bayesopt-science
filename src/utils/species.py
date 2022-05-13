######################################
############# Dictionary #############
######################################
# Tc: Critical Temperature (K)
# Pc: Critical Pressure (Pa)
# Vc: Critical Molar Specific Volume (1/mol)
# MW: Molar Weight (g/mol)
# omega: Acentric Factor
# m: Segments Per Chain
# sigma: Segment Diameter (A)
# eps_k: Potential Depth (K)
# sigma_coll: Molecular Collision Diameter (A)
# dipole: Dipole Moment (D)
######################################

def get_properties(species):
	if species == 'CO2':
		Tc = 304.2 
		Pc = 73.8e5
		MW = 44.01 
		# Gross Sadowski 2001
		m = 2.0729
		sigma = 2.7852 
		eps_k = 169.21
		kappa_AB = 0.
		epsAB_k = 0.
		# Number of sites
		Nassoc = 0

		# 2B Sites
		# m = 2.2414
		# sigma = 2.713
		# eps_k = 159.00
		# kappa_AB = 0.0283
		# epsAB_k = 512.88
		# Nassoc = 2

	if species == 'Xe':
		Tc = 289.733 
		Pc = 58.42e5
		MW = 131.293
		# Xenon PC-SAFT Google
		m = 0.9633
		sigma = 3.9903 
		eps_k = 232.050
		kappa_AB = 0.
		epsAB_k = 0.
		# Number of sites
		Nassoc = 0

		# 2B Sites
		# m = 2.2414
		# sigma = 2.713
		# eps_k = 159.00
		# kappa_AB = 0.0283
		# epsAB_k = 512.88
		# Nassoc = 2
	
	if species == 'H2O':
		Tc = 647.14
		Pc = 22.12e6
		MW = 18.015
		# Gross Sadowski 2002
		m = 1.09528
		sigma = 2.88980
		eps_k = 365.956
		kappa_AB = 0.03487
		epsAB_k = 2515.7
		# Number of sites
		Nassoc = 2

		# 4C Sites
		# m = 2.1945
		# sigma = 2.229
		# eps_k = 141.66
		# kappa_AB = 0.2039
		# epsAB_k = 1804.17
		# Nassoc = 4

	if species == 'CH4':
		Tc = 190.6
		Pc = 46.0e5
		MW = 16.043
		m = 1
		sigma = 3.7039
		eps_k = 150.03
		kappa_AB = 0.
		epsAB_k = 0.
		# Number of sites
		Nassoc = 0

	if species == 'Hex':
		Tc = 507.44
		Pc = 3.031e6
		MW = 86.177
		m = 3.0576
		sigma = 3.7983
		eps_k = 236.77
		kappa_AB = 0.
		epsAB_k = 0.
		# Number of sites
		Nassoc = 0

	if species == 'Dec':
		Tc = 617.8
		Pc = 21.1e5
		MW = 142.285
		m = 4.6627
		sigma = 3.8384
		eps_k = 243.87
		kappa_AB = 0.
		epsAB_k = 0.
		# Number of sites
		Nassoc = 0

	return Tc, Pc, MW, m, sigma, eps_k, kappa_AB, epsAB_k, Nassoc

	