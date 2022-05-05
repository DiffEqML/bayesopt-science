import numpy as np
from numpy.linalg import norm

# Model Constants
kb = 1.38064852*10**(-23)
NA = 6.02214076*10**23
R = 8.314462618

a0 = [0.9105631445, 0.6361281449, 2.6861347891, -26.547362491, 97.759208784, -159.59154087, 91.297774084]
a1 = [-0.3084016918, 0.1860531159, -2.5030047259, 21.419793629, -65.255885330, 83.318680481, -33.746922930]
a2 = [-0.0906148351, 0.4527842806, 0.5962700728, -1.7241829131, -4.1302112531, 13.776631870, -8.6728470368]
b0 = [0.7240946941, 2.2382791861, -4.0025849485, -21.003576815, 26.855641363, 206.55133841, -355.60235612]
b1 = [-0.5755498075, 0.6995095521, 3.8925673390, -17.215471648, 192.67226447, -161.82646165, -165.20769346]
b2 = [0.0976883116, -0.2557574982, -9.1558561530, 20.642075974, -38.804430052, 93.626774077, -29.666905585]

def init_cond(T, P, MW, m, sigma, eps_k, kappa_AB, epsAB_k, Nassoc, x_vec, eta_guess):
	S = len(x_vec)
	M = int(sum(Nassoc))

	# Temperature Dependent Collision Diameter
	d = np.zeros(S)
	for i in range(S):
		d[i] = sigma[i]*(1. - 0.12*np.exp(-3.*eps_k[i]/T))

	mbar = np.dot(x_vec,m)

	a = np.zeros(len(a0))
	b = np.zeros(len(b0))
	for i in range(7):
		a[i] = (a0[i] + (mbar - 1)*a1[i]/mbar + (mbar - 1)*(mbar - 2)*a2[i]/mbar**2)
		b[i] = (b0[i] + (mbar - 1)*b1[i]/mbar + (mbar - 1)*(mbar - 2)*b2[i]/mbar**2)

	apx = np.zeros((len(a0),S))
	bpx = np.zeros((len(b0),S))

	for i in range(7):
		for j in range(S):
			apx[i,j] = m[j]*a1[i]/mbar**2 + m[j]*(3 - 4/mbar)*a2[i]/mbar**2
			bpx[i,j] = m[j]*b1[i]/mbar**2 + m[j]*(3 - 4/mbar)*b2[i]/mbar**2

	# Finding the density multiplier
	dummy_rho = 0
	for i in range(S):
		dummy_rho += x_vec[i]*m[i]*d[i]**3

	# Applying the mixing rules
	sigma_glob = np.zeros((S,S))
	eps_glob = np.zeros((S,S))
	k1 = np.zeros((S,S))
	for i in range(S):
		for j in range(S):
			if i != j:
				# Getting the binary interaction parameter, assuming it is temperature dependent
				k1[i,j] = -28.782/T + 0.0731
			sigma_glob[i,j] = 0.5*(sigma[i] + sigma[j])
			eps_glob[i,j] = (eps_k[i]*eps_k[j]*kb*kb)**0.5*(1. - k1[i,j])

	kappa_AB_glob = np.zeros((M,M))
	epsAB_glob = np.zeros((M,M))
	for i in range(M):
		for j in range(M):
			if i != j:
				# Upper Left Block
				if i < Nassoc[0] and j < Nassoc[0]:
					kappa_AB_glob[i,j] = kappa_AB[0]
					epsAB_glob[i,j] = epsAB_k[0]
				# Upper Right Block
				elif i < Nassoc[0] and j != Nassoc[0]+i:
					kappa_AB_glob[i,j] = (kappa_AB[0]*kappa_AB[1])**0.5*\
						((sigma[0]*sigma[1])**0.5/(0.5*(sigma[0]+sigma[1])))**3
					epsAB_glob[i,j] = (epsAB_k[0] + eps_k[1])*0.5 
				# Lower Left Block
				elif i != Nassoc[0]+j and j < Nassoc[0]:
					kappa_AB_glob[i,j] = (kappa_AB[0]*kappa_AB[1])**0.5*\
						((sigma[0]*sigma[1])**0.5/(0.5*(sigma[0]+sigma[1])))**3
					epsAB_glob[i,j] = (epsAB_k[0] + eps_k[1])*0.5 
			# Lower Right Block
			if i != j and i >= Nassoc[0] and j >= Nassoc[0]:
				kappa_AB_glob[i,j] = kappa_AB[1]
				epsAB_glob[i,j] = epsAB_k[1]


	#############################################################
	################# BEGINNING OF CODE #########################
	#############################################################
	eta = eta_guess
	rho = (6/np.pi)*eta/dummy_rho
	rho_hat = (6/np.pi)*(eta/dummy_rho)*(10**30)/NA

	#############################################################
	############### HARD CHAIN CONTRIBUTION #####################
	#############################################################
	zeta0 = 0
	zeta1 = 0
	zeta2 = 0
	zeta3 = 0
	for i in range(S):
		zeta0 += x_vec[i]*m[i]*rho*np.pi/6
		zeta1 += x_vec[i]*m[i]*d[i]*rho*np.pi/6
		zeta2 += x_vec[i]*m[i]*d[i]**2*rho*np.pi/6
		zeta3 += x_vec[i]*m[i]*d[i]**3*rho*np.pi/6

	g_hs = np.zeros((S,S))
	rho_dg_drho = np.zeros((S,S))
	for i in range(S):
		for j in range(S):
			g_hs[i,j] = 1./(1 - zeta3) + (d[i]*d[j]/(d[i] + d[j]))*3*(zeta2/(1. - zeta3)**2) + \
				((d[i]*d[j]/(d[i] + d[j]))**2)*2*(zeta2**2)/(1. - zeta3)**3
			rho_dg_drho[i,j] = zeta3/(1. - zeta3)**2 + \
				(d[i]*d[j]/(d[i] + d[j]))*(3*zeta2/(1 - zeta3)**2 + 6*zeta2*zeta3/(1 - zeta3)**3) + \
				(d[i]*d[j]/(d[i] + d[j]))**2*(4*zeta2**2/(1 - zeta3)**3 + 6*zeta2**2*zeta3/(1 - zeta3)**4)

	Zhs = zeta3/(1 - zeta3) + 3*zeta1*zeta2/(zeta0*(1 - zeta3)**2) + \
		(3*zeta2**3 - zeta3*zeta2**3)/(zeta0*(1 - zeta3)**3)

	temp_Z = 0
	for i in range(S):
		temp_Z += x_vec[i]*(m[i] - 1)*rho_dg_drho[i,i]/g_hs[i,i]

	Zhc = mbar*Zhs - temp_Z

	# Free Energy
	ahs_tilde = (3*zeta1*zeta2/(1 - zeta3) + zeta2**3/(zeta3*(1 - zeta3)**2) + \
		(zeta2**3/zeta3**2 - zeta0)*np.log(1 - zeta3))/zeta0
	
	sum_ahc = 0.
	for i in range(S):
		sum_ahc += x_vec[i]*(m[i] - 1)*np.log(g_hs[i,i])

	ahc_tilde = mbar*ahs_tilde - sum_ahc

	# Fugacity
	zeta0px = np.zeros(S)
	zeta1px = np.zeros(S)
	zeta2px = np.zeros(S)
	zeta3px = np.zeros(S)
	for i in range(S):
		zeta0px[i] = m[i]*rho*np.pi/6
		zeta1px[i] = m[i]*d[i]*rho*np.pi/6
		zeta2px[i] = m[i]*d[i]**2*rho*np.pi/6
		zeta3px[i] = m[i]*d[i]**3*rho*np.pi/6

	dg_dx = np.zeros((S,S,S))
	for k in range(S):
		for i in range(S):
			for j in range(S):
				dg_dx[k,i,j] = zeta3px[k]/(1 - zeta3)**2 + (d[i]*d[j]/(d[i] + d[j]))*\
					(3*zeta2px[k]/(1 - zeta3)**2 + 6*zeta2*zeta3px[k]/(1 - zeta3)**3) + \
					(d[i]*d[j]/(d[i] + d[j]))**2*(4*zeta2px[k]*zeta2/(1 - zeta3)**3 + \
					6*zeta2**2*zeta3px[k]/(1 - zeta3)**4)

	dahsdx = np.zeros(S)
	dahcdx = np.zeros(S)	
	for i in range(S):
		dahsdx[i] = -zeta0px[i]*ahs_tilde/zeta0 + (3*(zeta1px[i]*zeta2 + zeta1*zeta2px[i])/(1 - zeta3) + \
			3*zeta1*zeta2*zeta3px[i]/(1 - zeta3)**2 + 3*zeta2**2*zeta2px[i]/(zeta3*(1 - zeta3)**2) + \
			zeta2**3*zeta3px[i]*(3*zeta3 - 1)/(zeta3**2*(1 - zeta3)**3) + ((3*zeta2**2*zeta2px[i]*zeta3 - \
			2*zeta2**3*zeta3px[i])/zeta3**3 - zeta0px[i])*np.log(1 - zeta3) + \
			(zeta0 - zeta2**3/zeta3**2)*zeta3px[i]/(1 - zeta3))/zeta0 

		sumF_ahc = 0
		for j in range(S):
			sumF_ahc += x_vec[j]*(m[j] - 1)*dg_dx[i,j,j]/g_hs[j,j]

		dahcdx[i] = m[i]*ahs_tilde + mbar*dahsdx[i] - sumF_ahc
		
	mu_hc = np.zeros(S)
	for i in range(S):
		sumF_muhc = 0
		for j in range(S):
			sumF_muhc += x_vec[j]*dahcdx[j]

		mu_hc[i] = ahc_tilde + Zhc + dahcdx[i] - sumF_muhc

	#############################################################
	############### DISPERSION CONTRIBUTION #####################
	#############################################################
	I1 = 0 
	I2 = 0
	for i in range(7):
		I1 += a[i]*eta**i
		I2 += b[i]*eta**i

	detaI1 = 0
	detaI2 = 0
	for j in range(7):
		detaI1 += a[j]*(j+1)*eta**j
		detaI2 += b[j]*(j+1)*eta**j

	term1 = 0
	term2 = 0
	for i in range(S):
		for j in range(S):
			term1 += x_vec[i]*x_vec[j]*m[i]*m[j]*(eps_glob[i,j]/kb/T)*sigma_glob[i,j]**3
			term2 += x_vec[i]*x_vec[j]*m[i]*m[j]*(eps_glob[i,j]/kb/T)**2*sigma_glob[i,j]**3

	C1 = (1. + mbar*(8*eta - 2*eta**2)/(1. - eta)**4 + (1 - mbar)*(20*eta - 27*eta**2 + \
		12*eta**3 - 2*eta**4)/((1 - eta)*(2 - eta))**2)**-1
	C2 = -C1**2*(mbar*(-4*eta**2 + 20*eta + 8)/(1 - eta)**5 + \
		(1 - mbar)*(2*eta**3 + 12*eta**2 - 48*eta + 40)/((1 - eta)*(2 - eta))**3)

	Zdisp = -2*np.pi*rho*detaI1*term1 - np.pi*rho*mbar*(C1*detaI2 + C2*eta*I2)*term2

	# Free Energy
	adisp_tilde = -2*np.pi*rho*I1*term1 - np.pi*rho*mbar*C1*I2*term2

	# Fugacity
	C1px = np.zeros(S)
	I1px = np.zeros(S)
	I2px = np.zeros(S)
	for i in range(S):
		C1px[i] = C2*zeta3px[i] - C1**2*(m[i]*(8*eta - 2*eta**2)/(1 - eta)**4 - \
			m[i]*(20*eta - 27*eta**2 + 12*eta**3 - 2*eta**4)/((1 - eta)*(2 - eta))**2)
		for j in range(7):
			I1px[i] += a[j]*j*zeta3px[i]*eta**(j-1) + apx[j,i]*eta**j
			I2px[i] += b[j]*j*zeta3px[i]*eta**(j-1) + bpx[j,i]*eta**j
	
	term1px = np.zeros(S)
	term2px = np.zeros(S)
	for i in range(S):
		sum_term1px = 0
		sum_term2px = 0
		for j in range(S):
			sum_term1px += x_vec[j]*m[j]*(eps_glob[i,j]/kb/T)*sigma_glob[i,j]**3
			sum_term2px += x_vec[j]*m[j]*(eps_glob[i,j]/kb/T)**2*sigma_glob[i,j]**3

		term1px[i] = 2*m[i]*sum_term1px
		term2px[i] = 2*m[i]*sum_term2px

	dadispdx = np.zeros(S)
	for i in range(S):
		dadispdx[i] = -2*np.pi*rho*(I1px[i]*term1 + I1*term1px[i]) \
			- np.pi*rho*((m[i]*C1*I2 + mbar*C1px[i]*I2 + mbar*C1*I2px[i])*term2 + \
			mbar*C1*I2*term2px[i]) 

	mu_disp = np.zeros(S)
	for i in range(S):
		sumF_mudisp = 0 
		for j in range(S):
			sumF_mudisp += x_vec[j]*dadispdx[j]

		mu_disp[i] = adisp_tilde + Zdisp + dadispdx[i] - sumF_mudisp

	#############################################################
	############### ASSOCIATION CONTRIBUTION ####################
	#############################################################
	Zassoc = 0.0
	aassoc_tilde = 0
	mu_assoc = np.zeros(S)

	if np.dot(x_vec,Nassoc) != 0:
		delta = np.zeros((M,M))
		ddelta_drho = np.zeros((S,M,M))
		dg_drho = np.zeros((S,S,S))

		# Forming the strength interaction matrix
		for i in range(M):
			for j in range(M):
				if i != j:
					delta[i,j] = kappa_AB_glob[i,j]*(np.exp(epsAB_glob[i,j]/T) - 1)
					if i < Nassoc[0] and j < Nassoc[0]:
						delta[i,j] *= d[0]**3*g_hs[0,0]
					elif i >= Nassoc[0] and j >= Nassoc[0]:
						delta[i,j] *= d[1]**3*g_hs[1,1]
					else:
						delta[i,j] *= ((d[0] + d[1])/2)**3*g_hs[0,1]

		# Finding the derivative of the site-site radial distribution function
		for k in range(S):
			for i in range(S):
				for j in range(S):
					dg_drho[k,i,j] = d[k]**3/(1. - zeta3)**2 + \
						3*(d[i]*d[j]/(d[i] + d[j]))*(d[k]**2/(1 - zeta3)**2 + 2*d[k]**3*zeta2/(1 - zeta3)**3) + \
						2*(d[i]*d[j]/(d[i] + d[j]))**2*(2*d[k]**2*zeta2/(1 - zeta3)**3 + 3*d[k]**3*zeta2**2/(1 - zeta3)**4)

					dg_drho[k,i,j] *= (np.pi/6)*m[k]

		# Finding the derivative of the strength interaction matrix
		for k in range(S):
			for i in range(M):
				for j in range(M):
					if i != j:
						ddelta_drho[k,i,j] = kappa_AB_glob[i,j]*(np.exp(epsAB_glob[i,j]/T) - 1)
						if i < Nassoc[0] and j < Nassoc[0]:
							ddelta_drho[k,i,j] *= d[0]**3*dg_drho[k,0,0]
						elif i >= Nassoc[0] and j >= Nassoc[0]:
							ddelta_drho[k,i,j] *= d[1]**3*dg_drho[k,1,1]
						else:
							ddelta_drho[k,i,j] *= ((d[0] + d[1])/2)**3*dg_drho[k,0,1]

		XA = np.ones(M)*1e-3
		dXA_drho = np.ones((S,M))*1e-3
		resX = np.ones(M)
		resdXA = np.ones((S,M))
		
		# Calculating the mole fractions
		for i in range(M):
			temp_res = 0
			indx = 0
			for j in range(S):
				for k in range(int(Nassoc[j])):
					temp_res += rho*x_vec[j]*XA[indx]*delta[i,indx]
					indx += 1

			resX[i] = XA[i] - (1 + temp_res)**-1
			XA[i] = XA[i] - 0.1*resX[i]

		# Calculating the derivative of mole fractions
		for i in range(S):
			if i == 0:
				indx1 = 0
			if i == 1:
				indx1 = int(Nassoc[i-1])
			for j in range(M):
				sum1 = 0
				for k in range(int(Nassoc[i])):
					sum1 += XA[k+indx1]*delta[j,indx1+k]
				sum2 = 0
				indx2 = 0
				for l in range(S):
					for q in range(int(Nassoc[l])):
						sum2 += rho*x_vec[l]*(delta[j,indx2]*dXA_drho[i,indx2] + XA[indx2]*ddelta_drho[i,j,indx2])
						indx2 += 1

				resdXA[i,j] = dXA_drho[i,j] + (sum1 + sum2)*(XA[j])**2 
				dXA_drho[i,j] = dXA_drho[i,j] - 0.01*resdXA[i,j] 

		# Chemical Potential due to Association
		indx1 = 0
		for i in range(S):
			sum1 = 0
			for ii in range(int(Nassoc[i])):
				sum1 += np.log(XA[indx1]) - XA[indx1]/2
				indx1 += 1
			temp_A = 0
			indx2 = 0
			for j in range(S):
				for k in range(int(Nassoc[j])):
					temp_A += rho*x_vec[j]*dXA_drho[i,indx2]*(1/XA[indx2] - 0.5)
					indx2 += 1
			mu_assoc[i] = sum1 + 0.5*Nassoc[i] + temp_A

		# Residual Free Energy due to Association
		indx1 = 0
		for i in range(S):
			sum1 = 0
			for j in range(int(Nassoc[i])):
				sum1 += np.log(XA[indx1]) - XA[indx1]/2
				indx1 += 1
			sum1 += 0.5*Nassoc[i]
			aassoc_tilde += x_vec[i]*sum1

		Zassoc = np.dot(x_vec, mu_assoc) - aassoc_tilde

	# Compressibility Parameter
	Z = 1 + Zhc + Zdisp + Zassoc

	ares_tilde = ahc_tilde + adisp_tilde + aassoc_tilde

	Pnew = Z*kb*T*rho*(10**10)**3
	error = (Pnew - P)/P

	init_cond.eta = eta

	return error
