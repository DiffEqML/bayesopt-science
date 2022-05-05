	# nice_fonts = {
	# 		# Use LaTex to write all text
	# 		"text.usetex": False,
	# 		"font.family": "serif",
	# 		"mathtext.fontset": "dejavuserif",
	# 		# Thesis use 16 and 14, respectively
	# 		"axes.labelsize": 16,
	# 		"font.size": 16,
	# 		# Make the legend/label fonts a little smaller
	# 		"legend.fontsize": 14,
	# 		"xtick.labelsize": 16,
	# 		"ytick.labelsize": 16,
	# 		"figure.figsize": [7,5],
	# }

	# mpl.rcParams.update(nice_fonts)


	# start_time = time.time()

	# # config flags
	plot_result = False

	# Specify your two component mixture


	# Main Variables
	# S = len(mixture)
	# Tc = np.zeros(S)
	# Pc = np.zeros(S)
	# MW = np.zeros(S)
	# m = np.zeros(S)
	# sigma = np.zeros(S)
	# eps_k = np.zeros(S)
	# kappa_AB = np.zeros(S)
	# epsAB_k = np.zeros(S)
	# Nassoc = np.zeros(S)

	# Getting the properties


    

	elapsed_time_secs = time.time() - start_time
	print("Execution of solver took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs)))

	if plot_result:
		plt.figure(1)
		plt.plot(T_glob, density_PCSAFT, linestyle='-', color='k')
		plt.ylabel('$\\rho~[kg/m3]$')
		plt.xlabel('$T~[K]$')
		plt.tight_layout()
		plt.show()
