# This example is supposed to run with "1-HMC-ND - Autotune BFGS".
# It uses a 100-dimensional Gaussian as test function and nicely establishes the benefit of BFGS updating.


import numpy as np

def input_parameters():

	# General setup. ===============================================================================

	# Test function.
	test_function='gauss_1'

	# Model space dimension.
	dim=100

	# Total number of samples.
	N=10000

	# Maximum number of leap-frog iterations.
	Nit=7

	# Leap-frog time step.
	dt=0.04

	# Initial model (deterministic or randomly chosen).
	#m0=np.random.randn(dim)
	m0=np.zeros(dim)

	# Initial inverse mass matrix.	
	Minv=1.0*np.identity(dim)
	
	#Minv=np.identity(dim)
	#for i in range(dim): Minv[i,i]=0.01+0.99*float(i*dim/(dim*(dim-1)))

	# Auto-tuning parameters. =====================================================================

	# Autotuning ('BFGS', 'SR1', False). SR1 is only available without factorisation.
	autotune='BFGS'

	# Number of vectors to store for computation of Hessian approximation.
	ell=10

	# Updating interval. Number of samples after which new vectors are added to the Hessian approximation. 
	update_interval=1

	# Enable iterative updating of the initial matrix factor S0.
	preconditioning=True

	# Minimum allowable diagonal entry of S0, used for stabilisation.
	S0_min=0.0001

	# Output input. ===============================================================================

	# Screen output interval.
	plot_interval=1000

	# Dimensions for posterior analysis
	dimension1=0
	dimension2=99

	# Axis ranges for plotting of trajectories, etc.
	m1_min=-0.1
	m1_max=0.1
	m2_min=-1.0
	m2_max=1.0
	

	# Some screen output. =========================================================================
	
	print('test function: %s' % test_function)
	print('model space dimension: %d' % dim)
	print('number of samples: %d' % N)
	print('maximum number of leapfrog iterations: %d' % Nit)
	print('leapfrog time step: %f' % dt)
	print('auto-tuning: %s' % autotune)
	print('maximum number of LF-BFGS vectors: %d' % ell)
	print('updating interval for LF-BFGS: %d' % update_interval)
	print('preconditioning of initial diagonal matrix: %s' % preconditioning)
	print('minimum SO: %f\n' % S0_min)

	print('initial model:')
	print(m0)
	print('\n')

	print('initial inverse mass matrix:')
	print(Minv)
	print('\n')
	
	print('plotting interval: %d' % plot_interval)
	print('dimension 1 for analysis: %d' % dimension1)
	print('dimension 2 for analysis: %d' % dimension2)


	# Return.
	return test_function, dim, N, Nit, dt, m0, Minv, autotune, ell, update_interval, preconditioning, S0_min, plot_interval, dimension1, dimension2
