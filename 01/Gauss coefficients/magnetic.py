#= Python packages. ===============================================================================
import csv
import numpy as np
from numba import jit
from numpy.linalg import norm
import scipy.special as special

#= Define some global constants. ==================================================================

# Data standard deviation.
sigma_D=2000.0

# Expected energy.
E0=31000.0**2

# Energy standard deviation.
sigma_E=0.1*E0

# Precompute normalisation constants of associated Legendre functions
const=np.zeros((14,14))
for nn in range(0,14):
	for mm in range(0,nn+1):
    		
    	# Compute normalisation constant for the Schmidt quasi-normalised associated Legendre functions.
		if mm==0:
			const[nn,mm]=1.0
		else:
			const[nn,mm]=np.sqrt(2.0*np.math.factorial(nn-mm)/np.math.factorial(nn+mm))


#= Read Gauss coefficients. =======================================================================
def read_coefficients(verbose=False):
	"""
	Read Gauss coefficients from file IGRF13coeffs.csv.
	
	:param verbose: print coefficients to the screen when True
	"""

	# Read csv file that contains the Gauss coefficients.
	c=[]
	with open('./IGRF13coeffs.csv') as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    line_count = 0
	    for row in csv_reader:
	        if line_count < 4:
	            line_count += 1
	        else:
	            if verbose: print(f'\t{row[0]}_{row[1]}_{row[2]}={row[27]} nT')
	            c.append(float(row[27]))
	            line_count += 1

	# Initialise Gauss coefficients.
	g=np.zeros([14,14])
	h=np.zeros([14,14])

	# Assign Gauss coefficients.
	i=0
	for n in np.arange(1,14):

	    g[n,0]=c[i]
	    i+=1

	    for m in np.arange(1,n+1):
	        g[n,m]=c[i]
	        i+=1
	        h[n,m]=c[i]
	        i+=1

	# Return.
	return g,h


#= Precompute Schmidt quasi-normalised associated Legendre functions. =============================
def Pnmi(theta,ell_max=13):
	"""
	Precompute the associated Legendre functions Pnm in order to accelerate Monte Carlo sampling.

	:param theta: array of colatitude values
	:param ell_max: maximum degree of the spherical harmonic expansion

	Returns array of Pnm with additional dimension over observation colatitudes.
	"""

	# Initialise associated Legendre functions.
	Pnmi=np.zeros((ell_max+1,ell_max+1,len(theta)))

	# Loop through colatitudes of observation points.
	for i in range(len(theta)):

		Pmn=special.lpmn(ell_max,ell_max,np.cos(theta[i]))[0]

		# Loop over degrees and orders.
		for n in range(0,ell_max+1):
			for m in range(0,n+1):
				Pnmi[n,m,i]=Pmn[m,n]*const[n,m]

	return Pnmi


#= Assemble radial component of the magnetic field at the surface. ================================
@jit(nopython=True)
def B(phi,theta,g,h,Pnmi,ell_max=13):
	"""
	Radial component of the magnetic field at the observation points.
	
	:param phi: array of longitude values
	:param theta: array of colatitude values
	:param g,h: matrices of Gauss coefficients
	:param ell_max: maximum degree of the spherical harmonic expansion

	Returns data vector.
	"""

	# Number of data points.
	N=len(theta)

	# Initialise magnetic field, d.
	d=np.zeros(N)

	# Spherical harmonics summation.
	for n in range(0,ell_max+1):
		for m in range(0,n+1):
			d+=(n+1.0)*(g[n,m]*np.cos(m*phi)+h[n,m]*np.sin(m*phi))*Pnmi[n,m,:]

	return d


#= Assemble radial component of the magnetic field at the surface. ================================
def B_field(phi,theta,g,h,ell_max=13):
	"""
	Radial component of the magnetic field at the surface.

	:param phi: array of longitude values
	:param theta: array of colatitude values
	:param g,h: matrices of Gauss coefficients
	:param ell_max: maximum degree of the expansion

	Return radial component of geomagnetic field.
	"""

	# Initialise magnetic field, y, and precompute cos(theta).
	y=np.zeros((len(theta),len(phi)))
	z=np.cos(theta)

    # Spherical harmonics summation.
	for i in range(len(theta)):
		Pmn=special.lpmn(len(g)-1,len(g)-1,z[i])[0]
		for nn in range(0,ell_max+1):
			for mm in range(0,nn+1):
    			
    			# Add contributions.
				y[i,:]+=(nn+1.0)*(g[nn,mm]*np.cos(mm*phi)+h[nn,mm]*np.sin(mm*phi))*Pmn[mm,nn]*const[nn,mm]

	return y


#= Prior probability density in data space. =======================================================
def prior_data(d_obs,d):
	"""
	:param d_obs: observed data vector
	:param d: simulated data vector
	"""
    
    return np.exp(-0.5*norm(d-d_obs)**2/sigma_D**2)


#= Log Prior probability density in data space. ===================================================
def log_prior_data(d_obs,d):
	"""
	:param d_obs: observed data vector
	:param d: simulated data vector
	"""

	return -0.5*norm(d-d_obs)**2/sigma_D**2


#= Prior probability density in model space. ======================================================
def prior_model(g_in,h_in,ell_max):
	"""
	:param g_in, h_in: Gauss coefficient arrays
	:param ell_max: maximum harmonic degree
	"""

	E=0.0

	# Add energy contributions from included degrees.
	for n in range(0,ell_max+1):
			for m in range(0,n+1):
				E+=(g_in[n,m]**2+h_in[n,m]**2)

	return np.exp(-0.5*(E-E0)**2/sigma_E**2)


#= Log prior probability density in model space. ==================================================
def log_prior_model(g_in,h_in,ell_max):
	"""
	:param g_in, h_in: Gauss coefficient arrays
	:param ell_max: maximum harmonic degree
	"""
    
	E=0.0

	# Add energy contributions from included degrees.
	for n in range(0,ell_max+1):
			for m in range(0,n+1):
				E+=(g_in[n,m]**2+h_in[n,m]**2)

	return -0.5*(E-E0)**2/sigma_E**2


#= Posterior probability density (missing the evidence). ==========================================
def posterior(d_obs,phi_obs,theta_obs,g,h,ell_max):
	"""
	:param d_obs: observed data vector
	:param phi_obs, theta_obs: longitudes and colatitudes of observation points
	:param g,h: Gauss coefficient arrays
	:param ell_max: maximum harmonic degree
	"""

	d=B(phi_obs,theta_obs,g,h,ell_max)
	p=prior_model(g,h,ell_max)*prior_data(d,d_obs)
	return p


#= Log posterior probability density (missing the evidence). ======================================
def log_posterior(d_obs,phi_obs,theta_obs,g,h,Pnmi,ell_max):
	"""
	:param d_obs: observed data vector
	:param phi_obs, theta_obs: longitudes and colatitudes of observation points
	:param g,h: Gauss coefficient arrays
	:param ell_max: maximum harmonic degree
	"""

	d=B(phi_obs,theta_obs,g,h,Pnmi,ell_max)
	p=log_prior_model(g,h,ell_max)+log_prior_data(d,d_obs)
	return p


#= Gradient of logarithmic posterior. =============================================================
@jit(nopython=True)
def grad_posterior(d_obs,phi_obs,theta_obs,g,h,Pnmi,ell_max):
	"""
	Gradient of the logarithmic posterior.

	:param n,m: degree and order of derivative component
	:param coeff: choose between 'g' and 'h' for one of the class of Gauss coefficients
	:param d_obs: observed data vector
	:param phi_obs, theta_obs: longitudes and colatitudes of observation points [rad]
	:param g,h: Gauss coefficients
	:param Pnmi: precomputed Schmidt quasi-normalised associated Legendre functions
	:param ell_max: maximum harmonic degree

	Returns gradient of the logarithmic posterior.
	"""

	# Simulated data.
	d=B(phi_obs,theta_obs,g,h,Pnmi,ell_max)

	# Simulated energy.
	E=0.0

	# Add energy contributions from included degrees.
	for n in range(0,ell_max+1):
			for m in range(0,n+1):
				E+=(g[n,m]**2+h[n,m]**2)

	# Initialise gradient.
	Nc=np.shape(g)[0]
	grad=np.zeros((2,Nc,Nc))

    # Data part of the gradient. ------------------------------------------------------------------

	# Loop over degrees and orders.
	for n in range(1,ell_max+1):
		for m in range(0,n+1):

			# g component
			grad[0,n,m]+=np.sum((n+1)*np.cos(m*phi_obs)*Pnmi[n,m,:]*(d_obs-d))
			# h component
			grad[1,n,m]+=np.sum((n+1)*np.sin(m*phi_obs)*Pnmi[n,m,:]*(d_obs-d))

	# Divide by data variance.
	grad=grad/(sigma_D**2)

    # Energy part of the gradient. ----------------------------------------------------------------

	for n in range(0,ell_max+1):
			for m in range(0,n+1):

    			# g component
				grad[0,n,m]-=2.0*g[n,m]*(E-E0)/(sigma_E**2)
    			# h component
				grad[1,n,m]-=2.0*h[n,m]*(E-E0)/(sigma_E**2)


	# Return.
	return grad








