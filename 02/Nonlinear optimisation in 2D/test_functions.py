import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.special as special

from matplotlib import rc

#==========================================================
#- Functions. ---------------------------------------------
#==========================================================

def f(x, y, function="rosenbrock", plot=False):

	#- Rosenbrock function. -------------------------------

	if function=="rosenbrock":

		a=1.0
		b=100.0

		z=(a-x)**2+b*(y-x**2)**2

	#- Quadratic function. --------------------------------

	if function=="quadratic":

		z=x**2+y**2+x*y

	#- Himmelblau function. -------------------------------

	if function=="himmelblau":

		z=(x**2+y-11.0)**2 + (x+y**2-7.0)**2

	#- Bazaraa-Shetty function. ---------------------------

	if function=="bazaraa-shetty":

		z=(x-2.0)**4 + (x-2.0*y)**2


	#- Plot if wanted. ------------------------------------

	if plot==True:


		plt.contourf(x,y,z,40,cmap=plt.cm.Greys)
		plt.colorbar(shrink=0.75)
		plt.contour(x,y,z,40,colors=('k',))
		plt.xlabel(r'$m_1$')
		plt.ylabel(r'$m_2$')
		plt.tight_layout()
		
	#- Return. --------------------------------------------

	return z


#==========================================================
#- Derivatives. -------------------------------------------
#==========================================================

def J(x, y, function="rosenbrock"):

	#- Rosenbrock function. -------------------------------

	if function=="rosenbrock":

		a=1.0
		b=100.0

		Jx=-2.0*(a-x)-4.0*b*x*(y-x**2)
		Jy=2.0*b*(y-x**2)


	#- Quadratic function. --------------------------------

	if function=="quadratic":

		Jx=2.0*x+y
		Jy=2.0*y+x

	#- Himmelblau function. -------------------------------

	if function=="himmelblau":

		Jx=4.0*x*(x**2+y-11.0) + 2.0*(x+y**2-7.0)
		Jy=2.0*(x**2+y-11.0) + 4.0*y*(x+y**2-7.0)

	#- Bazaraa-Shetty function. ---------------------------

	if function=="bazaraa-shetty":

		Jx=4.0*(x-2.0)**3 + 2.0*(x-2.0*y)
		Jy=-4.0*(x-2.0*y)


	#- Return. --------------------------------------------

	return np.matrix([[Jx],[Jy]])



#==========================================================
#- Hessians. ----------------------------------------------
#==========================================================

def H(x, y, function="rosenbrock"):

	#- Rosenbrock function. -------------------------------

	if function=="rosenbrock":

		a=1.0
		b=100.0

		Hxx=2.0-4.0*b*y+12.0*b*x**2
		Hxy=-4.0*b*x
		Hyy=2.0*b


	#- Quadratic function. --------------------------------

	if function=="quadratic":

		Hxx=2.0
		Hyy=2.0
		Hxy=1.0

	#- Himmelblau function. -------------------------------

	if function=="himmelblau":

		Hxx=12.0*x**2 + 4.0*y - 42.0
		Hyy=12.0*y**2 + 4.0*x - 26.0
		Hxy=4.0*x + 4.0*y

	#- Bazaraa-Shetty function. ---------------------------

	if function=="bazaraa-shetty":

		Hxx=12.0*(x-2.0)**2 + 2.0
		Hyy=8.0
		Hxy=-4.0

	#- Return. --------------------------------------------
	
	return np.matrix([[Hxx, Hxy],[Hxy, Hyy]])


