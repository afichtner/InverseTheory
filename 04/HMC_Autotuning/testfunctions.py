import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import scipy.sparse as sp
import numpy as np

class f:
    
    #===============================================================================
    # Initialisation.
    #===============================================================================
    def __init__(self,dim,function='gauss'):
        """
        Initialise test function class.
        
        :param dim: Model space dimension.
        :param function: Type of test function ('gauss', 'rastrigin', 'styblinski')
        """

        self.dim=dim
        self.function=function
    
        # Gaussian with predefined covariance matrix. =============================
        if function=='gauss_1':

            # This is a Gaussian with diagonal covariance matrix. The diagonal entries grow linearly.
            
            offsets=np.array([0])
            data=np.zeros((1,dim))
            for i in range(dim): data[0,i]=0.01+0.99*float(i*dim/(dim*(dim-1)))
            self.C=sp.dia_matrix((data,offsets),shape=(dim,dim))
            self.Cinv=sla.inv(self.C.tocsc())
            
        if function=='gauss_2':

            # This is a band-diagonal Gaussian where the diagonal entries are 1, and the off-diagonal entries decrease slowly decrease.

            offsets=np.array([0,1,2,3,-1,-2,-3])
            data=np.zeros((7,dim))
            data[0,:]=np.ones(dim)
            data[1,:]=0.25*np.ones(dim)
            data[2,:]=0.15*np.ones(dim)
            data[3,:]=0.10*np.ones(dim)
            data[4,:]=0.25*np.ones(dim)
            data[5,:]=0.15*np.ones(dim)
            data[6,:]=0.10*np.ones(dim)
            self.C=sp.dia_matrix((data,offsets),shape=(dim,dim))
            self.Cinv=sla.inv(self.C)
           
            
        # Modified Rastrigin or Styblinski-Tang functions==========================
        if function=='rastrigin_0' or function=='styblinski_0':

            # These are modified Styblinski-Tang and Rastrigin functions where the parameters are scaled through the application of a diagonal transformation matrix with exponentially decaying parameters.
            
            offsets=np.array([0])
            data=np.zeros((1,dim))

            for i in range(dim): data[0,i]=np.exp(-3.0*float(i)/float(dim))+0.1

            self.A=sp.dia_matrix((data,offsets),shape=(dim,dim))

        # Modified Styblinski-Tang function =======================================
        if function=='styblinski_1':

            # The same as above but for a band-diagonal transformation that introduces inter-parameter trade-offs.
            
            offsets=np.array([0,1,-1])
            data=np.zeros((3,dim))

            for i in range(dim): data[0,i]=1.0-0.85*float(i)/float(dim)
            data[1,:]=0.1*data[0,i]
            data[2,:]=0.1*data[0,i]

            self.A=sp.dia_matrix((data,offsets),shape=(dim,dim))

        # Modified Styblinski-Tang function =======================================
        if function=='styblinski_2' or function=='rastrigin_1':

            # The same as above but with a completely filled transformation matrix. Only works for small examples.

            Ad=np.zeros((dim,dim))
            for i in range(dim): 
                Ad[i,i]=np.exp(-3.0*float(i)/float(dim))
                Ad[i,dim-1-i]=0.1
            self.A=sp.coo_matrix(Ad)
        

    #===============================================================================
    # Potential energy.
    #===============================================================================   
    def U(self,m):
        """
        Compute the potential energy of the distributions defined above.
        
        :param m: Model paramter vector.
        """

        n=np.float(self.dim)
        
        # Gaussian with predefined covariance matrix. =============================
        if self.function=='gauss_1' or self.function=='gauss_2':
            return 0.5*np.dot(m,self.Cinv.dot(m))
        
        # Modified Rastrigin function =============================================
        if self.function=='rastrigin_0' or self.function=='rastrigin_1':
            x=self.A.dot(m)
            return 10.0*n+np.sum(x*x-1.0*np.cos(2.0*np.pi*x))
        
        # Modified Styblinski-Tang function. ======================================
        if self.function=='styblinski_0' or self.function=='styblinski_1':
            x=self.A.dot(m)
            return (1.0/(39.16599))*(0.5*np.sum(x**4-16.0*x**2+5.0*x)+39.16599*n)

        # Modified Styblinski-Tang function. ======================================
        if self.function=='styblinski_2':
            x=self.A.dot(m)
            return (1.0/(39.16599))*(0.5*np.sum(x**4-16.0*x**2+5.0*x)+39.16599*n)+0.5*np.dot(m,m)/(100.0**2)
    
    #===============================================================================
    # Gradient of potential energy.
    #===============================================================================
    def J(self,m):
        """
        Compute the gradient of the potential energy for the distributions defined above.
        """

        n=np.float(self.dim)
        
        # Gaussian with predefined covariance matrix. =============================
        if self.function=='gauss_1' or self.function=='gauss_2':
            return self.Cinv.dot(m)
        
        # Modified Rastrigin function =============================================
        if self.function=='rastrigin_0' or self.function=='rastrigin_1':
            x=self.A.dot(m)
            d=2.0*x+2.0*np.pi*np.sin(2.0*np.pi*x)
            return self.A.dot(d)
        
        # Modified Styblinski-Tang function. ======================================
        if self.function=='styblinski_0' or self.function=='styblinski_1':
            x=self.A.dot(m)
            d=4.0*x**3-32.0*x+5
            return (1.0/(39.16599))*0.5*self.A.dot(d)

        # Modified Styblinski-Tang function. ======================================
        if self.function=='styblinski_2':
            x=self.A.dot(m)
            d=4.0*x**3-32.0*x+5
            return (1.0/(39.16599))*0.5*self.A.dot(d)+m/(100.0**2)

    #===============================================================================
    # Plot U.
    #===============================================================================
    def plotU(self,dim,dim1,dim2,m1_min=-0.1,m1_max=0.1,m2_min=-1.0,m2_max=1.0):
        """
        Plot probability density without invoking the plt.show() command. Useful, 
        for instance, to plot Hamiltonian trajectories.
        """

        m=np.zeros(dim)

        m1=np.linspace(m1_min,m1_max,100)
        m2=np.linspace(m2_min,m2_max,100)

        m1v,m2v=np.meshgrid(m1,m2)
        UU=np.zeros(np.shape(m1v))

        # March through grid to evaluate function.
        for i in range(len(m1)):
            for j in range(len(m2)):
                
                m[dim1]=m1[i]
                m[dim2]=m2[j]
                UU[j,i]=self.U(m)

        # Subtract minimum for better plotting.
        UU-=np.min(np.abs(UU))

        # Plot.
        plt.subplots(1, figsize=(12,12))
        plt.pcolor(m1v,m2v,np.exp(-UU),cmap='Greys')
        plt.colorbar()
        plt.contour(m1v,m2v,np.exp(-UU),10,colors='k')
        plt.xlabel('dimension %d' % dim1)
        plt.ylabel('dimension %d' % dim2)
